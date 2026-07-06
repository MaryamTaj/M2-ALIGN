"""Stage 2 WIT training: image + multilingual caption → English caption.

Trains the Mapping layer of :class:`VisualMindMerger` on WIT image-text pairs
produced by :mod:`load_image`.  NLLB encoder and Qwen3-VL (including the
vision tower) are frozen; only the Mapping MLP is updated.

Input JSONL fields (from load_image.py):
    image_url, caption_text, target_caption, source_language, nllb_lang_tag

Usage
-----
    python train.py \\
        --data-path ./data/wit_pairs.jsonl \\
        --output-dir ./outputs/wit \\
        --stage1-mapping-ckpt ../Stage1/outputs/.../pytorch_model.bin \\
        --mt-path facebook/nllb-200-distilled-600M \\
        --llm-path Qwen/Qwen3-VL-8B-Instruct \\
        --image-cache-dir ./data/image_cache
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import os
import random
from datetime import datetime
from functools import partial

import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, NllbTokenizer

from model import VisualMindMerger

try:
    import wandb
except ImportError:
    wandb = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("wit_train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"train_wit_{ts}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WITDataset(Dataset):
    """Loads WIT JSONL pairs and fetches images from URLs with local caching.

    Args:
        rows: List of dicts from the WIT JSONL file.
        cache_dir: Directory for caching downloaded images as JPEG files.
    """

    def __init__(self, rows: list[dict], cache_dir: str) -> None:
        self.rows = rows
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict | None:
        row = self.rows[idx]
        image = self._load_image(row["image_url"])
        if image is None:
            return None
        return {
            "image": image,
            "caption_text": row["caption_text"],
            "target_caption": row["target_caption"],
            "nllb_lang_tag": row["nllb_lang_tag"],
        }

    def _load_image(self, url: str) -> Image.Image | None:
        cache_key = hashlib.sha1(url.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, cache_key + ".jpg")
        if os.path.exists(cache_path):
            try:
                return Image.open(cache_path).convert("RGB")
            except Exception:
                pass
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "M2-ALIGN/1.0"})
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img.save(cache_path, format="JPEG", quality=85)
            return img
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_wit(
    batch: list[dict | None],
    processor: AutoProcessor,
    min_pixels: int,
    max_pixels: int,
) -> dict | None:
    """Collate a batch of WIT samples, processing images with the Qwen3-VL processor.

    Setting ``min_pixels == max_pixels`` forces all images to the same patch
    count, eliminating the need to pad the visual token dimension across the
    batch.

    Args:
        batch: List of sample dicts (or ``None`` for failed image loads).
        processor: Qwen3-VL AutoProcessor.
        min_pixels: Minimum total pixel count for image resizing.
        max_pixels: Maximum total pixel count for image resizing.

    Returns:
        Batched dict or ``None`` if all images in the batch failed to load.
    """
    valid = [x for x in batch if x is not None]
    if not valid:
        return None

    images = [x["image"] for x in valid]
    image_inputs = processor(
        images=images,
        return_tensors="pt",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return {
        "pixel_values": image_inputs["pixel_values"],
        "image_grid_thw": image_inputs["image_grid_thw"],
        "caption_texts": [x["caption_text"] for x in valid],
        "target_captions": [x["target_caption"] for x in valid],
        "nllb_lang_tags": [x["nllb_lang_tag"] for x in valid],
    }


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def mt_input_features(
    texts: list[str],
    nllb_lang_tags: list[str],
    tokenizer_mt: NllbTokenizer,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise multilingual captions with the NLLB tokenizer.

    Sets ``src_lang`` per example from the per-row NLLB language tag.

    Args:
        texts: Source-language caption strings.
        nllb_lang_tags: NLLB language tags parallel to *texts*
            (e.g. ``"fra_Latn"``).
        tokenizer_mt: An instantiated :class:`NllbTokenizer`.
        max_seq_len: Maximum token length; longer strings are right-truncated.
        device: Target device.

    Returns:
        ``(input_ids, attention_mask)`` of shape ``[B, seq_len]``.
    """
    ids, masks = [], []
    for text, tag in zip(texts, nllb_lang_tags):
        tokenizer_mt.src_lang = tag
        enc = tokenizer_mt(text, truncation=True, max_length=max_seq_len, padding=False)
        ids.append(enc["input_ids"])
        masks.append(enc["attention_mask"])

    max_len = max(len(x) for x in ids)
    pad_id = tokenizer_mt.pad_token_id
    for i in range(len(ids)):
        while len(ids[i]) < max_len:
            ids[i].append(pad_id)
            masks[i].append(0)

    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def llm_input_features(
    texts: list[str],
    tokenizer_llm: AutoTokenizer,
    max_seq_len: int,
    add_bos: bool,
    add_eos: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise strings with the LLM tokenizer.

    Args:
        texts: Strings to tokenize.
        tokenizer_llm: Qwen3-VL tokenizer.
        max_seq_len: Maximum token length.
        add_bos: Whether to prepend BOS.
        add_eos: Whether to append EOS.
        device: Target device.

    Returns:
        ``(input_ids, attention_mask)`` of shape ``[B, seq_len]``.
    """
    tokenizer_llm.add_bos_token = add_bos
    tokenizer_llm.add_eos_token = add_eos
    enc = tokenizer_llm(
        texts,
        truncation=True,
        max_length=max_seq_len,
        padding=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, model: VisualMindMerger, step: int, loss: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"step": step, "loss": loss, "model_state_dict": model.mapping.state_dict()},
        path,
    )


# ---------------------------------------------------------------------------
# W&B helpers (identical pattern to existing Stage 2 train.py)
# ---------------------------------------------------------------------------

def _load_wandb_key_from_tokens() -> bool:
    for candidate in (
        os.path.join(os.getcwd(), ".tokens"),
        os.path.join(os.path.dirname(os.getcwd()), ".tokens"),
    ):
        if not os.path.isfile(candidate):
            continue
        with open(candidate, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "WANDB_API_KEY":
                    value = value.strip().strip('"').strip("'")
                    if value:
                        os.environ["WANDB_API_KEY"] = value
                        return True
    return False


def _init_wandb_or_disable(args, config: dict) -> bool:
    if not args.use_wandb or wandb is None:
        return False
    mode = args.wandb_mode.lower()
    if mode in {"auto", "online"} and not os.environ.get("WANDB_API_KEY"):
        _load_wandb_key_from_tokens()
    if mode == "offline" or (mode == "auto" and not os.environ.get("WANDB_API_KEY")):
        os.environ["WANDB_MODE"] = "offline"
    try:
        if os.environ.get("WANDB_API_KEY"):
            wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            config=config,
        )
        return True
    except Exception as exc:
        print(f"wandb init failed ({exc}); disabling.")
        return False


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main(args, logger: logging.Logger) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("train_wit.py requires CUDA.")

    # Load data
    rows: list[dict] = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    random.shuffle(rows)
    split = int(len(rows) * (1.0 - args.val_ratio))
    train_rows, val_rows = rows[:split], rows[split:]
    logger.info("Dataset: train=%d val=%d", len(train_rows), len(val_rows))

    # Tokenizers and processor
    tokenizer_mt = NllbTokenizer.from_pretrained(args.mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(args.llm_path, use_fast=True)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"

    processor = AutoProcessor.from_pretrained(args.llm_path)

    # Model
    model = VisualMindMerger(
        mt_path=args.mt_path,
        llm_path=args.llm_path,
        max_gen_len=args.max_gen_len,
        llm_bos_token_id=tokenizer_llm.bos_token_id,
        llm_pad_token_id=tokenizer_llm.pad_token_id,
        local_files_only=args.local_files_only,
    ).to(device)

    if args.stage1_mapping_ckpt:
        ckpt = torch.load(args.stage1_mapping_ckpt, map_location="cpu")
        model.mapping.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info("Loaded Stage 1 mapping: %s", args.stage1_mapping_ckpt)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    # DataLoaders — collate_fn needs processor and pixel budget
    _collate = partial(
        collate_wit,
        processor=processor,
        min_pixels=args.visual_pixels,
        max_pixels=args.visual_pixels,
    )
    train_loader = DataLoader(
        WITDataset(train_rows, args.image_cache_dir),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        WITDataset(val_rows, args.image_cache_dir),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=args.num_workers,
    )

    use_wandb = _init_wandb_or_disable(args, {
        "stage": "stage2_wit",
        "mt_path": args.mt_path,
        "llm_path": args.llm_path,
        "lr": args.lr,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "grad_accum": args.grad_accum,
        "visual_pixels": args.visual_pixels,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
    })

    best_val = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        running, steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch={epoch}")
        optimizer.zero_grad(set_to_none=True)

        for batch in pbar:
            if batch is None:
                continue

            pixel_values = batch["pixel_values"].to(device)
            image_grid_thw = batch["image_grid_thw"].to(device)

            input_ids_mt, mask_mt = mt_input_features(
                batch["caption_texts"],
                batch["nllb_lang_tags"],
                tokenizer_mt,
                args.max_mt_seq_len,
                device,
            )

            labels, mask_label = llm_input_features(
                batch["target_captions"], tokenizer_llm, args.max_gen_len,
                add_bos=False, add_eos=True, device=device,
            )

            loss = model(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                input_ids_mt=input_ids_mt,
                attention_mask_mt=mask_mt,
                labels=labels,
                mask_label=mask_label,
            )

            (loss / args.grad_accum).backward()
            if (global_step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item()
            steps += 1
            global_step += 1
            pbar.set_postfix(loss=f"{running / steps:.4f}")
            if use_wandb:
                wandb.log({"train/loss": running / steps, "train/global_step": global_step})

        # Validation
        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                pixel_values = batch["pixel_values"].to(device)
                image_grid_thw = batch["image_grid_thw"].to(device)
                input_ids_mt, mask_mt = mt_input_features(
                    batch["caption_texts"], batch["nllb_lang_tags"],
                    tokenizer_mt, args.max_mt_seq_len, device,
                )
                labels, mask_label = llm_input_features(
                    batch["target_captions"], tokenizer_llm, args.max_gen_len,
                    add_bos=False, add_eos=True, device=device,
                )
                loss = model(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    input_ids_mt=input_ids_mt,
                    attention_mask_mt=mask_mt,
                    labels=labels,
                    mask_label=mask_label,
                )
                val_loss += loss.item()
                val_steps += 1

        val_loss /= max(1, val_steps)
        val_ppl = math.exp(min(20.0, val_loss))
        logger.info("Epoch %d | val_loss=%.4f | val_ppl=%.4f", epoch, val_loss, val_ppl)
        if use_wandb:
            wandb.log({"eval/loss": val_loss, "eval/ppl": val_ppl, "train/global_step": global_step})

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.output_dir, "mapping", "pytorch_model.bin")
            save_checkpoint(ckpt_path, model, global_step, val_loss)
            logger.info("Saved best checkpoint (val_loss=%.4f) → %s", val_loss, ckpt_path)

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 WIT visual grounding training.")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to wit_pairs.jsonl from load_wit_data.py.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for checkpoints.")
    parser.add_argument("--stage1-mapping-ckpt", type=str, default=None,
                        help="Stage 1 mapping checkpoint to warm-start from.")
    parser.add_argument("--mt-path", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--image-cache-dir", type=str, default="./data/image_cache")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-mt-seq-len", type=int, default=512,
                        help="NLLB max token length for WIT captions.")
    parser.add_argument("--max-gen-len", type=int, default=512,
                        help="Max token length for the target English caption.")
    parser.add_argument(
        "--visual-pixels", type=int, default=256 * 256,
        help=(
            "Total pixel budget per image (min_pixels = max_pixels = this value). "
            "Setting min=max forces a fixed visual token count across the batch, "
            "avoiding padding in the visual dimension."
        ),
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="m2align-stage2-wit")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(os.path.dirname(__file__), "logs"))
    main(args, logger)
