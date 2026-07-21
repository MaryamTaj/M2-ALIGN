"""Stage 3b VQA augmentation training for AugmentedVisualMindMerger.

Trains the Mapping layer of :class:`AugmentedVisualMindMerger` on GQA-derived
VQA data (see :mod:`load_vqa_data`) translated into Bengali (currently the
sole active Stage 3 VQA language -- Indonesian/Javanese are deferred).
Initialises directly from Stage 2's vision-mapping checkpoint (chained
lineage: Stage 1 -> Stage 2 -> Stage 3 (VQA)) so the final checkpoint has
been trained on translation, vision-grounding, and VQA in that order.
Stage 3a (text-only augmentation training) is no longer part of the
pipeline -- the code still exists in train_text.py if you want to run it
separately, but this script no longer chains from it. NLLB encoder, vision
tower, and LLM all stay frozen; only the Mapping is updated.

Input JSONL fields (from load_vqa_data.py):
    vg_image_id, query, answer, source_language, nllb_lang_tag

Images are resolved from a local GQA images directory (download and
extract https://nlp.stanford.edu/data/gqa/images.zip once, then pass its
path via --images-dir); GQA/xGQA reference images by id
(`{images_dir}/{vg_image_id}.jpg`) rather than a per-question URL.

Usage
-----
    python train_vqa.py \\
        --data-dir ./data/stage3b \\
        --images-dir /path/to/gqa/images \\
        --output-dir ./outputs/stage3b \\
        --init-mapping-ckpt ../Stage2/outputs/wit/pytorch_model.bin \\
        --mt-path facebook/nllb-200-3.3B \\
        --llm-path Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import random
from datetime import datetime
from functools import partial

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, NllbTokenizer

from model import AugmentedVisualMindMerger

try:
    import wandb
except ImportError:
    wandb = None


_VQA_SYSTEM = "You are a helpful assistant that answers questions about images."


def _build_user_prompt(question: str) -> str:
    """Build the task-formatted user prompt for the LLM-side query term `T`.

    Must match :func:`evaluate_vqa.build_vqa_prompt` exactly so training
    and evaluation see identical LLM inputs (same convention as Stage 3a's
    `_build_user_prompt` / `evaluate_text.py`).
    """
    return f"Question: {question}\nAnswer with a single word or short phrase, in English."


def _build_chat_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": _VQA_SYSTEM},
        {"role": "user", "content": _build_user_prompt(question)},
    ]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("stage3b_train_vqa")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"train_vqa_{ts}.log"))
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


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VQADataset(Dataset):
    """Loads GQA-translated VQA JSONL rows and resolves images locally.

    Args:
        rows: List of dicts from :mod:`load_vqa_data`'s output JSONL.
        images_dir: Local directory of extracted GQA images, named
            ``{vg_image_id}.jpg`` (see module docstring).
    """

    def __init__(self, rows: list[dict], images_dir: str) -> None:
        self.rows = rows
        self.images_dir = images_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict | None:
        row = self.rows[idx]
        image = self._load_image(row["vg_image_id"])
        if image is None:
            return None
        return {
            "image": image,
            "query": row["query"],
            "answer": row["answer"],
            "nllb_lang_tag": row["nllb_lang_tag"],
        }

    def _load_image(self, image_id: str) -> Image.Image | None:
        for ext in (".jpg", ".jpeg", ".png"):
            path = os.path.join(self.images_dir, f"{image_id}{ext}")
            if os.path.exists(path):
                try:
                    return Image.open(path).convert("RGB")
                except Exception:
                    return None
        return None


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_vqa(
    batch: list[dict | None],
    processor: AutoProcessor,
    min_pixels: int,
    max_pixels: int,
) -> dict | None:
    """Collate a batch of VQA samples, processing images with the Qwen3-VL processor.

    Args:
        batch: List of sample dicts (or ``None`` for missing images).
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
        text=["" for _ in images],
        return_tensors="pt",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return {
        "pixel_values": image_inputs["pixel_values"],
        "image_grid_thw": image_inputs["image_grid_thw"],
        "queries": [x["query"] for x in valid],
        "answers": [x["answer"] for x in valid],
        "nllb_lang_tags": [x["nllb_lang_tag"] for x in valid],
    }


# ---------------------------------------------------------------------------
# Tokenisation helpers (mirror Stage 2/3a's mt_input_features/llm_input_features)
# ---------------------------------------------------------------------------

def mt_input_features(
    texts: list[str],
    nllb_lang_tags: list[str],
    tokenizer_mt: NllbTokenizer,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    tokenizer_llm.add_bos_token = add_bos
    tokenizer_llm.add_eos_token = add_eos
    enc = tokenizer_llm(
        texts, truncation=True, max_length=max_seq_len, padding=True, return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, model: AugmentedVisualMindMerger, step: int, loss: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"step": step, "loss": loss, "model_state_dict": model.mapping.state_dict()},
        path,
    )


# ---------------------------------------------------------------------------
# W&B helpers (same pattern as Stage 2/3a)
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
        wandb.init(project=args.wandb_project, name=args.wandb_run_name or None, config=config)
        return True
    except Exception as exc:
        print(f"wandb init failed ({exc}); disabling.")
        return False


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main(args, logger: logging.Logger) -> None:
    """Run the Stage 3b VQA augmentation training loop.

    Loads GQA-translated multilingual VQA data, builds an
    :class:`AugmentedVisualMindMerger`, initialises its mapping from
    Stage 2's vision-mapping checkpoint, and trains with
    validation-based checkpointing.
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage 3b VQA augmentation expects CUDA.")

    if os.path.isfile(args.data_dir):
        # Per-language pipelines pass a single flat file (e.g. Stage3/data/ru.jsonl)
        # rather than a directory to glob.
        jsonl_files = [args.data_dir]
    else:
        jsonl_files = sorted(glob.glob(os.path.join(args.data_dir, "*.jsonl")))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in --data-dir: {args.data_dir}")
    rows: list[dict] = []
    for path in jsonl_files:
        file_rows = read_jsonl(path)
        logger.info("Loaded %d rows from %s", len(file_rows), path)
        rows.extend(file_rows)
    random.shuffle(rows)

    split = int(len(rows) * (1.0 - args.val_ratio))
    train_rows, val_rows = rows[:split], rows[split:]
    logger.info("Dataset: train=%d val=%d", len(train_rows), len(val_rows))

    tokenizer_mt = NllbTokenizer.from_pretrained(args.mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(args.llm_path, use_fast=True)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"
    processor = AutoProcessor.from_pretrained(args.llm_path)

    model = AugmentedVisualMindMerger(
        mt_path=args.mt_path,
        llm_path=args.llm_path,
        max_gen_len=args.max_gen_len,
        llm_bos_token_id=tokenizer_llm.bos_token_id,
        llm_pad_token_id=tokenizer_llm.pad_token_id,
        local_files_only=args.local_files_only,
    ).to(device)

    if args.init_mapping_ckpt:
        ckpt = torch.load(args.init_mapping_ckpt, map_location="cpu")
        model.mapping.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info("Initialised mapping from: %s", args.init_mapping_ckpt)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    _collate = partial(collate_vqa, processor=processor, min_pixels=args.visual_pixels, max_pixels=args.visual_pixels)
    train_loader = DataLoader(
        VQADataset(train_rows, args.images_dir), batch_size=args.train_batch_size,
        shuffle=True, collate_fn=_collate, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        VQADataset(val_rows, args.images_dir), batch_size=args.eval_batch_size,
        shuffle=False, collate_fn=_collate, num_workers=args.num_workers,
    )

    use_wandb = _init_wandb_or_disable(args, {
        "stage": "stage3b_vqa_augmentation",
        "mt_path": args.mt_path, "llm_path": args.llm_path,
        "lr": args.lr, "epochs": args.epochs,
        "train_size": len(train_rows), "val_size": len(val_rows),
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
                batch["queries"], batch["nllb_lang_tags"], tokenizer_mt, args.max_mt_seq_len, device,
            )
            formatted_query = [
                tokenizer_llm.apply_chat_template(
                    _build_chat_messages(q), tokenize=False, add_generation_prompt=True,
                )
                for q in batch["queries"]
            ]
            input_ids_query_llm, mask_query_llm = llm_input_features(
                formatted_query, tokenizer_llm, args.max_seq_len, add_bos=False, add_eos=False, device=device,
            )
            labels, mask_label = llm_input_features(
                batch["answers"], tokenizer_llm, args.max_gen_len, add_bos=False, add_eos=True, device=device,
            )

            loss = model(
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                input_ids_mt=input_ids_mt, attention_mask_mt=mask_mt,
                input_ids_query_llm=input_ids_query_llm, mask_query_llm=mask_query_llm,
                labels=labels, mask_label=mask_label,
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
                    batch["queries"], batch["nllb_lang_tags"], tokenizer_mt, args.max_mt_seq_len, device,
                )
                formatted_query = [
                    tokenizer_llm.apply_chat_template(
                        _build_chat_messages(q), tokenize=False, add_generation_prompt=True,
                    )
                    for q in batch["queries"]
                ]
                input_ids_query_llm, mask_query_llm = llm_input_features(
                    formatted_query, tokenizer_llm, args.max_seq_len, add_bos=False, add_eos=False, device=device,
                )
                labels, mask_label = llm_input_features(
                    batch["answers"], tokenizer_llm, args.max_gen_len, add_bos=False, add_eos=True, device=device,
                )
                loss = model(
                    pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                    input_ids_mt=input_ids_mt, attention_mask_mt=mask_mt,
                    input_ids_query_llm=input_ids_query_llm, mask_query_llm=mask_query_llm,
                    labels=labels, mask_label=mask_label,
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
            ckpt_path = os.path.join(args.output_dir, "pytorch_model.bin")
            save_checkpoint(ckpt_path, model, global_step, val_loss)
            logger.info("Saved best checkpoint (val_loss=%.4f) -> %s", val_loss, ckpt_path)

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3b: train the VQA-augmented mapping layer.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing GQA-translated *.jsonl files from load_vqa_data.py, "
                             "or a path to a single .jsonl file directly.")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Local GQA images directory (extract https://nlp.stanford.edu/data/gqa/images.zip).")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--init-mapping-ckpt", type=str, required=True,
                        help="Path to Stage 2's vision-mapping pytorch_model.bin "
                             "(chained lineage: Stage 1 -> Stage 2 -> Stage 3 (VQA)).")
    parser.add_argument("--mt-path", type=str, default="facebook/nllb-200-3.3B")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-mt-seq-len", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=256,
                        help="LLM tokenizer max length for the query prompt T.")
    parser.add_argument("--max-gen-len", type=int, default=16,
                        help="Max token length for the short VQA answer.")
    parser.add_argument(
        "--visual-pixels", type=int, default=256 * 256,
        help="Total pixel budget per image (min_pixels=max_pixels forces a fixed visual token count).",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="mindmerger-stage3b")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(os.path.dirname(__file__), "logs"))
    main(args, logger)
