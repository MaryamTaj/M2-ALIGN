"""Stage 0: English-GQA SFT of Qwen3-VL-8B-Instruct -- the "MonoVQA" baseline.

Mirrors MindMerger's Stage 0: the English-task fine-tune that becomes BOTH
the "MonoReason"-equivalent baseline row AND the frozen LLM backbone (phi)
that every later stage (Stage 1 mapping, Stage 2 vision grounding, Stage 3b
VQA augmentation) builds on. Here the English task is GQA VQA: given an image
and an English question, emit a short English answer.

Input JSONL (Stage3/data/english.jsonl, produced by Stage3/load_base_data.py):
    question, answer, vg_image_id
This is the SAME 30k balanced-GQA sample (seed 42) whose per-language NLLB
translations feed Stage 3b, so the only thing that varies between the
MonoVQA baseline and the full pipeline downstream is the language path, not
the question set.

Images resolve locally from a GQA images dir ({images_dir}/{vg_image_id}.jpg),
the same directory Stage 3 uses (download+extract
https://nlp.stanford.edu/data/gqa/images.zip once).

Only the language model is trained; the vision tower + patch merger stay
frozen so `model.get_image_features` remains byte-identical to what Stage 3b's
`AugmentedVisualMindMerger._build_prefix_raw` expects from the frozen phi.

Two modes:
  * default -- LoRA (peft) on the LM's attention/MLP projections, merged back
    into the base weights before saving. Fits a single 80 GB GPU.
  * --deepspeed -- full-weight SFT of the LM under DeepSpeed ZeRO-2 + CPU
    offload (reuses Stage1/tools/deepspeed_config.py), the closest mirror of
    MetaMath-style full SFT. Launch with `deepspeed train.py --deepspeed ...`
    and >=2 GPUs (see Stage0/job-scripts/train.sh).

Output is a plain Hugging Face model directory (save_pretrained + processor)
-> pass its path verbatim as --llm_path (Stage 1), --llm-path (Stage 2/3),
and --model-id (Baseline/evaluate.py).

Usage (LoRA, single GPU)
-----------------------
    python Stage0/train.py \\
        --data-path  $SCRATCH/M2-ALIGN/Stage3/data/english.jsonl \\
        --images-dir $SCRATCH/M2-ALIGN/Stage3/data/gqa/images \\
        --output-dir $SCRATCH/M2-ALIGN/Stage0/outputs/qwen3vl-8b-gqa \\
        --llm-path   Qwen/Qwen3-VL-8B-Instruct

Usage (full SFT, DeepSpeed)
--------------------------
    deepspeed --master_port 50040 Stage0/train.py --deepspeed \\
        --full-finetune \\
        --data-path  .../english.jsonl --images-dir .../gqa/images \\
        --output-dir .../Stage0/outputs/qwen3vl-8b-gqa \\
        --llm-path   Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from functools import partial

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

try:
    import wandb
except ImportError:
    wandb = None

# Reuse Stage 1's DeepSpeed config builder for the --full-finetune path.
try:
    from Stage1.tools.deepspeed_config import get_train_ds_config
except ImportError:  # running from inside Stage1/ on sys.path, or no deepspeed
    try:
        from tools.deepspeed_config import get_train_ds_config
    except ImportError:
        get_train_ds_config = None

SCRATCH_ROOT = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN", "Stage0")

# Must match Stage3/train.py + Stage3/evaluate.py + Baseline/evaluate.py exactly
# so Stage 0 SFT, the MonoVQA baseline eval, and every later stage see an
# identical rendered prompt.
_VQA_SYSTEM = "You are a helpful assistant that answers questions about images."


def _build_user_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer with a single word or short phrase, in English."


def _build_chat_messages_with_image(image: Image.Image, question: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": _VQA_SYSTEM}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": _build_user_prompt(question)},
        ]},
    ]


# ---------------------------------------------------------------------------
# Boilerplate (mirrors Stage 2 / Stage 3)
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("stage0_sft")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
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


def _init_wandb_or_disable(args, config: dict, is_main: bool) -> bool:
    if not args.use_wandb or wandb is None or not is_main:
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
# Dataset / collation
# ---------------------------------------------------------------------------

class GQADataset(Dataset):
    def __init__(self, rows: list[dict], images_dir: str) -> None:
        self.rows = rows
        self.images_dir = images_dir

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, image_id: str) -> Image.Image | None:
        for ext in (".jpg", ".jpeg", ".png"):
            path = os.path.join(self.images_dir, f"{image_id}{ext}")
            if os.path.exists(path):
                try:
                    return Image.open(path).convert("RGB")
                except Exception:
                    return None
        return None

    def __getitem__(self, idx: int) -> dict | None:
        row = self.rows[idx]
        image = self._load_image(str(row["vg_image_id"]))
        if image is None:
            return None
        return {"image": image, "question": row["question"], "answer": str(row["answer"])}


def collate(
    batch: list[dict | None],
    processor: AutoProcessor,
    max_seq_len: int,
    visual_pixels: int,
) -> dict | None:
    """Render system->user(image+question)->assistant, then append the answer.

    Labels mask every prompt token (including the assistant header) so loss
    is taken only on the answer + EOS. Per-example rendering is required
    because the <|image_pad|> count depends on that image's grid; the batch
    is right-padded afterwards (pads are masked via labels=-100 and
    attention_mask=0).
    """
    valid = [b for b in batch if b is not None]
    if not valid:
        return None

    tok = processor.tokenizer
    eos_id = tok.eos_token_id
    # Qwen3-VL's M-RoPE needs mm_token_type_ids (1 = image/video token, 0 = text).
    # The processor returns it alongside input_ids; fall back to deriving it from
    # the <|image_pad|> id if an older processor omits it.
    image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    ids_list, labels_list, pv_list, grid_list, mmtt_list = [], [], [], [], []
    for x in valid:
        messages = _build_chat_messages_with_image(x["image"], x["question"])
        enc = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            processor_kwargs={
                "min_pixels": visual_pixels, "max_pixels": visual_pixels,
                "truncation": True, "max_length": max_seq_len,
            },
        )
        prompt_ids = enc["input_ids"][0]
        ans_ids = tok(x["answer"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        eos = torch.tensor([eos_id], dtype=torch.long)
        ids = torch.cat([prompt_ids, ans_ids, eos])
        labels = torch.cat([
            torch.full((prompt_ids.size(0),), -100, dtype=torch.long),
            ans_ids, eos,
        ])
        if "mm_token_type_ids" in enc:
            prompt_mmtt = enc["mm_token_type_ids"][0].to(torch.long)
        else:
            prompt_mmtt = (prompt_ids == image_pad_id).to(torch.long)
        # answer + eos are text tokens -> type 0
        mmtt = torch.cat([prompt_mmtt, torch.zeros(ans_ids.size(0) + 1, dtype=torch.long)])
        ids_list.append(ids)
        labels_list.append(labels)
        pv_list.append(enc["pixel_values"])
        grid_list.append(enc["image_grid_thw"])
        mmtt_list.append(mmtt)

    n = len(valid)
    max_len = max(t.size(0) for t in ids_list)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    labels = torch.full((n, max_len), -100, dtype=torch.long)
    attn = torch.zeros((n, max_len), dtype=torch.long)
    mm_token_type_ids = torch.zeros((n, max_len), dtype=torch.long)
    for i, (ids, lab, mmtt) in enumerate(zip(ids_list, labels_list, mmtt_list)):
        L = ids.size(0)
        input_ids[i, :L] = ids
        labels[i, :L] = lab
        attn[i, :L] = 1
        mm_token_type_ids[i, :L] = mmtt

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "pixel_values": torch.cat(pv_list, dim=0),
        "image_grid_thw": torch.cat(grid_list, dim=0),
        "mm_token_type_ids": mm_token_type_ids,
    }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(args, logger: logging.Logger):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.llm_path,
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )

    # Vision tower + merger frozen in BOTH modes.
    for p in model.model.visual.parameters():
        p.requires_grad = False

    if args.full_finetune:
        for name, p in model.named_parameters():
            if not name.startswith("model.visual."):
                p.requires_grad = True
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        logger.info("Full-finetune: training the language model + lm_head, vision tower frozen.")
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError(
            "LoRA mode needs `peft` (not in the m2-align venv). Either "
            "`pip install peft` on the workstation node, or use --full-finetune "
            "with --deepspeed (MODE=full in the job script)."
        ) from e

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Regex: only the text decoder's projections, never the vision tower's.
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _forward_loss(model, batch, device, dtype) -> torch.Tensor:
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        pixel_values=batch["pixel_values"].to(device, dtype=dtype),
        image_grid_thw=batch["image_grid_thw"].to(device),
        mm_token_type_ids=batch["mm_token_type_ids"].to(device),
        labels=batch["labels"].to(device),
    )
    return out.loss


def save_hf_lora(model, processor, tokenizer, out_dir: str, logger: logging.Logger) -> None:
    """Merge LoRA into the base weights and write a plain HF model dir."""
    os.makedirs(out_dir, exist_ok=True)
    model.merge_and_unload().save_pretrained(out_dir, safe_serialization=True)
    processor.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info("Saved MonoVQA checkpoint (LoRA merged) -> %s", out_dir)


def save_hf_zero3(engine, base_config, processor, tokenizer, out_dir: str,
                  is_main: bool, logger: logging.Logger) -> None:
    """Consolidate ZeRO-3 shards to a single fp16/bf16 checkpoint + HF metadata.

    ``save_16bit_model`` is collective (every rank must call it); the
    config/processor/tokenizer writes are rank-0 only. The result loads with
    ``from_pretrained`` like any HF dir.
    """
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    # Keep the frozen vision tower in the checkpoint -- downstream stages load
    # the whole model with from_pretrained().
    engine.save_16bit_model(out_dir, "pytorch_model.bin", exclude_frozen_parameters=False)
    if is_main:
        base_config.save_pretrained(out_dir)
        processor.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        logger.info("Saved MonoVQA checkpoint (ZeRO-3 consolidated) -> %s", out_dir)


def main(args, logger: logging.Logger) -> None:
    use_deepspeed = args.deepspeed
    if use_deepspeed:
        import deepspeed

        deepspeed.init_distributed()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("Stage 0 SFT expects CUDA.")
        world_size, global_rank, local_rank = 1, 0, 0
    is_main = global_rank == 0

    set_seed(args.seed)

    rows: list[dict] = []
    for path in args.data_path:
        r = read_jsonl(path)
        logger.info("Loaded %d rows from %s", len(r), path)
        rows.extend(r)
    random.shuffle(rows)
    split = int(len(rows) * (1.0 - args.val_ratio))
    train_rows, val_rows = rows[:split], rows[split:]
    logger.info("Dataset: train=%d val=%d", len(train_rows), len(val_rows))

    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(args.llm_path)

    model = build_model(args, logger)
    is_lora = not args.full_finetune
    base_config = model.config
    model.to(device)
    dtype = torch.bfloat16

    _collate = partial(
        collate, processor=processor,
        max_seq_len=args.max_seq_len, visual_pixels=args.visual_pixels,
    )
    train_ds = GQADataset(train_rows, args.images_dir)
    val_ds = GQADataset(val_rows, args.images_dir)

    if use_deepspeed:
        if get_train_ds_config is None:
            raise ImportError("get_train_ds_config unavailable; add Stage1/ to PYTHONPATH.")
        micro = args.train_batch_size
        grad_accum = max(1, args.global_batch_size // (micro * world_size))
        ds_config = get_train_ds_config(
            train_batch_size=micro * world_size * grad_accum,
            train_micro_batch_size_per_gpu=micro,
            lr=args.lr,
            gradient_accumulation_steps=grad_accum,
            offload=True,
            # ZeRO-3 (not 2): on A100-40 an 8B model's replicated bf16
            # weights+grads (32 GB) leave no room; stage 3 shards params too.
            stage=3,
        )
        # Let save_16bit_model() gather a consolidated checkpoint on save.
        ds_config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
        model, optimizer, _, _ = deepspeed.initialize(
            config=ds_config, model=model,
            model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
        )
        from torch.utils.data.distributed import DistributedSampler

        train_loader = DataLoader(
            train_ds, batch_size=micro, sampler=DistributedSampler(train_ds),
            collate_fn=_collate, num_workers=args.num_workers,
        )
    else:
        grad_accum = args.grad_accum
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.train_batch_size, shuffle=True,
            collate_fn=_collate, num_workers=args.num_workers,
        )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=_collate, num_workers=args.num_workers,
    )

    use_wandb = _init_wandb_or_disable(args, {
        "stage": "stage0_english_gqa_sft",
        "llm_path": args.llm_path, "mode": "full" if args.full_finetune else "lora",
        "lr": args.lr, "epochs": args.epochs,
        "train_size": len(train_rows), "val_size": len(val_rows),
        "grad_accum": grad_accum, "world_size": world_size,
    }, is_main)

    best_val = float("inf")
    best_adapter_sd = None   # LoRA path: CPU snapshot of the best adapter weights
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        if use_deepspeed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        running, steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch={epoch}", disable=not is_main)
        if not use_deepspeed:
            optimizer.zero_grad(set_to_none=True)

        for batch in pbar:
            if batch is None:
                continue
            loss = _forward_loss(model, batch, device, dtype)

            if use_deepspeed:
                model.backward(loss)
                model.step()
            else:
                (loss / grad_accum).backward()
                if (global_step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad), 1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            running += loss.item()
            steps += 1
            global_step += 1
            if is_main:
                pbar.set_postfix(loss=f"{running / steps:.4f}")
                if use_wandb:
                    wandb.log({"train/loss": running / steps, "train/global_step": global_step})

        # Validation. Under ZeRO-3 every rank must take part in the forward
        # (param all-gather is collective), so all ranks run the full val set
        # and the loss is reduced across them.
        model.eval()
        vl, vs = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                vl += _forward_loss(model, batch, device, dtype).item()
                vs += 1
        if use_deepspeed:
            t = torch.tensor([vl, float(vs)], device=device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            vl, vs = t[0].item(), t[1].item()
        val_loss = vl / max(1.0, vs)
        if is_main:
            logger.info("Epoch %d | val_loss=%.4f | val_ppl=%.4f",
                        epoch, val_loss, math.exp(min(20.0, val_loss)))
            if use_wandb:
                wandb.log({"eval/loss": val_loss, "train/global_step": global_step})

        if val_loss < best_val:
            best_val = val_loss
            if use_deepspeed:
                # ZeRO-3 consolidation is non-destructive -- safe to save in-loop.
                save_hf_zero3(model, base_config, processor, tokenizer,
                              args.output_dir, is_main, logger)
            elif is_main:
                # merge_and_unload() would mutate the live model and break the
                # next backward(), so just snapshot the adapter and merge once
                # after training (below).
                from peft import get_peft_model_state_dict

                best_adapter_sd = {
                    k: v.detach().to("cpu", copy=True)
                    for k, v in get_peft_model_state_dict(model).items()
                }
                logger.info("Epoch %d is new best (val_loss=%.4f); adapter snapshot kept",
                            epoch, val_loss)

    # LoRA path: reload the best adapter and write the merged HF checkpoint once.
    if not use_deepspeed and is_main and best_adapter_sd is not None:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model, best_adapter_sd)
        save_hf_lora(model, processor, tokenizer, args.output_dir, logger)

    if is_main and use_wandb:
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 0: English-GQA SFT of Qwen3-VL (MonoVQA baseline).")
    p.add_argument("--data-path", type=str, nargs="+", required=True,
                   help="English GQA JSONL (Stage3/data/english.jsonl); one or more files.")
    p.add_argument("--images-dir", type=str, required=True,
                   help="Local GQA images dir ({vg_image_id}.jpg), same as Stage 3.")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Destination HF model dir -> use as --llm_path / --llm-path / --model-id.")
    p.add_argument("--llm-path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--local-files-only", action="store_true")

    p.add_argument("--full-finetune", action="store_true",
                   help="Full-weight LM SFT instead of LoRA (needs --deepspeed, launch via `deepspeed ...`).")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--train-batch-size", type=int, default=4,
                   help="Per-GPU micro-batch.")
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Single-GPU (non-deepspeed) gradient accumulation.")
    p.add_argument("--global-batch-size", type=int, default=64,
                   help="DeepSpeed effective batch (micro * world_size * accum).")
    p.add_argument("--max-seq-len", type=int, default=256,
                   help="Rendered-prompt truncation length (matches Stage 3).")
    p.add_argument("--visual-pixels", type=int, default=256 * 256,
                   help="min_pixels = max_pixels per image (matches Stage 3).")
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="m2-align")
    p.add_argument("--wandb-run-name", type=str, default="stage0-english-gqa-sft")
    p.add_argument("--wandb-mode", type=str, default="offline")
    # deepspeed launcher injects this
    p.add_argument("--local_rank", type=int, default=0)
    # Adds --deepspeed / --deepspeed_config (same pattern as Stage1/train.py).
    import deepspeed as _ds
    p = _ds.add_config_arguments(p)
    args = p.parse_args()

    if args.full_finetune and not args.deepspeed:
        raise SystemExit("--full-finetune requires --deepspeed (launch with `deepspeed train.py ...`).")

    os.makedirs(args.output_dir, exist_ok=True)
    main(args, setup_logging())
