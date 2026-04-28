from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, NllbTokenizer

from modeling_augmentation import AugmentedMindMerger

try:
    import wandb
except ImportError:
    wandb = None


NLLB_LANG_MAP = {
    "English": "eng_Latn",
    "Swahili": "swh_Latn",
    "Yoruba": "yor_Latn",
    "Wolof": "wol_Latn",
}


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


class Stage2Dataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def mt_input_features(
    texts: Iterable[str],
    langs: Iterable[str],
    tokenizer_mt: NllbTokenizer,
    max_seq_len: int,
    device: torch.device,
):
    ids, masks = [], []
    for text, lang in zip(texts, langs):
        tokenizer_mt.src_lang = NLLB_LANG_MAP[lang]
        enc = tokenizer_mt(text, truncation=True, max_length=max_seq_len, padding=False)
        ids.append(enc["input_ids"])
        masks.append(enc["attention_mask"])

    max_len = max(len(x) for x in ids)
    pad_id = tokenizer_mt.pad_token_id
    for i in range(len(ids)):
        while len(ids[i]) < max_len:
            ids[i].append(pad_id)
            masks[i].append(0)

    return torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(masks, dtype=torch.long, device=device)


def llm_input_features(
    texts: Iterable[str],
    tokenizer_llm: AutoTokenizer,
    max_seq_len: int,
    add_bos: bool,
    add_eos: bool,
    device: torch.device,
):
    tokenizer_llm.add_bos_token = add_bos
    tokenizer_llm.add_eos_token = add_eos
    enc = tokenizer_llm(
        list(texts),
        truncation=True,
        max_length=max_seq_len,
        padding=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def collate_batch(batch):
    return {
        "query": [x["query"] for x in batch],
        "answer": [x["answer"] for x in batch],
        "source_language": [x.get("source_language", "English") for x in batch],
    }


def save_mapping_checkpoint(path: str, model: AugmentedMindMerger, step: int, loss: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "loss": loss,
            "model_state_dict": model.mapping.state_dict(),
        },
        path,
    )


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
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() != "WANDB_API_KEY":
                    continue
                value = value.strip().strip('"').strip("'")
                if value:
                    os.environ["WANDB_API_KEY"] = value
                    return True
    return False


def _init_wandb_or_disable(args, config: dict) -> bool:
    if not args.use_wandb:
        return False
    if wandb is None:
        print("wandb not installed; disabling Weights & Biases logging.")
        return False

    mode = args.wandb_mode.lower()
    if mode not in {"auto", "online", "offline"}:
        print(f"Invalid wandb mode '{args.wandb_mode}', disabling Weights & Biases logging.")
        return False

    if mode in {"auto", "online"} and not os.environ.get("WANDB_API_KEY"):
        _load_wandb_key_from_tokens()

    if mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
    elif mode == "auto" and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "offline"

    try:
        if os.environ.get("WANDB_API_KEY"):
            wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name if args.wandb_run_name else None,
            config=config,
            settings=wandb.Settings(init_timeout=args.wandb_init_timeout),
        )
        print(f"Initialized Weights & Biases (mode={os.environ.get('WANDB_MODE', 'online')}).")
        return True
    except Exception as exc:
        if mode == "auto":
            try:
                os.environ["WANDB_MODE"] = "offline"
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name if args.wandb_run_name else None,
                    config=config,
                    settings=wandb.Settings(init_timeout=args.wandb_init_timeout),
                )
                print(f"wandb online init failed ({exc}); fell back to offline mode.")
                return True
            except Exception as offline_exc:
                print(f"wandb init failed ({exc}); offline fallback failed ({offline_exc}); disabling logging.")
                return False
        print(f"wandb init failed ({exc}); disabling Weights & Biases logging.")
        return False


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage2 augmentation expects CUDA.")

    rows = read_jsonl(args.english_data)
    if args.translated_data:
        rows.extend(read_jsonl(args.translated_data))
    random.shuffle(rows)

    split_idx = int(len(rows) * (1.0 - args.val_ratio))
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    print(json.dumps({"train_size": len(train_rows), "val_size": len(val_rows)}, indent=2))
    use_wandb = _init_wandb_or_disable(
        args,
        {
            "stage": "stage2_augmentation",
            "mt_path": args.mt_path,
            "llm_path": args.llm_path,
            "train_size": len(train_rows),
            "val_size": len(val_rows),
            "lr": args.lr,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "grad_accum": args.grad_accum,
            "max_seq_len": args.max_seq_len,
            "max_gen_len": args.max_gen_len,
            "val_ratio": args.val_ratio,
        },
    )

    tokenizer_mt = NllbTokenizer.from_pretrained(args.mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(args.llm_path, use_fast=False)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"

    model = AugmentedMindMerger(
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
        print(f"Loaded Stage1 mapping: {args.stage1_mapping_ckpt}")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    train_loader = DataLoader(
        Stage2Dataset(train_rows),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        Stage2Dataset(val_rows),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    best_val = float("inf")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"epoch={epoch}")
        optimizer.zero_grad(set_to_none=True)
        for batch in pbar:
            query = batch["query"]
            answer = batch["answer"]
            source_lang = batch["source_language"]

            input_ids_mt, mask_mt = mt_input_features(
                query, source_lang, tokenizer_mt, args.max_seq_len, device
            )
            input_ids_query_llm, mask_query_llm = llm_input_features(
                query, tokenizer_llm, args.max_seq_len, add_bos=False, add_eos=False, device=device
            )
            labels, mask_label = llm_input_features(
                answer, tokenizer_llm, args.max_gen_len, add_bos=False, add_eos=True, device=device
            )

            loss = model(
                input_ids_mt=input_ids_mt,
                attention_mask_mt=mask_mt,
                input_ids_query_llm=input_ids_query_llm,
                mask_query_llm=mask_query_llm,
                labels=labels,
                mask_label=mask_label,
            )
            loss = loss / args.grad_accum
            loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            running += loss.item() * args.grad_accum
            steps += 1
            train_loss = running / max(steps, 1)
            pbar.set_postfix(loss=f"{train_loss:.4f}")
            if use_wandb:
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    }
                )

        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                query = batch["query"]
                answer = batch["answer"]
                source_lang = batch["source_language"]

                input_ids_mt, mask_mt = mt_input_features(
                    query, source_lang, tokenizer_mt, args.max_seq_len, device
                )
                input_ids_query_llm, mask_query_llm = llm_input_features(
                    query, tokenizer_llm, args.max_seq_len, add_bos=False, add_eos=False, device=device
                )
                labels, mask_label = llm_input_features(
                    answer, tokenizer_llm, args.max_gen_len, add_bos=False, add_eos=True, device=device
                )

                loss = model(
                    input_ids_mt=input_ids_mt,
                    attention_mask_mt=mask_mt,
                    input_ids_query_llm=input_ids_query_llm,
                    mask_query_llm=mask_query_llm,
                    labels=labels,
                    mask_label=mask_label,
                )
                val_loss += loss.item()
                val_steps += 1

        val_loss = val_loss / max(1, val_steps)
        val_ppl = math.exp(min(20.0, val_loss))
        print(f"[epoch={epoch}] val_loss={val_loss:.4f}, val_ppl={val_ppl:.4f}")
        if use_wandb:
            wandb.log(
                {
                    "eval/loss": val_loss,
                    "eval/perplexity": val_ppl,
                    "eval/epoch": epoch,
                    "train/global_step": global_step,
                }
            )

        if val_loss < best_val:
            best_val = val_loss
            save_mapping_checkpoint(
                os.path.join(args.output_dir, "mapping", "pytorch_model.bin"),
                model,
                global_step,
                val_loss,
            )
            print(f"Saved best checkpoint (val_loss={val_loss:.4f})")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-mapping-ckpt", type=str, required=True)
    parser.add_argument("--english-data", type=str, required=True)
    parser.add_argument("--translated-data", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mt-path", type=str, default="facebook/nllb-200-3.3B")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-gen-len", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="mindmerger-stage2")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="auto")
    parser.add_argument("--wandb-init-timeout", type=int, default=90)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
