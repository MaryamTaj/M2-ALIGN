"""Stage 2 augmentation training for AugmentedMindMerger."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from datetime import datetime
from typing import Iterable

import glob

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, NllbTokenizer

from model import AugmentedMindMerger

try:
    import wandb
except ImportError:
    wandb = None


NLLB_LANG_MAP = {
    "English": "eng_Latn",
    "Swahili": "swh_Latn",
    "Yoruba": "yor_Latn",
    "Wolof": "wol_Latn",
    "French": "fra_Latn",
}

# Task-specific system messages — must match evaluate_text.py exactly.
_SYSTEM_MSGS: dict[str, str] = {
    "xnli":   "You are a helpful assistant that determines textual entailment.",
    "xcsqa":  "You are a helpful assistant that answers commonsense questions.",
    "mgsm":   "You are a helpful assistant that solves math problems step by step.",
    "msvamp": "You are a helpful assistant that solves math problems step by step.",
}


def _build_user_prompt(raw_query: str, task: str) -> str:
    """Build the task-formatted user prompt from the raw JSONL query string.

    The returned string matches the prompt format used in evaluate_text.py so
    training and evaluation see identical LLM inputs.

    Args:
        raw_query: The ``query`` field from the training JSONL.  Format per task:
            xnli   → ``"{premise}\\n{hypothesis}"``
            xcsqa  → ``"{stem}\\nA. {c1}\\nB. {c2}..."``
            math   → Swahili question string.
        task: One of ``"xnli"``, ``"xcsqa"``, ``"mgsm"``, ``"msvamp"``.

    Returns:
        A formatted prompt string ready for :func:`apply_chat_template`.
    """
    if task == "xnli":
        parts = raw_query.split("\n", 1)
        premise = parts[0]
        hypothesis = parts[1] if len(parts) > 1 else ""
        return (
            "Determine the relationship between the premise and the hypothesis. "
            "Respond with exactly one word: entailment, neutral, or contradiction.\n\n"
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            "Answer:"
        )
    if task == "xcsqa":
        parts = raw_query.split("\n", 1)
        stem = parts[0]
        choice_lines = parts[1] if len(parts) > 1 else ""
        return (
            f"Question: {stem}\n"
            f"Choices:\n{choice_lines}\n\n"
            "Answer with the letter corresponding to the correct choice (e.g., A, B, C, D, ...)."
        )
    # mgsm / msvamp
    return f"{raw_query}\n\nLet's think step by step."


def _build_chat_messages(raw_query: str, task: str) -> list[dict]:
    """Return the system + user message list for :func:`apply_chat_template`.

    Args:
        raw_query: Raw query string from training JSONL.
        task: Task identifier (see :func:`_build_user_prompt`).

    Returns:
        List of message dicts with ``role`` and ``content`` keys.
    """
    return [
        {"role": "system", "content": _SYSTEM_MSGS[task]},
        {"role": "user",   "content": _build_user_prompt(raw_query, task)},
    ]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> logging.Logger:
    """Create a logger that writes to stdout and a timestamped file in *log_dir*.

    Args:
        log_dir: Directory in which to create the log file (created if absent).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"augmentation_{timestamp}.log")

    logger = logging.getLogger("stage2_augmentation")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for Python and PyTorch for reproducibility.

    Args:
        seed: Integer seed value applied to :mod:`random`,
            ``torch.manual_seed``, and all CUDA devices.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file and return its rows as a list of dicts.

    Tolerates a leading BOM character and stray non-JSON prefixes before
    the opening ``{`` on a line.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        A list of parsed JSON objects.

    Raises:
        ValueError: If any line cannot be parsed as JSON.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip().lstrip("﻿")
            if not line:
                continue
            if not line.startswith("{"):
                brace = line.find("{")
                if brace > 0:
                    print(
                        f"warning: {path}:{lineno} starts with non-JSON prefix "
                        f"{line[:brace]!r}; stripping before parse."
                    )
                    line = line[brace:]
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON at {path}:{lineno}: {exc.msg} "
                    f"(line preview: {line[:120]!r})"
                ) from exc
    return rows


class Stage2Dataset(Dataset):
    """Minimal PyTorch Dataset wrapper around a list of sample dicts.

    Args:
        rows: List of dicts, each with at least ``"query"``, ``"answer"``,
            and optionally ``"source_language"`` keys.
    """

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        """Return the sample at index *idx*.

        Args:
            idx: Integer index into the dataset.

        Returns:
            The raw sample dict at position *idx*.
        """
        return self.rows[idx]


def mt_input_features(
    texts: Iterable[str],
    langs: Iterable[str],
    tokenizer_mt: NllbTokenizer,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise source-language texts with the NLLB tokenizer.

    Sets ``tokenizer_mt.src_lang`` per example and right-pads the batch
    to the length of the longest sequence.

    Args:
        texts: Source-language strings to tokenize, one per example.
        langs: Human-readable language names parallel to *texts*
            (must be keys in :data:`NLLB_LANG_MAP`).
        tokenizer_mt: An instantiated :class:`NllbTokenizer`.
        max_seq_len: Maximum sequence length; longer sequences are
            right-truncated.
        device: Target device for the returned tensors.

    Returns:
        A 2-tuple ``(input_ids, attention_mask)`` of shape
        ``[batch, seq_len]`` on *device*.
    """
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

    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def llm_input_features(
    texts: Iterable[str],
    tokenizer_llm: AutoTokenizer,
    max_seq_len: int,
    add_bos: bool,
    add_eos: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise texts with the LLM tokenizer.

    Controls BOS/EOS insertion and moves the result to *device*.

    Args:
        texts: Strings to tokenize.
        tokenizer_llm: An instantiated LLM tokenizer (e.g. Qwen3-VL).
        max_seq_len: Maximum sequence length; longer sequences are
            right-truncated.
        add_bos: Whether to prepend a BOS token.
        add_eos: Whether to append an EOS token.
        device: Target device for the returned tensors.

    Returns:
        A 2-tuple ``(input_ids, attention_mask)`` on *device*.
    """
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


def collate_batch(batch: list[dict]) -> dict[str, list]:
    """Collate a list of sample dicts into a batched dict.

    Args:
        batch: List of sample dicts from :class:`Stage2Dataset`.

    Returns:
        A dict with keys ``"query"``, ``"answer"``, and
        ``"source_language"``, each mapping to a list of strings.
    """
    return {
        "query": [x["query"] for x in batch],
        "answer": [x["answer"] for x in batch],
        "source_language": [x.get("source_language", "English") for x in batch],
    }


def save_mapping_checkpoint(path: str, model: AugmentedMindMerger, step: int, loss: float) -> None:
    """Save the mapping layer's state dict to *path*.

    Creates parent directories as needed.  The checkpoint includes the
    training step and validation loss alongside the state dict so that
    the best checkpoint can be identified after training.

    Args:
        path: Full file path for the checkpoint (e.g.
            ``"outputs/augmentation/mapping/pytorch_model.bin"``).
        model: The :class:`AugmentedMindMerger` whose mapping to save.
        step: Global training step at which the checkpoint was saved.
        loss: Validation loss at this checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"step": step, "loss": loss, "model_state_dict": model.mapping.state_dict()},
        path,
    )


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def _load_wandb_key_from_tokens() -> bool:
    """Load WANDB_API_KEY from a ``.tokens`` file if present.

    Returns:
        ``True`` if a key was found and set, ``False`` otherwise.
    """
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
                if key.strip() != "WANDB_API_KEY":
                    continue
                value = value.strip().strip('"').strip("'")
                if value:
                    os.environ["WANDB_API_KEY"] = value
                    return True
    return False


def _init_wandb_or_disable(args, config: dict) -> bool:
    """Initialise Weights & Biases with online/offline fallback.

    Args:
        args: Parsed argument namespace with W&B-related attributes.
        config: Hyperparameter dict to log to the W&B run.

    Returns:
        ``True`` if W&B was initialised successfully, ``False`` otherwise.
    """
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
            dir=os.path.dirname(os.path.abspath(__file__)),
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main(args, logger: logging.Logger) -> None:
    """Run the Stage 2 augmentation training loop.

    Loads English and optionally translated task-specialization data,
    builds an :class:`AugmentedMindMerger`, initialises its mapping from
    a Stage 1 checkpoint, and trains with validation-based checkpointing.

    Args:
        args: Parsed argument namespace.
        logger: Logger for training progress and checkpoint messages.
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage2 augmentation expects CUDA.")

    jsonl_files = sorted(glob.glob(os.path.join(args.data_dir, "*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in --data-dir: {args.data_dir}")
    rows = []
    for path in jsonl_files:
        file_rows = read_jsonl(path)
        logger.info("Loaded %d rows from %s", len(file_rows), path)
        rows.extend(file_rows)
    random.shuffle(rows)

    split_idx = int(len(rows) * (1.0 - args.val_ratio))
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]
    logger.info("Dataset split: train=%d, val=%d", len(train_rows), len(val_rows))

    logger.info("Task: %s | prompt format: %s", args.task, _build_user_prompt("<query>", args.task)[:60])

    use_wandb = _init_wandb_or_disable(args, {
        "stage": "stage2_augmentation",
        "task": args.task,
        "mt_path": args.mt_path,
        "llm_path": args.llm_path,
        "data_dir": args.data_dir,
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
    })

    tokenizer_mt = NllbTokenizer.from_pretrained(args.mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(args.llm_path, use_fast=False)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"
    tokenizer_llm.truncation_side = "left"

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
        logger.info("Loaded Stage 1 mapping: %s", args.stage1_mapping_ckpt)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

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
            formatted_query = [
                tokenizer_llm.apply_chat_template(
                    _build_chat_messages(q, args.task),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for q in query
            ]
            input_ids_query_llm, mask_query_llm = llm_input_features(
                formatted_query, tokenizer_llm, args.max_seq_len, add_bos=False, add_eos=False, device=device
            )
            label_texts = [
                tokenizer_llm.apply_chat_template(
                    _build_chat_messages(q, args.task) + [{"role": "assistant", "content": a}],
                    tokenize=False,
                    add_generation_prompt=False,
                )[len(fq):]
                for q, a, fq in zip(query, answer, formatted_query)
            ]
            labels, mask_label = llm_input_features(
                label_texts, tokenizer_llm, args.max_gen_len, add_bos=False, add_eos=False, device=device
            )

            loss = model(
                input_ids_mt=input_ids_mt,
                attention_mask_mt=mask_mt,
                input_ids_query_llm=input_ids_query_llm,
                mask_query_llm=mask_query_llm,
                labels=labels,
                mask_label=mask_label,
            )
            (loss / args.grad_accum).backward()

            if (global_step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            running += loss.item()
            steps += 1
            train_loss = running / steps
            pbar.set_postfix(loss=f"{train_loss:.4f}")
            if use_wandb:
                wandb.log({
                    "train/loss": train_loss,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                })

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
                formatted_query = [
                    tokenizer_llm.apply_chat_template(
                        _build_chat_messages(q, args.task),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for q in query
                ]
                input_ids_query_llm, mask_query_llm = llm_input_features(
                    formatted_query, tokenizer_llm, args.max_seq_len, add_bos=False, add_eos=False, device=device
                )
                label_texts = [
                    tokenizer_llm.apply_chat_template(
                        _build_chat_messages(q, args.task) + [{"role": "assistant", "content": a}],
                        tokenize=False,
                        add_generation_prompt=False,
                    )[len(fq):]
                    for q, a, fq in zip(query, answer, formatted_query)
                ]
                labels, mask_label = llm_input_features(
                    label_texts, tokenizer_llm, args.max_gen_len, add_bos=False, add_eos=False, device=device
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
        logger.info("Epoch %d | val_loss=%.4f | val_ppl=%.4f", epoch, val_loss, val_ppl)
        if use_wandb:
            wandb.log({
                "eval/loss": val_loss,
                "eval/perplexity": val_ppl,
                "eval/epoch": epoch,
                "train/global_step": global_step,
            })

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.output_dir, "mapping", "pytorch_model.bin")
            save_mapping_checkpoint(ckpt_path, model, global_step, val_loss)
            logger.info("Saved best checkpoint (val_loss=%.4f) to %s", val_loss, ckpt_path)

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: train the augmented mapping layer.")
    parser.add_argument("--task", type=str, required=True,
                        choices=["xnli", "xcsqa", "mgsm", "msvamp"],
                        help="Task being trained; controls the LLM prompt format.")
    parser.add_argument("--stage1-mapping-ckpt", type=str, required=True,
                        help="Path to the Stage 1 pytorch_model.bin checkpoint.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing translated *.jsonl training files.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory in which to save checkpoints.")
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
    logger = setup_logging(os.path.join(os.path.dirname(__file__), "logs"))
    main(args, logger)
