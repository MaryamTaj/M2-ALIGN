"""Load Stage 2 Swahili training data from HuggingFace Hub, mirroring MindMerger.

Tasks and sources
-----------------
mgsm / msvamp  → 30,000 samples from Mathoctopus/Mathoctopus_parallel_train
                  (Chen et al., 2023 multilingual math data, Swahili rows)
xnli           → 2,490 samples from xnli Swahili validation split
xcsqa          → 8,888 samples from INK-USC/xcsr X-CSQA-sw train split

Each task writes one JSONL to --output_dir/<task>.jsonl with fields:
    query            – Swahili input text
    answer           – expected response (English for math/XNLI label for NLI/letter for CSQA)
    source_language  – "Swahili" (used by Stage2/train.py for NLLB tokenization)
    source_dataset   – dataset identifier string
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
from datetime import datetime

from datasets import load_dataset

TASK_DEFAULTS: dict[str, int] = {
    "mgsm":   30_000,
    "msvamp": 30_000,
    "xnli":    2_940,
    "xcsqa":   8_888,
}

_XNLI_INT_TO_STR = {0: "entailment", 1: "neutral", 2: "contradiction"}


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"load_data_{timestamp}.log")
    logger = logging.getLogger("stage2_load_data")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def write_jsonl(path: str, rows: list[dict], logger: logging.Logger) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows → %s", len(rows), path)


# ---------------------------------------------------------------------------
# Math — Chen et al. (2023) via Mathoctopus/Mathoctopus_parallel_train
# ---------------------------------------------------------------------------

def load_math(n_samples: int, seed: int, logger: logging.Logger) -> list[dict]:
    """Stream Mathoctopus_parallel_train and collect n_samples Swahili rows."""
    logger.info("Streaming Mathoctopus/Mathoctopus_parallel_train for Swahili ...")
    ds = load_dataset(
        "Mathoctopus/Mathoctopus_parallel_train",
        split="train",
        streaming=True,
    )

    sw_rows: list[dict] = []
    for sample in ds:
        # The dataset uses the full language name as the lang field.
        if sample.get("lang") != "Swahili":
            continue
        query = sample.get("query", "").strip()
        response = sample.get("response", "").strip()
        if not query or not response:
            continue
        sw_rows.append({
            "id": stable_id(f"math_{query}"),
            "query": query,
            "answer": response,
            "source_language": "Swahili",
            "source_dataset": "mathoctopus_parallel_train",
        })
        if len(sw_rows) >= n_samples * 2:
            break  # Collect a buffer, then sample down to avoid full scan.

    logger.info("Collected %d Swahili math rows before sampling.", len(sw_rows))
    n = min(n_samples, len(sw_rows))
    if n < n_samples:
        logger.warning("Requested %d but only %d available; using all.", n_samples, n)
    sampled = random.Random(seed).sample(sw_rows, n)
    return sampled


# ---------------------------------------------------------------------------
# XNLI — xnli Swahili validation split
# ---------------------------------------------------------------------------

def load_xnli(n_samples: int, seed: int, logger: logging.Logger) -> list[dict]:
    """Load XNLI Swahili validation rows."""
    logger.info("Loading xnli sw validation ...")
    ds = load_dataset("xnli", "sw", split="validation")
    logger.info("XNLI sw validation: %d rows", len(ds))

    rows: list[dict] = []
    for r in ds:
        label_int = int(r["label"])
        label_str = _XNLI_INT_TO_STR.get(label_int, "entailment")
        query = f"{r['premise']}\n{r['hypothesis']}"
        rows.append({
            "id": stable_id(f"xnli_{r['premise']}_{r['hypothesis']}"),
            "query": query,
            "answer": label_str,
            "source_language": "Swahili",
            "source_dataset": "xnli_sw_validation",
        })

    n = min(n_samples, len(rows))
    if n < n_samples:
        logger.warning("Requested %d but only %d available; using all.", n_samples, n)
    return random.Random(seed).sample(rows, n)


# ---------------------------------------------------------------------------
# X-CSQA — INK-USC/xcsr X-CSQA-sw train split
# ---------------------------------------------------------------------------

def load_xcsqa(n_samples: int, seed: int, logger: logging.Logger) -> list[dict]:
    """Load X-CSQA Swahili training rows."""
    logger.info("Loading INK-USC/xcsr X-CSQA-sw train ...")
    ds = load_dataset("INK-USC/xcsr", "X-CSQA-sw", split="train")
    logger.info("X-CSQA-sw train: %d rows", len(ds))

    rows: list[dict] = []
    for r in ds:
        stem: str = r["question"]["stem"]
        choices: dict = r["question"]["choices"]
        # HuggingFace stores choices as {"label": [...], "text": [...]}
        labels: list[str] = choices["label"]
        texts: list[str] = choices["text"]
        options_block = "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(labels, texts))
        query = f"{stem}\n{options_block}"
        rows.append({
            "id": r["id"],
            "query": query,
            "answer": r["answerKey"],
            "source_language": "Swahili",
            "source_dataset": "xcsqa_sw_train",
        })

    n = min(n_samples, len(rows))
    if n < n_samples:
        logger.warning("Requested %d but only %d available; using all.", n_samples, n)
    return random.Random(seed).sample(rows, n)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

LOADERS = {
    "mgsm":   load_math,
    "msvamp": load_math,
    "xnli":   load_xnli,
    "xcsqa":  load_xcsqa,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Stage 2 Swahili training data from HuggingFace (MindMerger protocol)."
    )
    parser.add_argument("--task", required=True, choices=list(LOADERS),
                        help="Task to download: mgsm, msvamp, xnli, or xcsqa.")
    parser.add_argument("--output_dir", type=str, default="./data/stage2",
                        help="Directory to write the output JSONL file.")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples (defaults to MindMerger paper values per task).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logging(os.path.join(os.path.dirname(__file__), "logs"))

    n_samples = args.n_samples if args.n_samples is not None else TASK_DEFAULTS[args.task]
    logger.info("Task=%s  n_samples=%d", args.task, n_samples)

    rows = LOADERS[args.task](n_samples, args.seed, logger)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.task}.jsonl")
    write_jsonl(out_path, rows, logger)
    logger.info("Done.")


if __name__ == "__main__":
    main()
