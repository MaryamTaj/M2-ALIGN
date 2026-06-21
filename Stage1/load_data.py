"""Download 100K Swahili-English sentence pairs from the NLLB corpus.

Streams allenai/nllb from the Hugging Face Hub and writes up to
--n_samples Swahili-English pairs to a single JSONL file.

Each output line:
    {"source": "<Swahili text>", "target": "<English text>"}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime

from datasets import load_dataset

NLLB_SWAHILI = "swh_Latn"
NLLB_ENGLISH = "eng_Latn"
NLLB_CONFIG  = f"{NLLB_ENGLISH}-{NLLB_SWAHILI}"


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"load_data_{timestamp}.log")
    logger = logging.getLogger("stage1_load_data")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NLLB Swahili-English sentence pairs.")
    parser.add_argument("--output_dir", type=str, default="./data/stage1",
                        help="Directory to write the output JSONL file.")
    parser.add_argument("--n_samples", type=int, default=100_000,
                        help="Number of sentence pairs to download (default: 100000).")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True,
                        help="Stream from the Hub (default). Use --no-streaming for a full local cache.")
    args = parser.parse_args()

    logger = setup_logging(os.path.join(os.path.dirname(__file__), "logs"))
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading NLLB config=%s split=%s target=%d samples", NLLB_CONFIG, args.split, args.n_samples)
    ds = load_dataset("allenai/nllb", NLLB_CONFIG, split=args.split, streaming=args.streaming)

    out_path = os.path.join(args.output_dir, "Swahili_to_English.jsonl")
    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in ds:
            translation = sample.get("translation", {})
            sw_text = translation.get(NLLB_SWAHILI, "").strip()
            en_text = translation.get(NLLB_ENGLISH, "").strip()
            if not sw_text or not en_text:
                continue
            f.write(json.dumps({"source": sw_text, "target": en_text}, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 10_000 == 0:
                logger.info("Progress: %d / %d", kept, args.n_samples)
            if kept >= args.n_samples:
                break

    logger.info("Done — wrote %d pairs to %s", kept, out_path)


if __name__ == "__main__":
    main()
