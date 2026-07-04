"""Download sentence pairs from the NLLB corpus for one or more languages.

Streams allenai/nllb from the Hugging Face Hub and writes up to
--n_samples pairs per language to separate JSONL files.

Each output line:
    {"source": "<source text>", "target": "<English text>"}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime

from datasets import load_dataset

NLLB_ENGLISH = "eng_Latn"

LANG_TO_NLLB = {
    "English": "eng_Latn",
    "Swahili": "swh_Latn",
    "Yoruba": "yor_Latn",
    "Wolof": "wol_Latn",
    "French": "fra_Latn",
    # Stage 3 VQA-track language (see Stage3 plan): needs its own NLLB
    # mapping/translation checkpoint before Stage 2/3 can use it.
    # Indonesian/Javanese were scoped out for now -- add back the same way
    # if/when the VQA track is widened again.
    "Bengali": "ben_Beng",
}


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


def download_language(
    source_lang: str,
    output_dir: str,
    split: str,
    n_samples: int,
    streaming: bool,
    logger: logging.Logger,
) -> None:
    if source_lang not in LANG_TO_NLLB:
        raise ValueError(f"Unknown language {source_lang!r}; allowed: {sorted(LANG_TO_NLLB)}")

    source_code = LANG_TO_NLLB[source_lang]
    config = f"{NLLB_ENGLISH}-{source_code}"

    logger.info("Loading NLLB config=%s split=%s target=%d samples", config, split, n_samples)
    ds = load_dataset("allenai/nllb", config, split=split, streaming=streaming)

    out_path = os.path.join(output_dir, f"{source_lang}_to_English.jsonl")
    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in ds:
            translation = sample.get("translation", {})
            source_text = translation.get(source_code, "").strip()
            en_text = translation.get(NLLB_ENGLISH, "").strip()
            if not source_text or not en_text:
                continue
            f.write(json.dumps({"source": source_text, "target": en_text}, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 10_000 == 0:
                logger.info("%s progress: %d / %d", source_lang, kept, n_samples)
            if kept >= n_samples:
                break

    logger.info("%s: wrote %d pairs to %s", source_lang, kept, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NLLB sentence pairs.")
    parser.add_argument("--output_dir", type=str, default="./data/stage1",
                        help="Directory to write the output JSONL files.")
    parser.add_argument("--n_samples", type=int, default=100_000,
                        help="Number of sentence pairs to download per language (default: 100000).")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--languages", type=str, default="Swahili",
                        help="Comma-separated source language names, e.g. 'Swahili' or 'Swahili,Yoruba,Wolof'.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True,
                        help="Stream from the Hub (default). Use --no-streaming for a full local cache.")
    args = parser.parse_args()

    logger = setup_logging(os.path.join(os.path.dirname(__file__), "logs"))
    os.makedirs(args.output_dir, exist_ok=True)

    langs = [x.strip() for x in args.languages.split(",") if x.strip()]
    for lang in langs:
        download_language(lang, args.output_dir, args.split, args.n_samples, args.streaming, logger)


if __name__ == "__main__":
    main()
