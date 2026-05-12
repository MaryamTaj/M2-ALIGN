import argparse
import json
import logging
import os
from datetime import datetime

from datasets import load_dataset


LANG_TO_NLLB = {
    "English": "eng_Latn",
    "Swahili": "swh_Latn",
    "Yoruba": "yor_Latn",
    "Wolof": "wol_Latn",
    "French": "fra_Latn",
}


def setup_logging(log_dir: str) -> logging.Logger:
    """Create a logger that writes to stdout and a timestamped file in *log_dir*.

    Args:
        log_dir: Directory in which to create the log file (created if absent).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"load_nllb_corpus_{timestamp}.log")

    logger = logging.getLogger("load_nllb_corpus")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def download_and_save(
    source_lang: str,
    output_dir: str,
    split: str,
    samples_per_language: int,
    streaming: bool,
    logger: logging.Logger,
) -> None:
    """Download NLLB sentence pairs for one language and write them to JSONL.

    Streams the AllenAI NLLB dataset from the Hugging Face Hub and saves
    up to *samples_per_language* source↔English pairs to
    ``{output_dir}/{source_lang}_to_English.jsonl``.

    Args:
        source_lang: Human-readable source language name, e.g. ``"French"``.
            Must be a key in :data:`LANG_TO_NLLB`.
        output_dir: Directory in which to write the output JSONL file.
        split: Dataset split to stream (e.g. ``"train"``).
        samples_per_language: Stop after this many valid pairs.
        streaming: Whether to stream rows from the Hub.  Set to ``False``
            only when a full local cache is acceptable.
        logger: Logger for progress messages.

    Raises:
        ValueError: If *source_lang* is not in :data:`LANG_TO_NLLB`.
    """
    if source_lang not in LANG_TO_NLLB:
        raise ValueError(f"Unknown language {source_lang!r}; allowed: {sorted(LANG_TO_NLLB)}")

    source_code = LANG_TO_NLLB[source_lang]
    target_code = LANG_TO_NLLB["English"]
    config_name = f"{target_code}-{source_code}"

    ds = load_dataset("allenai/nllb", config_name, split=split, streaming=streaming)

    out_path = os.path.join(output_dir, f"{source_lang}_to_English.jsonl")
    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in ds:
            translation = sample.get("translation", {})
            source_text = translation.get(source_code, "").strip()
            target_text = translation.get(target_code, "").strip()
            if not source_text or not target_text:
                continue
            f.write(json.dumps({"source": source_text, "target": target_text}, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= samples_per_language:
                break

    logger.info("%s: wrote %d pairs to %s", source_lang, kept, out_path)


def main() -> None:
    """Entry point: parse arguments and download NLLB data for each language."""
    parser = argparse.ArgumentParser(description="Download NLLB sentence pairs from the Hugging Face Hub.")
    parser.add_argument("--output_dir", type=str, default="./data/nllb",
                        help="Directory in which to write the JSONL output files.")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to download (e.g. 'train').")
    parser.add_argument("--samples_per_language", type=int, default=3000,
                        help="Maximum number of sentence pairs to save per language.")
    parser.add_argument(
        "--languages", type=str, default="French",
        help="Comma-separated source language names matching keys in LANG_TO_NLLB "
             "(e.g. 'French' or 'Swahili,Yoruba,Wolof').",
    )
    parser.add_argument(
        "--streaming", action=argparse.BooleanOptionalAction, default=True,
        help="Stream rows from the Hub (default). Use --no-streaming for a full local cache.",
    )
    args = parser.parse_args()

    logger = setup_logging(os.path.join(os.path.dirname(__file__), "logs"))

    cache_hint = (
        os.environ.get("HF_DATASETS_CACHE")
        or os.environ.get("HF_HOME")
        or "~/.cache/huggingface"
    )
    logger.info(
        "load_nllb_corpus: streaming=%s; HF cache env effective root ~ %s",
        args.streaming, cache_hint,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    langs = [x.strip() for x in args.languages.split(",") if x.strip()]
    for name in langs:
        download_and_save(
            source_lang=name,
            output_dir=args.output_dir,
            split=args.split,
            samples_per_language=args.samples_per_language,
            streaming=args.streaming,
            logger=logger,
        )


if __name__ == "__main__":
    main()
