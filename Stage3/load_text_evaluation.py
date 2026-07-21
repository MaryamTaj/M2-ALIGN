"""Download the MGSM/MSVAMP/AfriMGSM evaluation sets and materialize them to JSONL.

Every other data source in this pipeline (Stage1/load_text.py,
Stage2/load_image.py, Stage3/load_text_data.py, Stage3/load_vqa_data.py,
Stage3/load_vqa_evaluation.py) is fetched once and written to a local JSONL
file, which train/evaluate scripts then read offline with no `datasets`
network calls of their own. This script used to be the one exception: it
only warmed the shared HuggingFace `datasets` cache, and
Stage3/evaluate_text.py (and Baseline/evaluate_text.py) called
`load_dataset(...)` directly at eval time, trusting that cache to still be
populated. That made evaluation runs depend on an unversioned, unreadable
Arrow cache instead of an inspectable file, and silently coupled Baseline's
eval to whatever Stage3 happened to have downloaded. Writing JSONL here
fixes both.

Run this on a login node (which has internet access):

    python Stage3/load_text_evaluation.py

Output layout (resolved relative to this script's location, not the cwd):

    Stage3/data/stage3a_eval/mgsm/{lang}.jsonl
    Stage3/data/stage3a_eval/msvamp/{lang}.jsonl
    Stage3/data/stage3a_eval/afrimgsm/{lang}.jsonl

Both Stage3/evaluate_text.py and Baseline/evaluate_text.py read from this
same directory by default, so both runs score the exact same rows.

Usage for the low-resource languages (am/ig/om; Yoruba dropped, xGQA/
WorldCuisines have no coverage for any of these four so CVQA covers the
VQA side instead -- see Stage3/load_vqa_evaluation.py):

    python Stage3/load_text_evaluation.py --tasks afrimgsm --languages am ig om
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime

from datasets import load_dataset

# (task, hf_id, default languages)
#
# xnli and x-csqa are not listed here: standard XNLI's language list is
# ar/bg/de/el/en/es/fr/hi/ru/sw/th/tr/ur/vi/zh -- "bg" is Bulgarian, not
# Bengali -- and INK-USC/xcsr (X-CSQA) has the same gap. Both tasks were
# dropped from Stage 3a training and evaluation entirely for now.
TASKS: dict[str, str] = {
    "mgsm": "juletxara/mgsm",
    "msvamp": "Mathoctopus/MSVAMP",
    "afrimgsm": "masakhane/afrimgsm",
}

DEFAULT_LANGS: list[str] = ["en"]

# afrimgsm's HF configs use 3-letter codes, not our pipeline's 2-letter ISO
# codes (am/ig/om) used everywhere else (NLLB_CODES, CVQA_SUBSET_CANDIDATES,
# WIT_TO_NLLB, ...) -- these are a different naming scheme for the same
# language, not a typo; don't conflate "orm" (this dataset's config key)
# with NLLB's "gaz_Latn" FLORES tag for Oromo.
AFRIMGSM_HF_CONFIG: dict[str, str] = {
    "am": "amh",
    "ig": "ibo",
    "om": "orm",
}


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("stage3a_load_text_evaluation")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"load_text_evaluation_{ts}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def write_jsonl(path: str, rows: list[dict], logger: logging.Logger) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows -> %s", len(rows), path)


def load_mgsm(lang: str) -> list[dict]:
    ds = load_dataset("juletxara/mgsm", lang, split="test")
    return [
        {"question": str(r["question"]),
         "answer": str(r.get("answer") or r.get("answer_number") or "").replace(",", "")}
        for r in ds
    ]


def load_msvamp(lang: str) -> list[dict]:
    ds = load_dataset("Mathoctopus/MSVAMP", lang, split="test")
    return [
        {"question": str(r.get("m_query") or r.get("query") or ""),
         "answer": str(r.get("response") or r.get("answer") or "").replace(",", "")}
        for r in ds
    ]


def load_afrimgsm(lang: str) -> list[dict]:
    """Load AfriMGSM's test split (250 rows/language, same size as MGSM).

    Row schema: question, answer (often null), answer_number, equation_solution
    -- mirrors load_mgsm's defensive field selection since "answer" isn't
    always populated.
    """
    if lang not in AFRIMGSM_HF_CONFIG:
        raise ValueError(f"No AfriMGSM coverage configured for {lang!r}; allowed: {sorted(AFRIMGSM_HF_CONFIG)}")
    hf_config = AFRIMGSM_HF_CONFIG[lang]
    ds = load_dataset("masakhane/afrimgsm", hf_config, split="test")
    return [
        {"question": str(r["question"]),
         "answer": str(r.get("answer") or r.get("answer_number") or "").replace(",", "")}
        for r in ds
    ]


_LOADERS = {"mgsm": load_mgsm, "msvamp": load_msvamp, "afrimgsm": load_afrimgsm}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MGSM/MSVAMP evaluation sets and write them to JSONL."
    )
    parser.add_argument(
        "--tasks", nargs="+", default=list(TASKS), choices=list(TASKS),
        help="Which benchmarks to fetch (default: both).",
    )
    parser.add_argument(
        "--languages", nargs="+", default=DEFAULT_LANGS,
        help="ISO 639-1 language codes (default: en sw bn).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Defaults to Stage3/data/stage3a_eval (resolved relative to this "
             "script's location, not the current working directory).",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "data", "stage3a_eval")

    logger = setup_logging(os.path.join(script_dir, "logs"))

    ok, failed = [], []
    for task in args.tasks:
        loader = _LOADERS[task]
        for lang in args.languages:
            label = f"{task} [{TASKS[task]}/{lang}]"
            try:
                rows = loader(lang)
                out_path = os.path.join(output_dir, task, f"{lang}.jsonl")
                write_jsonl(out_path, rows, logger)
                ok.append(label)
            except Exception as exc:
                logger.error("FAIL  %s  ->  %s", label, exc)
                failed.append(label)

    logger.info("%d written, %d failed.", len(ok), len(failed))
    if failed:
        logger.info("Failed — verify the HuggingFace IDs/language codes and re-run:")
        for f in failed:
            logger.info("  %s", f)


if __name__ == "__main__":
    main()
