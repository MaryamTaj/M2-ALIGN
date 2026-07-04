"""Download all evaluation datasets to the HuggingFace cache before submitting
the offline SLURM job.

Run this on a login node (which has internet access):

    python load_text_evaluation_data.py

Each dataset is fetched once and stored under $HF_DATASETS_CACHE (or
~/.cache/huggingface/datasets by default).  The offline job then reads
directly from that cache without any network calls.
"""
from __future__ import annotations

from datasets import load_dataset

# (dataset_id, config, split)
#
# xnli and x-csqa are not listed here: standard XNLI's language list is
# ar/bg/de/el/en/es/fr/hi/ru/sw/th/tr/ur/vi/zh -- "bg" is Bulgarian, not
# Bengali -- and INK-USC/xcsr (X-CSQA) has the same gap. Both tasks were
# dropped from Stage 3a training and evaluation entirely for now.
DATASETS: list[tuple[str, str | None, str]] = [
    # MGSM — English + Swahili + Bengali (Bengali is the Stage 3 VQA-track
    # language; mgsm/msvamp already have native bn coverage)
    ("juletxara/mgsm",       "en",        "test"),
    ("juletxara/mgsm",       "sw",        "test"),
    ("juletxara/mgsm",       "bn",        "test"),
    # MSVAMP — English + Swahili + Bengali
    ("Mathoctopus/MSVAMP",   "en",        "test"),
    ("Mathoctopus/MSVAMP",   "sw",        "test"),
    ("Mathoctopus/MSVAMP",   "bn",        "test"),
    # AfriMGSM — Swahili, Wolof, Yoruba
    # masakhane/afrimgsm uses full FLORES config codes: swa, wol, yor
    ("masakhane/afrimgsm",   "swa",       "test"),
    ("masakhane/afrimgsm",   "wol",       "test"),
    ("masakhane/afrimgsm",   "yor",       "test"),
]

OK, FAILED = [], []

for hf_id, config, split in DATASETS:
    label = f"{hf_id} [{config}/{split}]"
    try:
        ds = load_dataset(hf_id, config, split=split) if config else load_dataset(hf_id, split=split)
        print(f"  OK  {label}  ({len(ds)} examples)")
        OK.append(label)
    except Exception as exc:
        print(f"  FAIL  {label}  →  {exc}")
        FAILED.append(label)

print(f"\n{len(OK)} downloaded, {len(FAILED)} failed.")
if FAILED:
    print("Failed datasets — verify the HuggingFace IDs and re-run:")
    for f in FAILED:
        print(f"  {f}")
