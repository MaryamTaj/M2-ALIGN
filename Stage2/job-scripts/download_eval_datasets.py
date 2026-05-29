"""Download all evaluation datasets to the HuggingFace cache before submitting
the offline SLURM job.

Run this on a login node (which has internet access):

    python download_eval_datasets.py

Each dataset is fetched once and stored under $HF_DATASETS_CACHE (or
~/.cache/huggingface/datasets by default).  The offline job then reads
directly from that cache without any network calls.
"""
from __future__ import annotations

from datasets import load_dataset

# (dataset_id, config, split)
DATASETS: list[tuple[str, str, str]] = [
    # MGSM — Swahili (juletxara/mgsm, ISO-2 config codes)
    ("juletxara/mgsm",       "sw",        "test"),
    # MSVAMP — Swahili (Mathoctopus/MSVAMP, ISO-2 config codes, fields: query/response)
    ("Mathoctopus/MSVAMP",   "sw",        "test"),
    # X-CSQA — Swahili (INK-USC/xcsr, config = "X-CSQA-{lang}", test split = 1000 rows)
    ("INK-USC/xcsr",         "X-CSQA-sw", "test"),
    # XNLI — Swahili
    ("xnli",                 "sw",        "test"),
    # AfriMGSM — Swahili, Wolof, Yoruba
    # masakhane/afrimgsm uses full FLORES config codes: swa, wol, yor
    ("masakhane/afrimgsm",   "swa",       "test"),
    ("masakhane/afrimgsm",   "wol",       "test"),
    ("masakhane/afrimgsm",   "yor",       "test"),
    # AfriXNLI — Swahili, Wolof, Yoruba
    # masakhane/afrixnli uses full FLORES config codes: swa, wol, yor
    ("masakhane/afrixnli",   "swa",       "test"),
    ("masakhane/afrixnli",   "wol",       "test"),
    ("masakhane/afrixnli",   "yor",       "test"),
]

OK, FAILED = [], []

for hf_id, config, split in DATASETS:
    label = f"{hf_id} [{config}/{split}]"
    try:
        ds = load_dataset(hf_id, config, split=split)
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
