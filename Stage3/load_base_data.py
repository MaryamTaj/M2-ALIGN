"""Sample GQA's balanced train split to a portable JSONL, with no GPU/torch
dependency -- meant to run on a workstation with internet access, not on
Narval (see Stage3/load_translated_data.py, which needs this file's output rather
than calling `load_dataset` itself).

Why: `load_translated_data.py --gqa-jsonl` used to rely on the Narval login node
pre-warming `lmms-lab/GQA`'s raw HuggingFace `datasets` cache, then reading
it back with HF_HUB_OFFLINE=1 on a compute node. That's the same
unversioned/unreadable-Arrow-cache problem `load_text_evaluation.py`'s
docstring already flags for a different dataset, and it still meant a
`load_dataset()` network call on the login node. Sampling here instead and
writing plain JSONL removes that dependency entirely: the sample is
deterministic in (n_samples, seed) and independent of target language (see
`sample_gqa` in load_translated_data.py), so it only needs to be produced once and
Globus-transferred over, no matter how many languages read it afterward.

Output JSONL fields per row: question, answer, vg_image_id

Usage
-----
    python Stage3/load_base_data.py --n_samples 30000 --seed 42 \\
        --output ./gqa_raw_sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from datasets import load_dataset

GQA_HF_ID = "lmms-lab/GQA"
GQA_CONFIG = "train_balanced_instructions"
GQA_SPLIT = "train"

# Cap on questions drawn from any single image so the sample isn't dominated
# by a handful of heavily-annotated images (mirrors load_translated_data.py).
MAX_QUESTIONS_PER_IMAGE = 4


def _row_image_id(row: dict) -> str | None:
    for key in ("imageId", "image_id", "imageid"):
        val = row.get(key)
        if val is not None:
            return str(val)
    image = row.get("image")
    if isinstance(image, dict):
        for key in ("id", "image_id"):
            if image.get(key) is not None:
                return str(image[key])
    return None


def _row_question(row: dict) -> str | None:
    for key in ("question", "question_text", "sent"):
        val = row.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return None


def _row_answer(row: dict) -> str | None:
    for key in ("answer", "multiple_choice_answer", "label"):
        val = row.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return None


def sample_gqa(n_samples: int, seed: int) -> list[dict]:
    """Identical sampling logic to load_translated_data.py's sample_gqa -- keep in
    sync if either changes, so a JSONL produced here matches what an
    in-process call on Narval would have sampled."""
    print(f"Loading {GQA_HF_ID} [{GQA_CONFIG}/{GQA_SPLIT}] ...")
    ds = load_dataset(GQA_HF_ID, GQA_CONFIG, split=GQA_SPLIT)
    print(f"GQA balanced train: {len(ds)} rows total")

    per_image_count: dict[str, int] = defaultdict(int)
    rows: list[dict] = []
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    for i in indices:
        r = ds[i]
        image_id = _row_image_id(r)
        question = _row_question(r)
        answer = _row_answer(r)
        if not image_id or not question or not answer:
            continue
        if per_image_count[image_id] >= MAX_QUESTIONS_PER_IMAGE:
            continue
        per_image_count[image_id] += 1
        rows.append({"question": question, "answer": answer, "vg_image_id": image_id})
        if len(rows) >= n_samples:
            break

    print(f"Sampled {len(rows)} GQA rows over {len(per_image_count)} unique images "
          f"(cap={MAX_QUESTIONS_PER_IMAGE}/image)")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample GQA balanced train to portable JSONL.")
    parser.add_argument("--n_samples", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./gqa_raw_sample.jsonl")
    args = parser.parse_args()

    rows = sample_gqa(args.n_samples, args.seed)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
