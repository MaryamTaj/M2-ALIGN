"""Build Stage 1 data: sentence pairs from allenai/nllb, each language paired with English.

Needs internet, so run it on the workstation. It also needs datasets<4.0,
because allenai/nllb ships a loading script that datasets 4.0 no longer runs.

    python data/build_stage1.py [--langs bn,ru] [--overwrite]

Writes data/stage1/{lang}.jsonl with N_PER_LANGUAGE rows each:
    {"task": "translation", "lang": "bn", "source": "...", "target": "..."}
Files that already exist are skipped unless --overwrite is given.
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN")
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_LANGS = ["bn", "ru", "de", "zh", "pt", "id", "ko", "jv", "mn", "si", "ga"]
NLLB_TAGS = {
    "en": "eng_Latn", "bn": "ben_Beng", "ru": "rus_Cyrl", "de": "deu_Latn", "zh": "zho_Hans",
    "pt": "por_Latn", "id": "ind_Latn", "ko": "kor_Hang", "jv": "jav_Latn", "mn": "khk_Cyrl",
    "si": "sin_Sinh", "ga": "gle_Latn",
}
N_PER_LANGUAGE = 100_000


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows):>7} rows -> {path}", flush=True)


def download_language(lang: str) -> list[dict]:
    """The first N_PER_LANGUAGE pairs where both sides are non-empty."""
    from datasets import load_dataset

    tag, english_tag = NLLB_TAGS[lang], NLLB_TAGS["en"]
    config = "-".join(sorted([english_tag, tag]))  # allenai/nllb names each pair alphabetically
    stream = load_dataset("allenai/nllb", config, split="train", streaming=True, trust_remote_code=True)
    rows = []
    for sample in stream:
        pair = sample.get("translation") or {}
        source, english = pair.get(tag, "").strip(), pair.get(english_tag, "").strip()
        if source and english:
            rows.append({"task": "translation", "lang": lang, "source": source, "target": english})
            if len(rows) >= N_PER_LANGUAGE:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 1 sentence pairs.")
    parser.add_argument("--langs", default=",".join(TRAIN_LANGS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for lang in args.langs.split(","):
        path = os.path.join(DATA_DIR, "stage1", f"{lang}.jsonl")
        if os.path.exists(path) and not args.overwrite:
            print(f"exists, skipped: {path}")
            continue
        write_jsonl(path, download_language(lang))


if __name__ == "__main__":
    main()
