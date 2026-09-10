"""Convert Stage3/data/english.jsonl into a Stage 3b-augmentation JSONL.

MindMerger's augmentation stage trains on the task queries in every target
language *including English* -- the English portion is just the original,
untranslated training queries (paper Table 17: "N languages incl. English").
This makes the same happen for the pooled Stage 3b run: it emits English GQA
rows in the schema VQADataset expects (query / answer / vg_image_id /
nllb_lang_tag), so `english_aug.jsonl` can be dropped into the pooled data
dir alongside the 11 translated {lang}.jsonl files.

Field remap:
    english.jsonl:      question, answer, vg_image_id
    english_aug.jsonl:  vg_image_id, query=question, answer,
                        source_language="English", nllb_lang_tag="eng_Latn"

Usage:
    python Stage3/make_english_aug.py \\
        $SCRATCH/M2-ALIGN/Stage3/data/english.jsonl \\
        $SCRATCH/M2-ALIGN/Stage3/data/english_aug.jsonl
"""
from __future__ import annotations

import json
import sys


def main(src: str, dst: str) -> None:
    n = 0
    with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            g.write(json.dumps({
                "vg_image_id": r["vg_image_id"],
                "query": r["question"],
                "answer": r["answer"],
                "source_language": "English",
                "nllb_lang_tag": "eng_Latn",
            }) + "\n")
            n += 1
    print(f"wrote {n} English augmentation rows -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: make_english_aug.py <english.jsonl> <english_aug.jsonl>")
    main(sys.argv[1], sys.argv[2])
