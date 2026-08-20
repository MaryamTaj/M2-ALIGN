"""Compare WorldCuisines task1 (Dish Name) vs. task2 (Dish Origin) accuracy.

Task1 rows are a 3-way mix of prompt_type 1 (no-context), 3 (contextualized:
location named in the question) and 4 (adversarial: a distractor cuisine
named in the question) -- see load_evaluation_data.py's load_worldcuisines
docstring. Task2 rows are entirely prompt_type=2 (no-context origin
question). Read together, task1's no-context subset (prompt_type=1) is the
fair apples-to-apples comparison against task2: same "no context given"
condition, different question (name the dish vs. name its origin).

This reads the same --results-jsonl files aggregate_breakdown.py does, but
does a fixed two-benchmark join per language rather than a generic groupby,
since task1/task2 live in separate result files and task2 has no
prompt_type variation to break down on its own.

Example
-------
    python Stage3/analysis/worldcuisines_task_comparison.py \\
        --task1 $SCRATCH/M2-ALIGN/Stage3/results/stage3/worldcuisines_task1/bn.jsonl \\
        --task2 $SCRATCH/M2-ALIGN/Stage3/results/stage3/worldcuisines_task2/bn.jsonl \\
        --lang bn \\
        --out $SCRATCH/M2-ALIGN/Stage3/analysis_out/worldcuisines_bn_task_comparison.csv

Produces a table like:

    lang   condition                    n     accuracy
    bn     task1_no_context (pt=1)      100   54.0
    bn     task1_contextualized (pt=3)  100   71.0
    bn     task1_adversarial (pt=4)     100   49.0
    bn     task2_origin (pt=2)          80    38.5

Useful for: two separable claims. (1) task1_no_context vs. task2_origin
isolates whether the model finds *naming* a dish or *placing* it harder --
these use the same image and the same "no extra context" condition, so any
gap reflects task difficulty, not context. (2) task1_no_context vs.
task1_contextualized isolates the effect of giving the model the answer's
location directly in the prompt -- a clean context-helps-recognition
ablation for free, since WorldCuisines already varies this by design.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

_PROMPT_TYPE_LABELS = {
    1: "task1_no_context (pt=1)",
    2: "task2_origin (pt=2)",
    3: "task1_contextualized (pt=3)",
    4: "task1_adversarial (pt=4)",
}


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _accuracy_by_prompt_type(records: list[dict]) -> dict[int | None, tuple[int, int]]:
    counts: dict[int | None, list[int]] = {}
    for rec in records:
        pt = rec.get("prompt_type")
        n_correct, n_total = counts.setdefault(pt, [0, 0])
        counts[pt][1] += 1
        counts[pt][0] += int(bool(rec.get("correct")))
    return {k: (v[0], v[1]) for k, v in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task1", required=True, help="--results-jsonl output for --benchmark worldcuisines_task1.")
    parser.add_argument("--task2", required=True, help="--results-jsonl output for --benchmark worldcuisines_task2.")
    parser.add_argument("--lang", required=True, help="Language label for the output table.")
    parser.add_argument("--out", default=None, help="Write the table as CSV here; also printed to stdout.")
    args = parser.parse_args()

    task1_records = _read_jsonl(args.task1)
    task2_records = _read_jsonl(args.task2)
    if not task1_records:
        print(f"warning: {args.task1} has no records", file=sys.stderr)
    if not task2_records:
        print(f"warning: {args.task2} has no records", file=sys.stderr)

    by_pt = _accuracy_by_prompt_type(task1_records)
    by_pt.update(_accuracy_by_prompt_type(task2_records))

    fieldnames = ["lang", "condition", "n", "accuracy"]
    table_rows = []
    for pt in sorted(by_pt, key=lambda x: (x is None, x)):
        n_correct, n_total = by_pt[pt]
        acc = 100.0 * n_correct / n_total if n_total else 0.0
        label = _PROMPT_TYPE_LABELS.get(pt, f"unknown (pt={pt})")
        table_rows.append({"lang": args.lang, "condition": label, "n": n_total, "accuracy": round(acc, 2)})

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(table_rows)

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(table_rows)
        print(f"\nWrote {len(table_rows)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
