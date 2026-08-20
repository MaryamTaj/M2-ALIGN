"""Accuracy breakdown by a metadata field, over one or more results-jsonl files.

Reads the per-example JSONL written by Stage3/evaluate.py's or
Baseline/evaluate.py's `--results-jsonl` (one record per example: the full
eval row plus `pred`/`correct`), groups by `--group-by FIELD`, and reports
accuracy per group value -- optionally further split by `--lang` (each
input file is one language/checkpoint run, so multiple `--results` files
give a per-language x per-group table).

This one script covers three of the Discussion-section analyses, since
they're all the same operation over different fields already carried in
the eval rows (see load_evaluation_data.py):

    xGQA structural/semantic question types:
        --group-by question_type_structural
        --group-by question_type_semantic

    CVQA category/country:
        --group-by category
        --group-by country

    WorldCuisines role of context (task1 only -- task2 is single-value):
        --group-by prompt_type
        (1=no-context, 3=contextualized, 4=adversarial; see
        load_evaluation_data.py's load_worldcuisines docstring)

Example
-------
    python Stage3/analysis/aggregate_breakdown.py \\
        --group-by question_type_structural \\
        --results $SCRATCH/M2-ALIGN/Stage3/results/stage3/xgqa/bn.jsonl:bn \\
        --results $SCRATCH/M2-ALIGN/Stage3/results/stage3/xgqa/de.jsonl:de \\
        --out $SCRATCH/M2-ALIGN/Stage3/analysis_out/xgqa_by_structural_type.csv

Produces a table like:

    lang   group      n      accuracy
    bn     verify     412    61.7
    bn     query      803    38.2
    bn     choose     198    70.7
    bn     logical    255    45.5
    bn     compare    121    52.9
    de     verify     412    68.0
    ...

Useful for: pinpointing *which kind of reasoning* the mapping helps or
hurts -- e.g. if `compare`/`logical` (multi-hop, relational) trail
`verify`/`choose` (near single-lookup) far more than the aggregate gap
would suggest, that's evidence the mapping's benefit is concentrated in
simple grounding rather than compositional reasoning, a concrete claim for
the Discussion section rather than "accuracy is X%".
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_results_arg(spec: str) -> tuple[str, str | None]:
    """Parse `--results PATH[:LABEL]` into `(path, label)`.

    LABEL defaults to the run's own `source_language` field (read from its
    first row) when omitted -- explicit LABEL is only needed to disambiguate
    two files for the same language (e.g. Stage2 vs Stage3 checkpoints).
    """
    if ":" in spec:
        path, label = spec.rsplit(":", 1)
        return path, label
    return spec, None


def aggregate(records: list[dict], group_by: str) -> dict[str | None, tuple[int, int]]:
    """Return `{group_value: (n_correct, n_total)}`."""
    counts: dict[str | None, list[int]] = defaultdict(lambda: [0, 0])
    for rec in records:
        group_val = rec.get(group_by)
        correct = bool(rec.get("correct"))
        counts[group_val][1] += 1
        counts[group_val][0] += int(correct)
    return {k: (v[0], v[1]) for k, v in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group-by", required=True,
                         help="Field name to break accuracy down by, e.g. question_type_structural, "
                              "category, country, prompt_type.")
    parser.add_argument("--results", action="append", required=True,
                         help="Path to a --results-jsonl file from evaluate.py, optionally suffixed "
                              "':LABEL' (default: that run's source_language). Repeat for multiple "
                              "languages/checkpoints.")
    parser.add_argument("--min-n", type=int, default=1,
                         help="Drop group values with fewer than this many examples (default: 1, i.e. keep all).")
    parser.add_argument("--out", default=None, help="Write the table as CSV here; also printed to stdout.")
    args = parser.parse_args()

    fieldnames = ["lang", "group", "n", "accuracy"]
    table_rows = []

    for spec in args.results:
        path, label = _parse_results_arg(spec)
        records = _read_jsonl(path)
        if not records:
            print(f"warning: {path} has no records, skipping", file=sys.stderr)
            continue
        lang = label or records[0].get("source_language") or path
        counts = aggregate(records, args.group_by)
        for group_val, (n_correct, n_total) in sorted(counts.items(), key=lambda kv: str(kv[0])):
            if n_total < args.min_n:
                continue
            acc = 100.0 * n_correct / n_total if n_total else 0.0
            table_rows.append({"lang": lang, "group": group_val, "n": n_total, "accuracy": round(acc, 2)})

    if not table_rows:
        print("No records aggregated -- check --results paths and --group-by field name.", file=sys.stderr)
        sys.exit(1)

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
