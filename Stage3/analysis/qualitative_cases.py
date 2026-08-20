"""Pull qualitative case studies by diffing two --results-jsonl runs.

Matches examples by `id` across a "before" run (e.g. Baseline, or Stage2
zero-shot) and an "after" run (e.g. the final Stage 3b checkpoint) on the
*same* benchmark+language, and buckets them into:

    fixed        before wrong, after correct  -- the mapping's payoff
    broken       before correct, after wrong  -- regressions worth inspecting
    both_wrong   both wrong                   -- shared failure modes
    both_correct both correct                 -- omitted by default (not interesting)

Writes a Markdown report with up to `--n-per-bucket` examples per bucket,
each showing the question, gold answer, both predictions, and an image
reference. Pass `--images-dir`/`--image-cache-dir` (matching evaluate.py's
own flags for the benchmark being diffed) to also copy the referenced image
files into `--copy-images-dir` for easy viewing alongside the report.

Example
-------
    python Stage3/analysis/qualitative_cases.py \\
        --before $SCRATCH/M2-ALIGN/Baseline/results/xgqa/bn.jsonl \\
        --after  $SCRATCH/M2-ALIGN/Stage3/results/stage3/xgqa/bn.jsonl \\
        --before-label Baseline --after-label "Stage 3 (M2RB)" \\
        --benchmark xgqa --images-dir /path/to/gqa/images \\
        --n-per-bucket 8 \\
        --out $SCRATCH/M2-ALIGN/Stage3/analysis_out/xgqa_bn_cases.md \\
        --copy-images-dir $SCRATCH/M2-ALIGN/Stage3/analysis_out/xgqa_bn_images

Produces (for one "fixed" example):

    ### fixed #1 -- id 07333423
    - **Question (bn):** ছবিতে কি কোনো গাড়ি আছে?
    - **Gold:** yes
    - **Baseline pred:** no  (WRONG)
    - **Stage 3 (M2RB) pred:** yes  (CORRECT)
    - **Image:** vg_image_id=2317323 -> copied to xgqa_bn_images/2317323.jpg

Useful for: reviewers and readers trust a handful of concrete, inspectable
examples more than an aggregate percentage. "fixed" examples let you argue
*what kind of error* the mapping resolves (grounding failures? language
understanding?); "broken" examples are the honest counterpoint every
Discussion section needs -- cases the baseline already had right that the
added machinery disturbs, which is evidence for e.g. a regularization or
capacity argument rather than pure gain framing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _gold_answer_str(row: dict) -> str:
    """Render the gold answer for any of xgqa/worldcuisines/cvqa's row schemas."""
    if "answer" in row:
        return str(row["answer"])
    if "answers" in row:
        return " / ".join(str(a) for a in row["answers"])
    if "choices" in row and "answer_index" in row:
        try:
            return str(row["choices"][int(row["answer_index"])])
        except (IndexError, TypeError, ValueError):
            return f"choice[{row.get('answer_index')}]"
    return "?"


def _pred_str(row: dict) -> str:
    return str(row.get("pred", "?"))


def _image_ref(row: dict) -> str:
    if "vg_image_id" in row:
        return f"vg_image_id={row['vg_image_id']}"
    if "image_url" in row:
        return f"image_url={row['image_url']}"
    return f"id={row.get('id')}"


def _resolve_local_image(row: dict, benchmark: str, images_dir: str | None, image_cache_dir: str | None) -> str | None:
    """Best-effort local file path for the row's image, or None.

    Mirrors evaluate.py's `load_image_by_id`/`load_image_by_url` lookup
    conventions (by extension for xgqa/cvqa, by URL hash for worldcuisines)
    without needing to import evaluate.py (which requires CUDA/torch/model
    imports this script has no other reason to pull in).
    """
    import hashlib

    if benchmark == "xgqa" and images_dir:
        for ext in (".jpg", ".jpeg", ".png"):
            path = os.path.join(images_dir, f"{row['vg_image_id']}{ext}")
            if os.path.exists(path):
                return path
        return None
    if benchmark == "cvqa" and image_cache_dir:
        path = os.path.join(image_cache_dir, f"{row['id']}.jpg")
        return path if os.path.exists(path) else None
    if benchmark.startswith("worldcuisines") and image_cache_dir and row.get("image_url"):
        cache_path = os.path.join(image_cache_dir, hashlib.sha1(row["image_url"].encode()).hexdigest() + ".jpg")
        return cache_path if os.path.exists(cache_path) else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", required=True, help="--results-jsonl from the 'before' run (e.g. Baseline).")
    parser.add_argument("--after", required=True, help="--results-jsonl from the 'after' run (e.g. Stage 3b).")
    parser.add_argument("--before-label", default="before")
    parser.add_argument("--after-label", default="after")
    parser.add_argument("--benchmark", required=True,
                         choices=["xgqa", "worldcuisines_task1", "worldcuisines_task2", "cvqa"],
                         help="Controls image-reference resolution only; both files must already be "
                              "for this same benchmark+language.")
    parser.add_argument("--images-dir", default=None, help="Local GQA images dir, for --benchmark xgqa.")
    parser.add_argument("--image-cache-dir", default=None,
                         help="Local image cache dir, for --benchmark cvqa/worldcuisines_task1/_task2.")
    parser.add_argument("--copy-images-dir", default=None,
                         help="If given, copy each shown example's resolved image file here.")
    parser.add_argument("--n-per-bucket", type=int, default=10)
    parser.add_argument("--include-both-correct", action="store_true",
                         help="Also sample a few both-correct examples (omitted by default -- not diagnostic).")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed before truncating each bucket to n-per-bucket.")
    parser.add_argument("--out", required=True, help="Output Markdown path.")
    args = parser.parse_args()

    before_by_id = {r["id"]: r for r in _read_jsonl(args.before)}
    after_by_id = {r["id"]: r for r in _read_jsonl(args.after)}
    shared_ids = sorted(set(before_by_id) & set(after_by_id))
    if not shared_ids:
        print("No shared ids between --before and --after -- wrong files/benchmark/language?", file=sys.stderr)
        sys.exit(1)

    buckets: dict[str, list[str]] = {"fixed": [], "broken": [], "both_wrong": [], "both_correct": []}
    for id_ in shared_ids:
        b_ok = bool(before_by_id[id_].get("correct"))
        a_ok = bool(after_by_id[id_].get("correct"))
        if not b_ok and a_ok:
            buckets["fixed"].append(id_)
        elif b_ok and not a_ok:
            buckets["broken"].append(id_)
        elif not b_ok and not a_ok:
            buckets["both_wrong"].append(id_)
        else:
            buckets["both_correct"].append(id_)

    import random
    rng = random.Random(args.seed)
    for ids in buckets.values():
        rng.shuffle(ids)

    if args.copy_images_dir:
        os.makedirs(args.copy_images_dir, exist_ok=True)

    bucket_order = ["fixed", "broken", "both_wrong"]
    if args.include_both_correct:
        bucket_order.append("both_correct")

    lines = [
        f"# Qualitative cases: {args.before_label} vs {args.after_label} ({args.benchmark})",
        "",
        f"- Shared examples: {len(shared_ids)}",
    ]
    for name in ("fixed", "broken", "both_wrong", "both_correct"):
        lines.append(f"- {name}: {len(buckets[name])} ({100.0 * len(buckets[name]) / len(shared_ids):.1f}%)")
    lines.append("")

    for bucket_name in bucket_order:
        ids = buckets[bucket_name][: args.n_per_bucket]
        lines.append(f"## {bucket_name} (showing {len(ids)} of {len(buckets[bucket_name])})")
        lines.append("")
        for i, id_ in enumerate(ids, 1):
            before_row, after_row = before_by_id[id_], after_by_id[id_]
            gold = _gold_answer_str(after_row)
            b_pred, a_pred = _pred_str(before_row), _pred_str(after_row)
            b_ok, a_ok = bool(before_row.get("correct")), bool(after_row.get("correct"))
            lines.append(f"### {bucket_name} #{i} -- id {id_}")
            lines.append(f"- **Question ({after_row.get('source_language', '?')}):** {after_row.get('query', '?')}")
            lines.append(f"- **Gold:** {gold}")
            lines.append(f"- **{args.before_label} pred:** {b_pred}  ({'CORRECT' if b_ok else 'WRONG'})")
            lines.append(f"- **{args.after_label} pred:** {a_pred}  ({'CORRECT' if a_ok else 'WRONG'})")

            image_line = f"- **Image:** {_image_ref(after_row)}"
            local_path = _resolve_local_image(after_row, args.benchmark, args.images_dir, args.image_cache_dir)
            if local_path and args.copy_images_dir:
                dest_name = f"{bucket_name}_{id_}{os.path.splitext(local_path)[1]}"
                shutil.copy(local_path, os.path.join(args.copy_images_dir, dest_name))
                image_line += f" -> copied to {os.path.join(args.copy_images_dir, dest_name)}"
            elif local_path:
                image_line += f" -> {local_path}"
            lines.append(image_line)
            lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.out} ({len(shared_ids)} shared examples)", file=sys.stderr)


if __name__ == "__main__":
    main()
