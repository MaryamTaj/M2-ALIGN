from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Iterable

from datasets import load_dataset


def normalize_question(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def extract_mmlu_prox_en_test_questions() -> set[str]:
    ds = load_dataset("li-lab/MMLU-ProX", "en", split="test")
    return {normalize_question(x["question"]) for x in ds}


def iter_mmlu_aux_examples() -> Iterable[dict]:
    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    for ex in ds:
        choices = ex.get("choices", [])
        answer_idx = int(ex.get("answer", 0))
        answer = choices[answer_idx] if 0 <= answer_idx < len(choices) else str(ex.get("answer", ""))
        yield {
            "question": ex["question"],
            "answer": answer,
            "source_dataset": "mmlu_auxiliary_train",
        }


def iter_arc_examples() -> Iterable[dict]:
    for config in ["ARC-Challenge", "ARC-Easy"]:
        ds = load_dataset("ai2_arc", config, split="train")
        for ex in ds:
            choices_text = ex.get("choices", {}).get("text", [])
            choices_label = ex.get("choices", {}).get("label", [])
            key = str(ex.get("answerKey", ""))
            answer = ""
            if key in choices_label:
                answer = choices_text[choices_label.index(key)]
            yield {
                "question": ex["question"],
                "answer": answer,
                "source_dataset": f"arc_{config.lower()}",
            }


def iter_openbookqa_examples() -> Iterable[dict]:
    ds = load_dataset("openbookqa", "main", split="train")
    for ex in ds:
        choices_text = ex.get("choices", {}).get("text", [])
        choices_label = ex.get("choices", {}).get("label", [])
        key = str(ex.get("answerKey", ""))
        answer = ""
        if key in choices_label:
            answer = choices_text[choices_label.index(key)]
        yield {
            "question": ex["question_stem"],
            "answer": answer,
            "source_dataset": "openbookqa_train",
        }


def main(args):
    prox_test_questions = extract_mmlu_prox_en_test_questions()
    print(f"MMLU-ProX EN test questions: {len(prox_test_questions)}")

    candidates = []
    for fn in (iter_mmlu_aux_examples, iter_arc_examples, iter_openbookqa_examples):
        for ex in fn():
            q = ex["question"]
            if q and ex["answer"]:
                ex["question_norm"] = normalize_question(q)
                candidates.append(ex)
    print(f"Raw candidate examples: {len(candidates)}")

    kept = []
    leaked = []
    seen = set()
    for ex in candidates:
        qn = ex["question_norm"]
        if qn in prox_test_questions:
            leaked.append(ex)
            continue
        if qn in seen:
            continue
        seen.add(qn)
        kept.append(
            {
                "id": stable_id(ex["question"]),
                "query": ex["question"].strip(),
                "answer": ex["answer"].strip(),
                "source_language": "English",
                "target_language": "English",
                "source_dataset": ex["source_dataset"],
            }
        )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    report = {
        "mmlu_prox_en_test_count": len(prox_test_questions),
        "raw_candidate_count": len(candidates),
        "removed_due_to_overlap": len(leaked),
        "final_task_specialization_count": len(kept),
    }
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved task-specialization data to: {args.output_path}")
    print(f"Saved leakage report to: {args.report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--report-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
