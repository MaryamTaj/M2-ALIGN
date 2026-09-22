"""Build the text-only replay pool for pooled Stage 3b replay training.

Port of Santiago's Approach2/build_math_replay.py (branch
approach2-gemma-siglip, DESIGN.md D6/D13) to the pooled M2-ALIGN pipeline.
Writes two kinds of rows per language into --output-dir:

    translation_{code}.jsonl  N rows sampled from Stage 1's
        {Language}_to_English.jsonl (the mapping stage's own data):
        {"source": <lang text>, "target": <English>, "nllb_lang_tag": ..., "task": "translation"}
    math_{code}.jsonl         N MetaMathQA GSM_* problems, question
        NLLB-translated into the language, English CoT solution kept:
        {"query": <lang question>, "target": <English CoT ... The answer is: N>,
         "nllb_lang_tag": ..., "task": "math"}
    math_en.jsonl             (--include-english-math) the same N problems
        untranslated, tag eng_Latn.

As in his builder, every language gets the SAME seeded sample of MetaMathQA
problems (seed 13), so the math pool is parallel across languages.
MetaMathQA's `response` already ends in "The answer is: N", which
Stage3/evaluate.py's extract_math_answer parses.

Existing output files are skipped unless --overwrite, so a failed job can
be resubmitted. MetaMathQA must already be in the HF datasets cache
(compute nodes are offline); it is at $SCRATCH/huggingface/datasets.

Usage (see Stage3/job-scripts/load_replay_data.sh):
    python Stage3/load_replay_data.py \\
        --langs bn,ru,de,zh,pt,id,ko,jv,mn,si,ga \\
        --stage1-dir $SCRATCH/M2-ALIGN/Stage1/data \\
        --output-dir $SCRATCH/M2-ALIGN/Stage3/data/replay \\
        --mt-path $SCRATCH/huggingface/nllb-200-distilled-600M-full/<snapshot> \\
        --include-english-math --local-files-only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
from collections import Counter

import torch
from tqdm import tqdm

from evaluate import NLLB_CODES

# Stage 1 file names are {Language}_to_English.jsonl.
LANG_NAMES: dict[str, str] = {
    "bn": "Bengali", "ru": "Russian", "de": "German", "zh": "Chinese",
    "pt": "Portuguese", "id": "Indonesian", "ko": "Korean", "jv": "Javanese",
    "mn": "Mongolian", "si": "Sinhalese", "ga": "Irish",
}


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"Wrote {len(rows)} rows -> {path}")


# NLLB (greedy, no repetition penalty) sometimes loops: "kick kick kick ...",
# or the same sentence 3+ times. Such rows are gibberish to train on, and a
# 2k-token question OOMed pooled Stage 3b replay (job 3613937) because the
# math prompt T is not truncated. Checked: flags 0 of 5000 math_en rows.
_RUN = re.compile(r"(\S.{0,29}?)(?:\s*\1){7,}", re.S)   # a unit repeated 8+ times in a row
_SENT = re.compile(r"[^.\u3002?\uff1f!\uff01]+")          # split on . 。 ? ？ ! ！


def degenerate_reason(text: str, len_ratio: float, median_ratio: float) -> str | None:
    """Why a translated question looks like an NLLB repetition loop, or None.

    *len_ratio* is len(text) / len(English source) in characters and
    *median_ratio* its median over the language, so "len" means 2.5x longer
    than this language's translations usually are.
    """
    for m in _RUN.finditer(text):
        if not m.group(1).strip().replace(",", "").replace(".", "").isdigit():  # keep 1,000,000,000
            return "run"
    sents = Counter(x.strip() for x in _SENT.findall(text) if len(x.strip()) >= 10)
    if sents and sents.most_common(1)[0][1] >= 3:
        return "sentence"
    if len_ratio > 2.5 * median_ratio:
        return "length"
    return None


def drop_degenerate(rows: list[dict], english: list[str], code: str) -> list[dict]:
    """Filter translated math rows; *english* is the parallel source question per row."""
    ratios = [len(r["query"]) / max(1, len(q)) for r, q in zip(rows, english)]
    median = statistics.median(ratios)
    reasons = [degenerate_reason(r["query"], x, median) for r, x in zip(rows, ratios)]
    kept = [r for r, why in zip(rows, reasons) if why is None]
    print(f"math_{code}: dropped {len(rows) - len(kept)}/{len(rows)} degenerate translations "
          f"{dict(Counter(w for w in reasons if w))}")
    return kept


def load_metamath(types: tuple[str, ...], n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    rows = [
        {"question": r["query"].strip(), "answer": r["response"].strip()}
        for r in ds
        if str(r.get("type", "")).startswith(types)
        and r.get("query", "").strip() and r.get("response", "").strip()
    ]
    print(f"MetaMathQA[{'|'.join(types)}]: {len(rows)} problems in pool")
    if 0 < n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows


def translate(questions: list[str], nllb_tag: str, model, tokenizer, device,
              batch_size: int, max_len: int) -> list[str]:
    forced_bos = tokenizer.convert_tokens_to_ids(nllb_tag)
    out: list[str] = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(questions), batch_size), desc=f"translate->{nllb_tag}"):
            enc = tokenizer(questions[i: i + batch_size], return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len).to(device)
            gen = model.generate(**enc, forced_bos_token_id=forced_bos,
                                 max_new_tokens=max_len, num_beams=1)
            out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Text replay pool for pooled Stage 3b.")
    parser.add_argument("--langs", required=True, help="Comma-separated codes, e.g. bn,ru,de")
    parser.add_argument("--stage1-dir", required=True, help="Dir with {Language}_to_English.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mt-path", required=True)
    parser.add_argument("--n-translation", type=int, default=5000, help="Stage 1 rows per language.")
    parser.add_argument("--n-math", type=int, default=5000, help="MetaMathQA problems per language.")
    parser.add_argument("--metamath-types", default="GSM_",
                        help="Comma-separated MetaMathQA `type` prefixes to keep. GSM_ only: "
                             "MATH_ rows carry LaTeX that NLLB destroys (Santiago DESIGN.md).")
    parser.add_argument("--include-english-math", action="store_true",
                        help="Also write math_en.jsonl (untranslated, eng_Latn).")
    parser.add_argument("--seed", type=int, default=13, help="Santiago's seed; larger --n is a superset.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-src-len", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refilter", action="store_true",
                        help="Only re-apply the degenerate-translation filter to existing math_{code}.jsonl "
                             "(originals kept in <output-dir>/unfiltered/). Needs math_en.jsonl. No GPU.")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    codes = [c.strip() for c in args.langs.split(",") if c.strip()]
    unknown = [c for c in codes if c not in LANG_NAMES]
    if unknown:
        raise ValueError(f"No Stage 1 language name for: {unknown}")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.refilter:
        english = [r["query"] for r in read_jsonl(os.path.join(args.output_dir, "math_en.jsonl"))]
        os.makedirs(os.path.join(args.output_dir, "unfiltered"), exist_ok=True)
        for code in codes:
            path = os.path.join(args.output_dir, f"math_{code}.jsonl")
            backup = os.path.join(args.output_dir, "unfiltered", f"math_{code}.jsonl")
            if not os.path.isfile(backup):
                os.replace(path, backup)
            rows = read_jsonl(backup)
            if len(rows) != len(english):
                raise ValueError(f"{backup} has {len(rows)} rows, math_en has {len(english)}: not parallel")
            write_jsonl(path, drop_degenerate(rows, english, code))
        return

    def todo(path: str) -> bool:
        if os.path.isfile(path) and not args.overwrite:
            print(f"exists, skipping: {path}")
            return False
        return True

    # Stage 1 translation replay: a seeded subsample of what the mapping trained on.
    for code in codes:
        out = os.path.join(args.output_dir, f"translation_{code}.jsonl")
        if not todo(out):
            continue
        src = os.path.join(args.stage1_dir, f"{LANG_NAMES[code]}_to_English.jsonl")
        rows = read_jsonl(src)
        if 0 < args.n_translation < len(rows):
            rows = random.Random(args.seed).sample(rows, args.n_translation)
        write_jsonl(out, [
            {"source": r["source"], "target": r["target"],
             "nllb_lang_tag": NLLB_CODES[code], "task": "translation"}
            for r in rows
        ])

    # MetaMathQA math replay.
    math_outs = {code: os.path.join(args.output_dir, f"math_{code}.jsonl") for code in codes}
    en_out = os.path.join(args.output_dir, "math_en.jsonl")
    pending = [c for c, p in math_outs.items() if todo(p)]
    need_en = args.include_english_math and todo(en_out)
    if not pending and not need_en:
        return

    types = tuple(t.strip() for t in args.metamath_types.split(",") if t.strip())
    problems = load_metamath(types, args.n_math, args.seed)

    if need_en:
        write_jsonl(en_out, [
            {"query": p["question"], "target": p["answer"], "nllb_lang_tag": "eng_Latn", "task": "math"}
            for p in problems
        ])
    if not pending:
        return

    from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = NllbTokenizer.from_pretrained(args.mt_path, local_files_only=args.local_files_only)
    tokenizer.src_lang = "eng_Latn"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.mt_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        local_files_only=args.local_files_only,
    ).to(device).eval()

    questions = [p["question"] for p in problems]
    for code in pending:
        tag = NLLB_CODES[code]
        translated = translate(questions, tag, model, tokenizer, device, args.batch_size, args.max_src_len)
        rows = [
            {"query": q, "target": p["answer"], "nllb_lang_tag": tag, "task": "math"}
            for p, q in zip(problems, translated)
        ]
        write_jsonl(math_outs[code], drop_degenerate(rows, questions, code))


if __name__ == "__main__":
    main()
