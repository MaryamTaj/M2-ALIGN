"""Build the replay data that Stage 3 mixes in, so the mapping keeps its earlier skills.

Two kinds of rows:
- math: N_MATH grade-school math problems from MetaMathQA (its GSM_ types),
  the same ones for every language. NLLB-3.3B translates the questions;
  the English step-by-step solutions stay as they are. Translations that
  fell into a repetition loop are dropped.
- translation: N_TRANSLATION rows sampled from each language's Stage 1 file.

It runs in two steps:

    python data/build_replay.py download     # workstation: needs internet
    # copy data/ to Narval (Globus), then submit as a GPU job:
    sbatch --gres=gpu:1 --mem=48G --time=12:00:00 job.sh python data/build_replay.py translate

download writes data/replay/math_en.jsonl. translate writes
data/replay/math_{lang}.jsonl and data/replay/translation_{lang}.jsonl:
    {"task": "math", "lang": "bn", "source": "<question>", "target": "<solution> The answer is: 4"}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
from collections import Counter

ROOT = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN")
DATA_DIR = os.path.join(ROOT, "data")
HF_CACHE = os.path.join(os.environ.get("SCRATCH", "."), "huggingface", "hub")
SEED = 13  # the seed of the replay design this follows
TRAIN_LANGS = ["bn", "ru", "de", "zh", "pt", "id", "ko", "jv", "mn", "si", "ga"]
NLLB_TAGS = {
    "en": "eng_Latn", "bn": "ben_Beng", "ru": "rus_Cyrl", "de": "deu_Latn", "zh": "zho_Hans",
    "pt": "por_Latn", "id": "ind_Latn", "ko": "kor_Hang", "jv": "jav_Latn", "mn": "khk_Cyrl",
    "si": "sin_Sinh", "ga": "gle_Latn",
}

N_MATH = 5000
N_TRANSLATION = 5000
# MATH_ problems are full of LaTeX, which NLLB mangles, so only GSM_ problems are used.
METAMATH_TYPES = ("GSM_",)

TRANSLATOR = "facebook/nllb-200-3.3B"
TRANSLATE_BATCH_SIZE = 32
MAX_TRANSLATE_LEN = 512  # math questions are longer than captions

# NLLB sometimes loops ("kick kick kick ...", or one sentence 3+ times).
REPEATED_RUN = re.compile(r"(\S.{0,29}?)(?:\s*\1){7,}", re.S)  # one unit repeated 8+ times in a row
SENTENCE = re.compile(r"[^.。?？!！]+")  # split on . 。 ? ？ ! ！
MAX_LENGTH_RATIO = 2.5  # times the language's median translation-to-English length ratio


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows):>7} rows -> {path}", flush=True)


def load_metamath() -> list[dict]:
    """N_MATH MetaMathQA problems of the kept types, sampled with SEED."""
    from datasets import load_dataset

    ds = load_dataset("meta-math/MetaMathQA", split="train")
    pool = [
        {"question": r["query"].strip(), "solution": r["response"].strip()}
        for r in ds
        if str(r.get("type", "")).startswith(METAMATH_TYPES) and r["query"].strip() and r["response"].strip()
    ]
    return random.Random(SEED).sample(pool, N_MATH) if N_MATH < len(pool) else pool


def degenerate_reason(text: str, length_ratio: float, median_ratio: float) -> str | None:
    """Why a translated question looks like an NLLB repetition loop, or None if it looks fine."""
    for m in REPEATED_RUN.finditer(text):
        if not m.group(1).strip().replace(",", "").replace(".", "").isdigit():  # keep numbers like 1,000,000
            return "run"
    sentences = Counter(s.strip() for s in SENTENCE.findall(text) if len(s.strip()) >= 10)
    if sentences and sentences.most_common(1)[0][1] >= 3:
        return "sentence"
    if length_ratio > MAX_LENGTH_RATIO * median_ratio:
        return "length"
    return None


def drop_degenerate(rows: list[dict], english: list[str], lang: str) -> list[dict]:
    """Drop translated rows that look like repetition loops. english holds each row's source question."""
    ratios = [len(r["source"]) / max(1, len(q)) for r, q in zip(rows, english)]
    median = statistics.median(ratios)
    reasons = [degenerate_reason(r["source"], ratio, median) for r, ratio in zip(rows, ratios)]
    kept = [r for r, reason in zip(rows, reasons) if reason is None]
    print(f"  math_{lang}: dropped {len(rows) - len(kept)} repetition loops {dict(Counter(filter(None, reasons)))}")
    return kept


def load_translator():
    import torch
    from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

    tokenizer = NllbTokenizer.from_pretrained(TRANSLATOR, cache_dir=HF_CACHE, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR, cache_dir=HF_CACHE, dtype=torch.float16)
    return tokenizer, model.to("cuda").eval()


def translate(texts: list[str], tgt_tag: str, translator) -> list[str]:
    """Translate English texts into tgt_tag, in order. Greedy, as in the replay design this follows."""
    import torch

    tokenizer, model = translator
    out = []
    with torch.inference_mode():
        for start in range(0, len(texts), TRANSLATE_BATCH_SIZE):
            enc = tokenizer(texts[start:start + TRANSLATE_BATCH_SIZE], return_tensors="pt", truncation=True,
                            max_length=MAX_TRANSLATE_LEN, padding=True).to("cuda")
            ids = model.generate(**enc, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_tag),
                                 max_new_tokens=MAX_TRANSLATE_LEN, num_beams=1)
            out += tokenizer.batch_decode(ids, skip_special_tokens=True)
    return out


def skip(path: str, overwrite: bool) -> bool:
    if os.path.exists(path) and not overwrite:
        print(f"exists, skipped: {path}")
        return True
    return False


def download(overwrite: bool) -> None:
    path = os.path.join(DATA_DIR, "replay", "math_en.jsonl")
    if not skip(path, overwrite):
        write_jsonl(path, [
            {"task": "math", "lang": "en", "source": p["question"], "target": p["solution"]} for p in load_metamath()
        ])


def translate_step(langs: list[str], overwrite: bool) -> None:
    # Translation replay: a seeded sample of what Stage 1 trained on.
    for lang in langs:
        path = os.path.join(DATA_DIR, "replay", f"translation_{lang}.jsonl")
        if not skip(path, overwrite):
            rows = read_jsonl(os.path.join(DATA_DIR, "stage1", f"{lang}.jsonl"))
            write_jsonl(path, random.Random(SEED).sample(rows, min(N_TRANSLATION, len(rows))))

    english = read_jsonl(os.path.join(DATA_DIR, "replay", "math_en.jsonl"))
    questions = [r["source"] for r in english]
    translator = None
    for lang in langs:
        path = os.path.join(DATA_DIR, "replay", f"math_{lang}.jsonl")
        if skip(path, overwrite):
            continue
        translator = translator or load_translator()
        translated = translate(questions, NLLB_TAGS[lang], translator)
        rows = [{**r, "lang": lang, "source": q} for r, q in zip(english, translated)]
        write_jsonl(path, drop_degenerate(rows, questions, lang))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 3 replay data.")
    parser.add_argument("step", choices=["download", "translate"])
    parser.add_argument("--langs", default=",".join(TRAIN_LANGS), help="translate only.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.step == "download":
        download(args.overwrite)
    else:
        translate_step(args.langs.split(","), args.overwrite)


if __name__ == "__main__":
    main()
