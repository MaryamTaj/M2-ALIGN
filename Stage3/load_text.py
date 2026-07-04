"""Load Stage 3a training data from HuggingFace Hub, mirroring MindMerger.

Tasks and sources
-----------------
mgsm / msvamp  -> 30,000 samples from meta-math/MetaMathQA, queries translated
                  to the target language with NLLB-200-3.3B, English answers kept.
                  (Mathoctopus_cross_train from Chen et al. 2023 is no longer on
                  the Hub; MetaMathQA is the same class of EN->target cross-lingual
                  math data at the 30K-per-language scale MindMerger used.)

xnli and xcsqa were dropped for now: standard XNLI's language list is
ar/bg/de/el/en/es/fr/hi/ru/sw/th/tr/ur/vi/zh -- "bg" is Bulgarian, not
Bengali, and there is no Bengali config at all. INK-USC/xcsr (X-CSQA) has
the same gap. Rather than lean on a substitute dataset for one language
only, both tasks are removed from Stage 3a entirely (Swahili included)
until a clean native/dedicated source exists for both languages.

All tasks output JSONL with fields:
    query            - target-language input text
    answer           - expected response (English chain-of-thought)
    source_language  - e.g. "Swahili", "Bengali" (for NLLB tokenization in
                       Stage3/train_text.py)
    source_dataset   - dataset identifier string

Supports multiple languages per task: run once per --language, and the
output filename includes the language so `Stage3/train_text.py` (which
globs every *.jsonl in --data-dir) trains jointly on all of them without
one overwriting another.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

TASK_DEFAULTS: dict[str, int] = {
    "mgsm":   30_000,
    "msvamp": 30_000,
}

# Languages with a translation-based training recipe.
LANG_TO_NLLB: dict[str, str] = {
    "Swahili": "swh_Latn",
    "Bengali": "ben_Beng",
}

_NLLB_EN = "eng_Latn"


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"load_data_{timestamp}.log")
    logger = logging.getLogger("stage3a_load_data")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def write_jsonl(path: str, rows: list[dict], logger: logging.Logger) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows → %s", len(rows), path)


# ---------------------------------------------------------------------------
# NLLB translation (used by mgsm/msvamp)
# ---------------------------------------------------------------------------

def load_nllb(nllb_model: str, logger: logging.Logger):
    """Load NLLB-200-3.3B tokenizer and model."""
    logger.info("Loading NLLB tokenizer and model from: %s", nllb_model)
    tokenizer = NllbTokenizer.from_pretrained(nllb_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model, torch_dtype=dtype).to(device)
    model.eval()
    logger.info("NLLB loaded on %s.", device)
    return tokenizer, model, device


def translate_batch(
    texts: list[str],
    tokenizer: NllbTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    target_nllb_tag: str,
    batch_size: int = 16,
    max_length: int = 256,
    num_beams: int = 4,
) -> list[str]:
    """Translate a list of English strings to *target_nllb_tag* with NLLB."""
    tokenizer.src_lang = _NLLB_EN
    forced_bos = tokenizer.convert_tokens_to_ids(target_nllb_tag)
    results: list[str] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_length,
                num_beams=num_beams,
            )
        results.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
    return results


def _nllb_tag(language: str) -> str:
    if language not in LANG_TO_NLLB:
        raise ValueError(f"Unknown language {language!r}; allowed: {sorted(LANG_TO_NLLB)}")
    return LANG_TO_NLLB[language]


# ---------------------------------------------------------------------------
# Math — MetaMathQA queries translated to the target language, English answers kept
# ---------------------------------------------------------------------------

def load_math(
    n_samples: int,
    seed: int,
    logger: logging.Logger,
    language: str = "Swahili",
    nllb_model: str = "facebook/nllb-200-3.3B",
    batch_size: int = 16,
    num_beams: int = 4,
    **_,
) -> list[dict]:
    """Sample MetaMathQA, translate queries to *language* with NLLB, keep English answers.

    MetaMathQA (395K rows) augments GSM8K and MATH problems with varied
    reformulations. Sampling 30K and translating queries produces the same
    class of cross-lingual math training data MindMerger used from Chen
    et al. (2023), at the same 30K-per-language scale.
    """
    nllb_tag = _nllb_tag(language)
    logger.info("Loading meta-math/MetaMathQA ...")
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    logger.info("MetaMathQA: %d rows total", len(ds))

    rows = [
        {"query": r["query"].strip(), "answer": r["response"].strip()}
        for r in ds
        if r["query"].strip() and r["response"].strip()
    ]

    n = min(n_samples, len(rows))
    if n < n_samples:
        logger.warning("Requested %d but only %d available; using all.", n_samples, n)
    sampled = random.Random(seed).sample(rows, n)

    tokenizer, model, device = load_nllb(nllb_model, logger)
    queries_en = [r["query"] for r in sampled]
    logger.info("Translating %d queries EN->%s (%s) ...", len(queries_en), language, nllb_tag)
    queries_translated = translate_batch(
        queries_en, tokenizer, model, device, nllb_tag,
        batch_size=batch_size, num_beams=num_beams,
    )

    return [
        {
            "id": stable_id(f"math_{language}_{r['query']}"),
            "query": q_translated,
            "answer": r["answer"],
            "source_language": language,
            "source_dataset": "metamathqa_translated",
        }
        for r, q_translated in zip(sampled, queries_translated)
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

LOADERS = {
    "mgsm":   load_math,
    "msvamp": load_math,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Stage 3a training data from HuggingFace (MindMerger protocol)."
    )
    parser.add_argument("--task", required=True, choices=list(LOADERS),
                        help="Task: mgsm or msvamp.")
    parser.add_argument("--language", type=str, default="Swahili", choices=list(LANG_TO_NLLB),
                        help="Source language for this task's training data.")
    parser.add_argument("--output_dir", type=str, default="./data/stage3a",
                        help="Directory to write the output JSONL file.")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples (defaults to MindMerger paper values per task).")
    parser.add_argument("--seed", type=int, default=42)
    # NLLB translation args
    parser.add_argument("--nllb_model", type=str, default="facebook/nllb-200-3.3B",
                        help="HuggingFace ID or local path for NLLB-200-3.3B.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Translation batch size.")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Beam width for NLLB translation.")
    args = parser.parse_args()

    logger = setup_logging(os.path.join(os.path.dirname(__file__), "logs"))

    n_samples = args.n_samples if args.n_samples is not None else TASK_DEFAULTS[args.task]
    logger.info("Task=%s  Language=%s  n_samples=%d", args.task, args.language, n_samples)

    rows = LOADERS[args.task](
        n_samples=n_samples,
        seed=args.seed,
        logger=logger,
        language=args.language,
        nllb_model=args.nllb_model,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.task}_{args.language.lower()}.jsonl")
    write_jsonl(out_path, rows, logger)
    logger.info("Done.")


if __name__ == "__main__":
    main()
