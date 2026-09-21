"""Build translate-test eval data: native-language questions -> English via NLLB-3.3B.

Translate-test baseline for raw Qwen3-VL: the eval question is machine-
translated into English up front, then `Baseline/evaluate.py` runs on the
translated file unchanged -- same prompt, image, scoring, and answer
targets, only the question text differs. The translator is NLLB-200-3.3B,
the same model `Stage3/load_translated_data.py` used to translate the
Stage 3b training questions (same fp16 / beam / length settings).

Input rows are the native-language eval files from
`Stage3/load_evaluation_data.py` (`$SCRATCH/M2-ALIGN/Stage3/data/{xgqa,cvqa}/
$LANG.jsonl`). Each output row is the input row with:
    query           -- replaced by the English translation
    query_native    -- the original native-language question
    translation_model -- the --nllb-model path/id used
Every other field (id, answer, choices, answer_index, vg_image_id, ...) is
kept as-is, so ids line up with the native and English (`xgqa/en.jsonl`)
runs for per-example comparison. CVQA `choices` are NOT translated: they
are already the English "Translated Options", and keeping them identical
across native / translate-test / upper-bound runs keeps the comparison fair.

A benchmark with no file for the language (e.g. xGQA has no mn/si/ga/jv,
CVQA has no de) is skipped with a warning, so one per-language job can
always request both benchmarks.

Usage
-----
    python Baseline/load_translated_data.py --lang bn \\
        --benchmarks xgqa,cvqa \\
        --data-dir   $SCRATCH/M2-ALIGN/Stage3/data \\
        --output-dir $SCRATCH/M2-ALIGN/Baseline/data/translate_test \\
        --nllb-model /path/to/nllb-200-3.3B
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

# Data/outputs/logs live on $SCRATCH, not in the git checkout.
SCRATCH_ROOT = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN")

# Must match Stage3/evaluate.py's NLLB_CODES (not imported: that module
# pulls in the full M2-ALIGN model).
NLLB_CODES: dict[str, str] = {
    "bn": "ben_Beng",
    "id": "ind_Latn",
    "jv": "jav_Latn",
    "ru": "rus_Cyrl",
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "pt": "por_Latn",
    "ko": "kor_Hang",
    "mn": "khk_Cyrl",  # NLLB-200 only ships Halh Mongolian, Cyrillic script
    "si": "sin_Sinh",
    "ga": "gle_Latn",
}

SUPPORTED_BENCHMARKS = ("xgqa", "cvqa")


def setup_logging() -> logging.Logger:
    """Create a logger that writes to stdout.

    The job script's SLURM ``--output`` file is the single log file for a
    run, so this only needs to format stdout consistently -- it must not
    also write its own file, or every run ends up with two logs.
    """
    logger = logging.getLogger("baseline_load_translated_data")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict], logger: logging.Logger) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows -> %s", len(rows), path)


def load_nllb(nllb_model: str, logger: logging.Logger):
    """Load NLLB-200-3.3B tokenizer and model (same dtype as Stage3/load_translated_data.py)."""
    logger.info("Loading NLLB tokenizer and model from: %s", nllb_model)
    tokenizer = NllbTokenizer.from_pretrained(nllb_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model, torch_dtype=dtype).to(device)
    model.eval()
    logger.info("NLLB loaded on %s.", device)
    return tokenizer, model, device


def translate_to_english(
    texts: list[str],
    src_tag: str,
    tokenizer: NllbTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    batch_size: int,
    num_beams: int,
    logger: logging.Logger,
) -> dict[str, str]:
    """Translate *texts* from *src_tag* into English.

    Unique strings only (xGQA repeats many questions across images), sorted
    by length so each batch pads to a similar size. Beam search without
    sampling, so the output is deterministic.

    Returns:
        Mapping from each unique source string to its English translation.
    """
    tokenizer.src_lang = src_tag
    forced_bos = tokenizer.convert_tokens_to_ids("eng_Latn")

    unique = sorted(set(texts), key=len)
    logger.info("Translating %d unique questions (%d total) %s->eng_Latn ...", len(unique), len(texts), src_tag)

    out: dict[str, str] = {}
    n_batches = (len(unique) + batch_size - 1) // batch_size
    for b, start in enumerate(range(0, len(unique), batch_size)):
        chunk = unique[start:start + batch_size]
        enc = tokenizer(
            chunk, return_tensors="pt", truncation=True, max_length=256, padding=True,
        ).to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **enc, forced_bos_token_id=forced_bos, max_new_tokens=256, num_beams=num_beams,
            )
        out.update(zip(chunk, tokenizer.batch_decode(gen_ids, skip_special_tokens=True)))
        if (b + 1) % 50 == 0 or b + 1 == n_batches:
            logger.info("  batch %d/%d", b + 1, n_batches)
    return out


def build_translated_rows(rows: list[dict], translations: dict[str, str], model_name: str) -> list[dict]:
    """Swap each row's ``query`` for its English translation, keeping the original."""
    translated_rows = []
    for r in rows:
        new = dict(r)
        new["query_native"] = r["query"]
        new["query"] = translations[r["query"]]
        new["translation_model"] = model_name
        translated_rows.append(new)
    return translated_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate native-language xGQA/CVQA eval questions into English (NLLB) for translate-test."
    )
    parser.add_argument("--lang", required=True,
                        help=f"Comma-separated ISO 639-1 codes; allowed: {', '.join(sorted(NLLB_CODES))}.")
    parser.add_argument("--benchmarks", default="xgqa,cvqa",
                        help=f"Comma-separated subset of: {', '.join(SUPPORTED_BENCHMARKS)}.")
    parser.add_argument("--data-dir", default=os.path.join(SCRATCH_ROOT, "Stage3", "data"),
                        help="Root holding <benchmark>/<lang>.jsonl from Stage3/load_evaluation_data.py.")
    parser.add_argument("--output-dir", default=os.path.join(SCRATCH_ROOT, "Baseline", "data", "translate_test"),
                        help="Writes <output-dir>/<benchmark>/<lang>.jsonl.")
    parser.add_argument("--nllb-model", default="facebook/nllb-200-3.3B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=4)
    args = parser.parse_args()

    logger = setup_logging()

    langs = [c.strip() for c in args.lang.split(",") if c.strip()]
    for lang in langs:
        if lang not in NLLB_CODES:
            raise ValueError(f"Unknown language {lang!r}; allowed: {sorted(NLLB_CODES)}")
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    for bench in benchmarks:
        if bench not in SUPPORTED_BENCHMARKS:
            raise ValueError(f"Unknown benchmark {bench!r}; allowed: {list(SUPPORTED_BENCHMARKS)}")

    # (benchmark, lang, input path) for every file that exists.
    jobs = []
    for lang in langs:
        for bench in benchmarks:
            in_path = os.path.join(args.data_dir, bench, f"{lang}.jsonl")
            if os.path.exists(in_path):
                jobs.append((bench, lang, in_path))
            else:
                logger.warning("No %s eval data for lang=%s (%s) -- skipping.", bench, lang, in_path)
    if not jobs:
        raise FileNotFoundError(f"No eval files found under {args.data_dir} for langs={langs} benchmarks={benchmarks}")

    tokenizer, model, device = load_nllb(args.nllb_model, logger)

    for bench, lang, in_path in jobs:
        rows = read_jsonl(in_path)
        logger.info("=== %s/%s: %d rows from %s ===", bench, lang, len(rows), in_path)
        translations = translate_to_english(
            [r["query"] for r in rows], NLLB_CODES[lang],
            tokenizer, model, device, args.batch_size, args.num_beams, logger,
        )
        out_rows = build_translated_rows(rows, translations, args.nllb_model)
        for r in out_rows[:3]:
            logger.info("  sample: %r -> %r", r["query_native"], r["query"])
        write_jsonl(os.path.join(args.output_dir, bench, f"{lang}.jsonl"), out_rows, logger)

    logger.info("Done.")


if __name__ == "__main__":
    main()
