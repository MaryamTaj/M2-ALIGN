"""Load Stage 3b VQA training data: GQA (English) -> NLLB-3.3B translation.

Mirrors MindMerger's own augmentation-training recipe (translate an
established English task corpus into each target language, keep the
answer untranslated). `load_nllb`/`stable_id`/`write_jsonl` below were
originally shared with Stage 3a's now-removed `load_text_data.py`; they're
inlined here rather than reimported since that module no longer exists.

Why GQA rather than a from-scratch VQA corpus: `xGQA` (the Bengali
evaluation benchmark used in Stage 3b) is itself a
professionally-translated slice of GQA's own test-dev split. Training on
GQA's English train split, then NLLB-translating the questions, gives
training data that matches the eval benchmark's question style,
compositional structure, and image domain (Visual Genome) as closely as
possible short of training on the eval set itself.

Answers are kept in English (short, mostly single tokens) -- a
language-agnostic, easily-gradable shape. Only the question is
translated; images are resolved lazily by
`Stage3/train.py`, not downloaded here (same design as
`Stage2/load_wit_data.py`, which stores `image_url` rather than bytes).

Output JSONL fields per row:
    id, vg_image_id, query, answer, source_language, nllb_lang_tag,
    source_dataset

Usage
-----
    python load_translated_data.py --languages Bengali \\
        --n_samples 30000 --output_dir $SCRATCH/M2-ALIGN/Stage3/data/stage3b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

# Data/outputs/logs live on $SCRATCH, not in the git checkout.
SCRATCH_ROOT = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN", "Stage3")

# Stage 3b VQA-track language (must already be backfilled into Stage 1's
# NLLB corpus and Stage 2's WIT config -- see the Stage 3 plan).
LANG_TO_NLLB: dict[str, str] = {
    "Bengali": "ben_Beng",
    "Russian": "rus_Cyrl",
    "German": "deu_Latn",
    "Chinese": "zho_Hans",
    "Portuguese": "por_Latn",
    "Indonesian": "ind_Latn",
    "Korean": "kor_Hang",
    "Javanese": "jav_Latn",
    "Mongolian": "khk_Cyrl",  # NLLB-200 only ships Halh Mongolian, Cyrillic script
    "Sinhalese": "sin_Sinh",
    "Irish": "gle_Latn",
}

DEFAULT_N_SAMPLES = 30_000
# Cap on questions drawn from any single image so the ~30K sample isn't
# dominated by a handful of heavily-annotated images (per the Stage 3 plan).
MAX_QUESTIONS_PER_IMAGE = 4

# GQA balanced train split via the lmms-lab packaging (embeds images
# directly; falls back to Visual Genome image-id references if a given
# mirror only exposes ids). Field names are read defensively below since
# GQA mirrors on the Hub vary in schema.
GQA_HF_ID = "lmms-lab/GQA"
GQA_CONFIG = "train_balanced_instructions"
GQA_SPLIT = "train"


def setup_logging() -> logging.Logger:
    """Create a logger that writes to stdout.

    The job script's SLURM ``--output`` file is the single log file for a
    run, so this only needs to format stdout consistently -- it must not
    also write its own file, or every run ends up with two logs.
    """
    logger = logging.getLogger("stage3b_load_translated_data")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
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
    logger.info("Wrote %d rows -> %s", len(rows), path)


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


def _row_image_id(row: dict) -> str | None:
    """Extract a stable Visual Genome image id from a GQA row.

    Different GQA mirrors on the Hub use different field names for the
    same concept; try the common ones in order.
    """
    for key in ("imageId", "image_id", "imageid"):
        val = row.get(key)
        if val is not None:
            return str(val)
    # Some mirrors nest the id under an "image" dict/object.
    image = row.get("image")
    if isinstance(image, dict):
        for key in ("id", "image_id"):
            if image.get(key) is not None:
                return str(image[key])
    return None


def _row_question(row: dict) -> str | None:
    for key in ("question", "question_text", "sent"):
        val = row.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return None


def _row_answer(row: dict) -> str | None:
    for key in ("answer", "multiple_choice_answer", "label"):
        val = row.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return None


def sample_gqa(
    n_samples: int,
    seed: int,
    logger: logging.Logger,
) -> list[dict]:
    """Load GQA balanced train, cap per-image questions, and sample.

    Args:
        n_samples: Target number of (question, answer, image_id) rows.
        seed: Random seed for sampling and shuffling.
        logger: Logger instance.

    Returns:
        List of dicts with keys ``question``, ``answer``, ``vg_image_id``.
    """
    logger.info("Loading %s [%s/%s] ...", GQA_HF_ID, GQA_CONFIG, GQA_SPLIT)
    ds = load_dataset(GQA_HF_ID, GQA_CONFIG, split=GQA_SPLIT)
    logger.info("GQA balanced train: %d rows total", len(ds))

    per_image_count: dict[str, int] = defaultdict(int)
    rows: list[dict] = []
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    for i in indices:
        r = ds[i]
        image_id = _row_image_id(r)
        question = _row_question(r)
        answer = _row_answer(r)
        if not image_id or not question or not answer:
            continue
        if per_image_count[image_id] >= MAX_QUESTIONS_PER_IMAGE:
            continue
        per_image_count[image_id] += 1
        rows.append({"question": question, "answer": answer, "vg_image_id": image_id})
        if len(rows) >= n_samples:
            break

    logger.info(
        "Sampled %d GQA rows over %d unique images (cap=%d/image)",
        len(rows), len(per_image_count), MAX_QUESTIONS_PER_IMAGE,
    )
    return rows


def build_language_rows(
    lang_name: str,
    nllb_tag: str,
    gqa_rows: list[dict],
    nllb_model: str,
    batch_size: int,
    num_beams: int,
    logger: logging.Logger,
) -> list[dict]:
    """Translate GQA questions into *lang_name* with NLLB; keep answers as-is.

    Args:
        lang_name: Human-readable language name (key of :data:`LANG_TO_NLLB`).
        nllb_tag: NLLB FLORES-200 tag for *lang_name*.
        gqa_rows: Output of :func:`sample_gqa`.
        nllb_model: HF id or local path for NLLB-200-3.3B.
        batch_size: Translation batch size.
        num_beams: Beam width for NLLB translation.
        logger: Logger instance.

    Returns:
        List of JSONL-ready row dicts.
    """
    # load_text_data.py's own translate_batch() is hardcoded EN->Swahili (module
    # constants _NLLB_EN/_NLLB_SW); Stage 3b needs an arbitrary target
    # language, so the translation loop is reimplemented here against the
    # same load_nllb()-loaded tokenizer/model rather than parameterizing
    # Stage 3a's already-working helper.
    tokenizer, model, device = load_nllb(nllb_model, logger)
    tokenizer.src_lang = "eng_Latn"
    forced_bos = tokenizer.convert_tokens_to_ids(nllb_tag)

    questions_en = [r["question"] for r in gqa_rows]
    logger.info("Translating %d GQA questions EN->%s ...", len(questions_en), lang_name)

    translated: list[str] = []
    for start in range(0, len(questions_en), batch_size):
        chunk = questions_en[start:start + batch_size]
        enc = tokenizer(
            chunk, return_tensors="pt", truncation=True, max_length=256, padding=True,
        ).to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **enc, forced_bos_token_id=forced_bos, max_new_tokens=256, num_beams=num_beams,
            )
        translated.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))

    rows: list[dict] = []
    for r, q_translated in zip(gqa_rows, translated):
        rows.append({
            "id": stable_id(f"gqa_{lang_name}_{r['vg_image_id']}_{r['question']}"),
            "vg_image_id": r["vg_image_id"],
            "query": q_translated,
            "answer": r["answer"],
            "source_language": lang_name,
            "nllb_lang_tag": nllb_tag,
            "source_dataset": "gqa_balanced_train_translated",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Stage 3b VQA training data (GQA -> NLLB translation)."
    )
    parser.add_argument(
        "--languages", type=str, default="Bengali",
        help="Comma-separated language names (must be keys of LANG_TO_NLLB).",
    )
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES,
                        help="Target GQA questions sampled before translation (shared across languages). "
                             "Ignored if --gqa_jsonl is given.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(SCRATCH_ROOT, "data", "stage3b"))
    parser.add_argument("--seed", type=int, default=42,
                        help="Ignored if --gqa_jsonl is given (the JSONL was already sampled with its own seed).")
    parser.add_argument("--nllb_model", type=str, default="facebook/nllb-200-3.3B")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument(
        "--gqa_jsonl", type=str, default=None,
        help="Path to a JSONL of {question, answer, vg_image_id} rows produced offline by "
             "Stage3/load_base_data.py (run on a login-node-free workstation, then Globus-transferred "
             "here). When given, this script never calls load_dataset() itself -- no network access "
             "needed on Narval for this step at all.",
    )
    args = parser.parse_args()

    logger = setup_logging()

    langs = [x.strip() for x in args.languages.split(",") if x.strip()]
    for lang in langs:
        if lang not in LANG_TO_NLLB:
            raise ValueError(f"Unknown language {lang!r}; allowed: {sorted(LANG_TO_NLLB)}")

    # Sample GQA once; translate the *same* underlying questions into every
    # language so results are comparable across languages.
    if args.gqa_jsonl:
        logger.info("Loading pre-sampled GQA rows from %s (no network access)", args.gqa_jsonl)
        with open(args.gqa_jsonl, encoding="utf-8") as f:
            gqa_rows = [json.loads(line) for line in f if line.strip()]
        logger.info("Loaded %d pre-sampled GQA rows", len(gqa_rows))
    else:
        gqa_rows = sample_gqa(args.n_samples, args.seed, logger)

    os.makedirs(args.output_dir, exist_ok=True)
    for lang in langs:
        nllb_tag = LANG_TO_NLLB[lang]
        rows = build_language_rows(
            lang, nllb_tag, gqa_rows,
            args.nllb_model, args.batch_size, args.num_beams, logger,
        )
        out_path = os.path.join(args.output_dir, f"{lang.lower()}.jsonl")
        write_jsonl(out_path, rows, logger)

    logger.info("Done.")


if __name__ == "__main__":
    main()
