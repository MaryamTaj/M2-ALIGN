"""Load Stage 2 CC3M training data: English CC3M captions -> NLLB-3.3B translation.

Mirrors Stage 3b's own GQA -> NLLB pipeline (`Stage3/load_translated_data.py`):
`load_base_data.py` samples a single English (image_url, caption) set once and
downloads the images into a shared cache; this script translates the *same*
underlying captions into each target language, so the translated files stay
comparable across languages and only the images already cached by
`load_base_data.py` are read here, never fetched again.

Output rows match `wit_pairs.jsonl`'s schema (see `load_base_data.py`) so the
two are drop-in interchangeable as Stage 2 training data:
    id, image_url, caption_text, target_caption,
    source_language, language_code, nllb_lang_tag

caption_text is the translated caption; target_caption is the original
English CC3M caption for the same image.

Usage
-----
    python load_translated_data.py --languages bn \\
        --cc3m_jsonl $SCRATCH/M2-ALIGN/Stage2/data/cc3m/english.jsonl \\
        --output_dir $SCRATCH/M2-ALIGN/Stage2/data
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

from load_base_data import WIT_CODE_TO_NAME, WIT_TO_NLLB

# Data/outputs/logs live on $SCRATCH, not in the git checkout.
SCRATCH_ROOT = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN", "Stage2")


def setup_logging() -> logging.Logger:
    """Create a logger that writes to stdout.

    The job script's SLURM ``--output`` file is the single log file for a
    run, so this only needs to format stdout consistently -- it must not
    also write its own file, or every run ends up with two logs.
    """
    logger = logging.getLogger("stage2_load_translated_data")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


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


def build_language_rows(
    code: str,
    nllb_tag: str,
    cc3m_rows: list[dict],
    nllb_model: str,
    batch_size: int,
    num_beams: int,
    logger: logging.Logger,
) -> list[dict]:
    """Translate CC3M captions into *code* with NLLB; keep the English caption too.

    Args:
        code: ISO 639-1 language code (key of :data:`WIT_TO_NLLB`).
        nllb_tag: NLLB FLORES-200 tag for *code*.
        cc3m_rows: Output of `load_base_data.sample_cc3m` (id, image_url, caption).
        nllb_model: HF id or local path for NLLB-200-3.3B.
        batch_size: Translation batch size.
        num_beams: Beam width for NLLB translation.
        logger: Logger instance.

    Returns:
        List of JSONL-ready row dicts matching wit_pairs.jsonl's schema.
    """
    tokenizer, model, device = load_nllb(nllb_model, logger)
    tokenizer.src_lang = "eng_Latn"
    forced_bos = tokenizer.convert_tokens_to_ids(nllb_tag)

    captions_en = [r["caption"] for r in cc3m_rows]
    logger.info("Translating %d CC3M captions EN->%s ...", len(captions_en), WIT_CODE_TO_NAME[code])

    translated: list[str] = []
    for start in range(0, len(captions_en), batch_size):
        chunk = captions_en[start:start + batch_size]
        enc = tokenizer(
            chunk, return_tensors="pt", truncation=True, max_length=256, padding=True,
        ).to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **enc, forced_bos_token_id=forced_bos, max_new_tokens=256, num_beams=num_beams,
            )
        translated.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))

    rows: list[dict] = []
    for r, caption_translated in zip(cc3m_rows, translated):
        rows.append({
            "id": r["id"],
            "image_url": r["image_url"],
            "caption_text": caption_translated,
            "target_caption": r["caption"],
            "source_language": WIT_CODE_TO_NAME[code],
            "language_code": code,
            "nllb_lang_tag": nllb_tag,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate Stage 2's CC3M English sample into target languages (NLLB)."
    )
    parser.add_argument(
        "--languages", type=str, default="bn",
        help="Comma-separated ISO 639-1 language codes (must be in WIT_TO_NLLB).",
    )
    parser.add_argument(
        "--cc3m_jsonl", type=str, default=None,
        help="Path to english.jsonl produced by load_base_data.py "
             "(default: <output_dir>/cc3m/english.jsonl).",
    )
    parser.add_argument("--output_dir", type=str, default=os.path.join(SCRATCH_ROOT, "data"))
    parser.add_argument("--nllb_model", type=str, default="facebook/nllb-200-3.3B")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=4)
    args = parser.parse_args()

    logger = setup_logging()

    codes = [c.strip() for c in args.languages.split(",") if c.strip()]
    for code in codes:
        if code not in WIT_TO_NLLB:
            raise ValueError(f"Unknown language code {code!r}; allowed: {sorted(WIT_TO_NLLB)}")

    cc3m_jsonl = args.cc3m_jsonl or os.path.join(args.output_dir, "cc3m", "english.jsonl")
    logger.info("Loading CC3M English rows from %s", cc3m_jsonl)
    with open(cc3m_jsonl, encoding="utf-8") as f:
        cc3m_rows = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d CC3M rows", len(cc3m_rows))

    for code in codes:
        nllb_tag = WIT_TO_NLLB[code]
        rows = build_language_rows(
            code, nllb_tag, cc3m_rows,
            args.nllb_model, args.batch_size, args.num_beams, logger,
        )
        out_path = os.path.join(args.output_dir, code, "cc3m_pairs.jsonl")
        write_jsonl(out_path, rows, logger)

    logger.info("Done.")


if __name__ == "__main__":
    main()
