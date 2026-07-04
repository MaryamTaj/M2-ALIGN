"""Prepare WIT image-captioning pairs for Stage 2 visual grounding training.

Strategy
--------
For each target language we load that language's WIT subset and the English
subset, join on ``image_url`` (a Wikimedia Commons URL that is
language-agnostic), and emit paired rows:

    caption_text   = ref_desc + " " + attr_desc   (WIT paper: best field combo)
    target_caption = English ref_desc for the same image

Output JSONL fields per row:
    id, image_url, caption_text, target_caption,
    source_language, language_code, nllb_lang_tag

Usage
-----
    python load_wit_data.py \\
        --languages fr,de,zh,ar,hi,sw \\
        --n-per-language 50000 \\
        --output-dir ./data/wit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
from datetime import datetime

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Language mappings
# ---------------------------------------------------------------------------

# Wikipedia / WIT ISO 639-1 code → NLLB-200 language tag
WIT_TO_NLLB: dict[str, str] = {
    "ar": "arb_Arab",
    "bg": "bul_Cyrl",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sw": "swh_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
    # Stage 3 VQA-track language (see Stage3 plan). Run
    # `--stats-only --languages bn` before committing to a full download.
    # Indonesian/Javanese were scoped out for now -- add back the same way
    # if/when the VQA track is widened again.
    "bn": "ben_Beng",
}

WIT_CODE_TO_NAME: dict[str, str] = {
    "ar": "Arabic",    "bg": "Bulgarian", "de": "German",
    "el": "Greek",     "es": "Spanish",   "fr": "French",
    "hi": "Hindi",     "hr": "Croatian",  "hu": "Hungarian",
    "it": "Italian",   "ja": "Japanese",  "ko": "Korean",
    "nl": "Dutch",     "pl": "Polish",    "pt": "Portuguese",
    "ro": "Romanian",  "ru": "Russian",   "sw": "Swahili",
    "tr": "Turkish",   "uk": "Ukrainian", "vi": "Vietnamese",
    "zh": "Chinese",
    "bn": "Bengali",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("wit_load")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"load_wit_{ts}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_caption_text(ref: str | None, attr: str | None) -> str | None:
    """Concatenate reference + attribution with period separation.

    Per WIT paper: ref + attr gives the strongest image-text alignment.
    We use period-space rather than newline or bare space so that NLLB
    sees two complete sentences — matching the sentence-level parallel
    text it was trained on — rather than one run-on fragment.
    Labeled prefixes ("Reference:", "Attribution:") are intentionally
    avoided: they inject English tokens into otherwise non-English text
    and confuse NLLB's language detection.
    Falls back to whichever field is non-null.
    """
    parts = []
    if ref and ref.strip():
        r = ref.strip()
        if r[-1] not in ".!?":
            r += "."
        parts.append(r)
    if attr and attr.strip():
        a = attr.strip()
        if a[-1] not in ".!?":
            a += "."
        parts.append(a)
    return " ".join(parts) if parts else None


def _load_wit_subset(lang_code: str, logger: logging.Logger):
    """Load the WIT subset for *lang_code*, trying config-based access first."""
    try:
        ds = load_dataset("google/WIT", lang_code, split="train", streaming=True,
                          trust_remote_code=True)
        logger.info("Loaded WIT with config=%r", lang_code)
        return ds
    except Exception:
        pass
    # Fall back: full dataset filtered by language field
    logger.info("Config-based load failed; streaming full dataset and filtering for lang=%r", lang_code)
    ds = load_dataset("google/WIT", split="train", streaming=True, trust_remote_code=True)
    return ds.filter(lambda x: x.get("language") == lang_code)


def _write_jsonl(path: str, rows: list[dict], logger: logging.Logger) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows → %s", len(rows), path)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_english_index(
    index_size: int,
    logger: logging.Logger,
) -> dict[str, str]:
    """Stream the English WIT subset and build image_url → caption index.

    Args:
        index_size: Maximum number of English entries to index.
        logger: Logger instance.

    Returns:
        Dict mapping ``image_url`` to the English ``caption_reference_description``.
    """
    logger.info("Streaming English WIT to build image_url index (max %d)...", index_size)
    ds_en = _load_wit_subset("en", logger)

    index: dict[str, str] = {}
    for row in ds_en:
        url = row.get("image_url")
        caption = row.get("caption_reference_description")
        if url and caption and caption.strip():
            index[url] = caption.strip()
        if len(index) >= index_size:
            break

    logger.info("English index: %d entries", len(index))
    return index


def load_language_pairs(
    lang_code: str,
    english_index: dict[str, str],
    n_samples: int,
    seed: int,
    logger: logging.Logger,
) -> list[dict]:
    """Stream WIT for *lang_code*, join with English index, return paired rows.

    Args:
        lang_code: ISO 639-1 code (must be a key in :data:`WIT_TO_NLLB`).
        english_index: Dict from :func:`build_english_index`.
        n_samples: Target number of output pairs for this language.
        seed: Random seed for sampling.
        logger: Logger instance.

    Returns:
        List of paired dicts ready for JSONL serialisation.
    """
    lang_name = WIT_CODE_TO_NAME[lang_code]
    nllb_tag = WIT_TO_NLLB[lang_code]
    logger.info("Processing language: %s (%s)", lang_name, lang_code)

    ds = _load_wit_subset(lang_code, logger)

    pairs: list[dict] = []
    seen_urls: set[str] = set()
    for row in ds:
        url = row.get("image_url")
        if not url or url not in english_index or url in seen_urls:
            continue

        caption_text = _build_caption_text(
            row.get("caption_reference_description"),
            row.get("caption_attribution_description"),
        )
        if not caption_text:
            continue

        seen_urls.add(url)
        pairs.append({
            "id": hashlib.sha1(f"{lang_code}_{url}".encode()).hexdigest(),
            "image_url": url,
            "caption_text": caption_text,
            "target_caption": english_index[url],
            "source_language": lang_name,
            "language_code": lang_code,
            "nllb_lang_tag": nllb_tag,
        })

        # Over-collect then sample to get a random subset, not just the first N.
        if len(pairs) >= n_samples * 3:
            break

    n = min(n_samples, len(pairs))
    sampled = random.Random(seed).sample(pairs, n)
    logger.info("  %s: %d pairs found → sampled %d", lang_name, len(pairs), n)
    return sampled


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def print_coverage_table(
    lang_codes: list[str],
    english_index_size: int,
    max_rows_per_lang: int,
    logger: logging.Logger,
) -> None:
    """Stream WIT and print a per-language coverage table.

    For each language prints:
        - total unique images with a valid caption
        - how many of those also have an English caption in the indexed subset
        - coverage percentage

    Args:
        lang_codes: ISO 639-1 codes to survey (must be keys of :data:`WIT_TO_NLLB`).
        english_index_size: How many English WIT rows to index (caps memory use).
            The true overlap may be higher than reported if the English subset is
            larger than this value.
        max_rows_per_lang: Stop streaming each language after this many rows.
            Rows beyond the cap are not counted; a "+" marker flags capped languages.
        logger: Logger instance.
    """
    logger.info("Building English URL index (max %d)...", english_index_size)
    ds_en = _load_wit_subset("en", logger)
    en_urls: set[str] = set()
    for row in ds_en:
        url = row.get("image_url")
        if url:
            en_urls.add(url)
        if len(en_urls) >= english_index_size:
            break
    logger.info("English URL index: %d entries", len(en_urls))

    W = [16, 6, 16, 22, 10]
    header = (
        f"{'Language':<{W[0]}} {'Code':<{W[1]}} {'Images':<{W[2]}}"
        f" {'With EN caption':<{W[3]}} {'Coverage':<{W[4]}}"
    )
    sep = "-" * (sum(W) + len(W) - 1)
    print(f"\nEnglish URL index size: {len(en_urls):,}  "
          f"(rows-per-lang cap: {max_rows_per_lang:,})\n")
    print(header)
    print(sep)

    for code in lang_codes:
        lang_name = WIT_CODE_TO_NAME.get(code, code)
        ds = _load_wit_subset(code, logger)
        total = 0
        paired = 0
        seen: set[str] = set()
        scanned = 0
        for row in ds:
            scanned += 1
            if scanned > max_rows_per_lang:
                break
            url = row.get("image_url")
            if not url or url in seen:
                continue
            caption = _build_caption_text(
                row.get("caption_reference_description"),
                row.get("caption_attribution_description"),
            )
            if not caption:
                continue
            seen.add(url)
            total += 1
            if url in en_urls:
                paired += 1

        capped = scanned > max_rows_per_lang
        pct = 100.0 * paired / total if total else 0.0
        flag = "+" if capped else " "
        print(
            f"{lang_name:<{W[0]}} {code:<{W[1]}} {total:>{W[2]-1},}{flag}"
            f" {paired:>{W[3]-1},}  {pct:>{W[4]-2}.1f}%"
        )

    print(f"\n'+' = row cap hit; true image count and coverage may be higher.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WIT Stage 2 training pairs.")
    parser.add_argument(
        "--languages", type=str, default="fr,de,zh,ar,hi,sw",
        help="Comma-separated ISO 639-1 language codes (must be in WIT_TO_NLLB).",
    )
    parser.add_argument(
        "--n-per-language", type=int, default=50_000,
        help="Target number of image-caption pairs per language.",
    )
    parser.add_argument(
        "--english-index-size", type=int, default=500_000,
        help="How many English WIT entries to index for join coverage.",
    )
    parser.add_argument("--output-dir", type=str, default="./data/wit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stats-only", action="store_true",
        help=(
            "Print a per-language coverage table and exit without writing any data. "
            "Use --max-rows-per-lang to cap how far each language is streamed."
        ),
    )
    parser.add_argument(
        "--max-rows-per-lang", type=int, default=200_000,
        help="Row cap per language when --stats-only is set (default 200 000).",
    )
    args = parser.parse_args()

    logger = setup_logging(os.path.join(os.path.dirname(__file__), "logs"))

    lang_codes = [c.strip() for c in args.languages.split(",") if c.strip()]
    for code in lang_codes:
        if code not in WIT_TO_NLLB:
            raise ValueError(
                f"Language code {code!r} not in WIT_TO_NLLB. "
                f"Supported: {sorted(WIT_TO_NLLB)}"
            )

    if args.stats_only:
        print_coverage_table(lang_codes, args.english_index_size, args.max_rows_per_lang, logger)
        return

    english_index = build_english_index(args.english_index_size, logger)

    all_rows: list[dict] = []
    for code in lang_codes:
        rows = load_language_pairs(code, english_index, args.n_per_language, args.seed, logger)
        all_rows.extend(rows)

    random.Random(args.seed).shuffle(all_rows)
    out_path = os.path.join(args.output_dir, "wit_pairs.jsonl")
    _write_jsonl(out_path, all_rows, logger)

    by_lang: dict[str, int] = {}
    for row in all_rows:
        by_lang[row["language_code"]] = by_lang.get(row["language_code"], 0) + 1
    logger.info("Total pairs: %d | per-language: %s", len(all_rows), by_lang)


if __name__ == "__main__":
    main()
