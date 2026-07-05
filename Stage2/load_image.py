"""Prepare WIT image-captioning pairs for Stage 2 visual grounding training.

Strategy
--------
``wikimedia/wit_base`` (the parquet-native mirror of Google's WIT dataset)
dedupes rows by image: each row's ``wit_features`` list already holds the
per-language caption entries for that one image. So the English/target-
language join is free — for each row we just look up the "en" entry and the
target-language entry and, if both exist, emit a pair:

    caption_text   = ref_desc for the target language
    target_caption = English ref_desc for the same image

``caption_attribution_description`` is excluded from caption_text: it's
shared per image (not per language) in wikimedia/wit_base and is often
left in English regardless of which language's ref_desc is used.

Output JSONL fields per row:
    id, image_url, caption_text, target_caption,
    source_language, language_code, nllb_lang_tag

Usage
-----
    python load_image.py \\
        --languages fr,de,zh,ar,hi,sw \\
        --n-per-language 50000 \\
        --output-dir ./data
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

# Unicode block ranges keyed by the script suffix of the NLLB tag (e.g.
# "ben_Beng" -> "Beng"). Only scripts visually distinguishable from Latin
# are listed: for *_Latn languages (de, es, fr, hr, hu, it, nl, pl, pt, ro,
# sw, tr, vi) script alone can't tell target-language text apart from
# English attribution text, so those are left unchecked.
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "Arab": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)],
    "Cyrl": [(0x0400, 0x04FF)],
    "Grek": [(0x0370, 0x03FF)],
    "Deva": [(0x0900, 0x097F)],
    "Beng": [(0x0980, 0x09FF)],
    "Hang": [(0xAC00, 0xD7A3)],
    "Jpan": [(0x3040, 0x30FF), (0x4E00, 0x9FFF)],
    "Hans": [(0x4E00, 0x9FFF)],
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

def _script_for_lang(lang_code: str) -> str | None:
    """NLLB script suffix for *lang_code* (e.g. 'bn' -> 'Beng'), or None if
    it's a Latin-script language whose script can't distinguish target-
    language text from English attribution text."""
    script = WIT_TO_NLLB[lang_code].split("_")[1]
    return script if script in SCRIPT_RANGES else None


def _is_in_script(text: str, script: str, threshold: float = 0.5) -> bool:
    """True if at least *threshold* of *text*'s alphabetic characters fall
    within the Unicode ranges for *script*."""
    ranges = SCRIPT_RANGES[script]
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    in_script = sum(1 for c in alpha if any(lo <= ord(c) <= hi for lo, hi in ranges))
    return (in_script / len(alpha)) >= threshold


def _build_caption_text(ref: str | None) -> str | None:
    """Return the cleaned reference description as caption_text.

    ``caption_attribution_description`` is intentionally excluded: in
    wikimedia/wit_base it's shared per image rather than per language, and
    is frequently left in English even when *ref* is in the target
    language — including it would glue an English sentence onto an
    otherwise non-English caption_text.
    """
    if not ref or not ref.strip():
        return None
    r = ref.strip()
    if r[-1] not in ".!?":
        r += "."
    return r


def _load_wit_base(logger: logging.Logger):
    """Stream the full ``wikimedia/wit_base`` dataset (parquet, no script).

    Replaces ``google/WIT``, whose Hub repo only ships a loading script;
    ``datasets>=4.0`` removed script execution (``trust_remote_code`` is no
    longer honoured), so ``google/WIT`` can no longer be loaded at all.
    """
    logger.info("Streaming wikimedia/wit_base...")
    return load_dataset("wikimedia/wit_base", split="train", streaming=True)


def _extract_lang_entries(row: dict, wanted_langs: set[str]) -> dict[str, dict]:
    """Map each wanted language code to its ``wit_features`` entry, if present.

    A row is one image; ``wit_features`` holds one entry per Wikipedia
    language edition that captioned it. ``datasets`` stores a Sequence of a
    struct feature in columnar form — i.e. ``wit_features`` is a dict
    mapping field name to a list of per-entry values, not a list of dicts —
    so entries are reconstructed by index. Keeps the first match per language.
    """
    feats = row.get("wit_features") or {}
    languages = feats.get("language") or []
    found: dict[str, dict] = {}
    for i, lang in enumerate(languages):
        if lang in wanted_langs and lang not in found:
            found[lang] = {key: values[i] for key, values in feats.items()}
    return found


def _write_jsonl(path: str, rows: list[dict], logger: logging.Logger) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows → %s", len(rows), path)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_language_pairs(
    lang_codes: list[str],
    n_per_language: int,
    seed: int,
    logger: logging.Logger,
) -> dict[str, list[dict]]:
    """Single pass over WIT: emit English-paired rows for every requested language.

    Each row already carries every language's caption for that image, so all
    target languages are collected in one stream instead of one stream per
    language.

    Args:
        lang_codes: ISO 639-1 codes to collect (must be keys of :data:`WIT_TO_NLLB`).
        n_per_language: Target number of output pairs per language.
        seed: Random seed for sampling.
        logger: Logger instance.

    Returns:
        Dict mapping language code to its list of paired dicts.
    """
    wanted = set(lang_codes) | {"en"}
    ds = _load_wit_base(logger)

    collected: dict[str, list[dict]] = {code: [] for code in lang_codes}
    overcollect = n_per_language * 3
    done: set[str] = set()

    for row in ds:
        if len(done) == len(lang_codes):
            break
        url = row.get("image_url")
        if not url:
            continue

        entries = _extract_lang_entries(row, wanted)
        en_entry = entries.get("en")
        if not en_entry:
            continue
        en_caption = (en_entry.get("caption_reference_description") or "").strip()
        if not en_caption:
            continue

        for code in lang_codes:
            if code in done:
                continue
            entry = entries.get(code)
            if not entry:
                continue
            caption_text = _build_caption_text(entry.get("caption_reference_description"))
            if not caption_text:
                continue

            collected[code].append({
                "id": hashlib.sha1(f"{code}_{url}".encode()).hexdigest(),
                "image_url": url,
                "caption_text": caption_text,
                "target_caption": en_caption,
                "source_language": WIT_CODE_TO_NAME[code],
                "language_code": code,
                "nllb_lang_tag": WIT_TO_NLLB[code],
            })
            # Over-collect then sample to get a random subset, not just the first N.
            if len(collected[code]) >= overcollect:
                done.add(code)

    sampled: dict[str, list[dict]] = {}
    for code in lang_codes:
        pairs = collected[code]
        n = min(n_per_language, len(pairs))
        sampled[code] = random.Random(seed).sample(pairs, n)
        logger.info("  %s: %d pairs found → sampled %d", WIT_CODE_TO_NAME[code], len(pairs), n)
    return sampled


def analyze_attribution_script(
    lang_codes: list[str],
    max_rows: int,
    logger: logging.Logger,
) -> None:
    """Single pass over WIT: for each language, count how many images with a
    caption in that language also have a non-empty attribution, and how many
    of those attributions actually appear to be written in the target
    script (vs. left in English/Latin, which :func:`_build_caption_text`
    would then drop). Latin-script languages are reported as "n/a" since
    script alone can't distinguish them from English attribution text.
    """
    wanted = set(lang_codes)
    ds = _load_wit_base(logger)

    with_attr = {code: 0 for code in lang_codes}
    in_script = {code: 0 for code in lang_codes}

    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > max_rows:
            break
        attr = (row.get("caption_attribution_description") or "").strip()
        if not attr:
            continue
        entries = _extract_lang_entries(row, wanted)
        for code in lang_codes:
            entry = entries.get(code)
            if not entry or not (entry.get("caption_reference_description") or "").strip():
                continue
            with_attr[code] += 1
            script = _script_for_lang(code)
            if script is not None and _is_in_script(attr, script):
                in_script[code] += 1

    W = [16, 6, 12, 12, 20]
    header = (
        f"{'Language':<{W[0]}} {'Code':<{W[1]}} {'With attr':<{W[2]}}"
        f" {'In script':<{W[3]}} {'Notes':<{W[4]}}"
    )
    sep = "-" * (sum(W) + len(W) - 1)
    print(f"\nRows scanned: {scanned:,}  (cap: {max_rows:,})\n")
    print(header)
    print(sep)
    for code in lang_codes:
        lang_name = WIT_CODE_TO_NAME.get(code, code)
        script = _script_for_lang(code)
        if script is None:
            print(f"{lang_name:<{W[0]}} {code:<{W[1]}} {with_attr[code]:>{W[2]-1},}"
                  f" {'n/a':<{W[3]}} {'Latin script, not checked':<{W[4]}}")
        else:
            pct = 100.0 * in_script[code] / with_attr[code] if with_attr[code] else 0.0
            print(f"{lang_name:<{W[0]}} {code:<{W[1]}} {with_attr[code]:>{W[2]-1},}"
                  f" {in_script[code]:>{W[3]-1},} {pct:>5.1f}%")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def print_coverage_table(
    lang_codes: list[str],
    max_rows: int,
    logger: logging.Logger,
) -> None:
    """Stream WIT once and print a per-language coverage table.

    For each language prints:
        - total unique images with a valid caption
        - how many of those also have an English caption (same row)
        - coverage percentage

    Args:
        lang_codes: ISO 639-1 codes to survey (must be keys of :data:`WIT_TO_NLLB`).
        max_rows: Stop streaming after this many rows total. Rows beyond the
            cap are not counted; a "+" marker flags capped languages.
        logger: Logger instance.
    """
    wanted = set(lang_codes) | {"en"}
    ds = _load_wit_base(logger)

    total = {code: 0 for code in lang_codes}
    paired = {code: 0 for code in lang_codes}
    seen: dict[str, set[str]] = {code: set() for code in lang_codes}

    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > max_rows:
            break
        url = row.get("image_url")
        if not url:
            continue

        entries = _extract_lang_entries(row, wanted)
        en_entry = entries.get("en")
        has_en = bool(en_entry and (en_entry.get("caption_reference_description") or "").strip())

        for code in lang_codes:
            entry = entries.get(code)
            if not entry or url in seen[code]:
                continue
            caption = _build_caption_text(entry.get("caption_reference_description"))
            if not caption:
                continue
            seen[code].add(url)
            total[code] += 1
            if has_en:
                paired[code] += 1

    capped = scanned > max_rows

    W = [16, 6, 16, 22, 10]
    header = (
        f"{'Language':<{W[0]}} {'Code':<{W[1]}} {'Images':<{W[2]}}"
        f" {'With EN caption':<{W[3]}} {'Coverage':<{W[4]}}"
    )
    sep = "-" * (sum(W) + len(W) - 1)
    print(f"\nRows scanned: {scanned:,}  (cap: {max_rows:,})\n")
    print(header)
    print(sep)

    for code in lang_codes:
        lang_name = WIT_CODE_TO_NAME.get(code, code)
        pct = 100.0 * paired[code] / total[code] if total[code] else 0.0
        flag = "+" if capped else " "
        print(
            f"{lang_name:<{W[0]}} {code:<{W[1]}} {total[code]:>{W[2]-1},}{flag}"
            f" {paired[code]:>{W[3]-1},}  {pct:>{W[4]-2}.1f}%"
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
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stats-only", action="store_true",
        help=(
            "Print a per-language coverage table and exit without writing any data. "
            "Use --max-rows to cap how far the dataset is streamed."
        ),
    )
    parser.add_argument(
        "--attribution-script-stats", action="store_true",
        help=(
            "Print, per language, how many images with a target-language caption "
            "also have their attribution written in that language's script (vs. "
            "left in English) and exit without writing any data."
        ),
    )
    parser.add_argument(
        "--max-rows", type=int, default=200_000,
        help="Total row cap when --stats-only or --attribution-script-stats is set (default 200 000).",
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
        print_coverage_table(lang_codes, args.max_rows, logger)
        return

    if args.attribution_script_stats:
        analyze_attribution_script(lang_codes, args.max_rows, logger)
        return

    pairs_by_lang = build_language_pairs(lang_codes, args.n_per_language, args.seed, logger)

    all_rows: list[dict] = [row for rows in pairs_by_lang.values() for row in rows]
    random.Random(args.seed).shuffle(all_rows)
    out_path = os.path.join(args.output_dir, "wit_pairs.jsonl")
    _write_jsonl(out_path, all_rows, logger)

    by_lang: dict[str, int] = {}
    for row in all_rows:
        by_lang[row["language_code"]] = by_lang.get(row["language_code"], 0) + 1
    logger.info("Total pairs: %d | per-language: %s", len(all_rows), by_lang)


if __name__ == "__main__":
    main()
