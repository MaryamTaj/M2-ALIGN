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

Images themselves are downloaded here too (into ``<output-dir>/image_cache``
by default), not by the training script — this keeps train.py free of any
network dependency, which matters on clusters (e.g. Narval) whose compute
nodes have no internet access. Pairs whose image fails to download are
dropped from the JSONL. Pass ``--skip-download`` to only write the JSONL.

Usage
-----
    python load_image.py \\
        --languages fr,de,zh,ar,hi,sw \\
        --n-per-language 50000 \\
        --output-dir ./data
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import logging
import os
import random
import time
from datetime import datetime

import requests
from datasets import load_dataset
from PIL import Image

# How often (in rows scanned) to log progress / persist a checkpoint.
PROGRESS_EVERY = 20_000
CHECKPOINT_EVERY = 50_000


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

    No terminal punctuation is added: many scripts (Bengali/Hindi ``।``,
    CJK ``。``, ...) don't end sentences with a Latin period, and forcing
    one on — or worse, appending it after an existing non-Latin terminator
    — introduces an artifact not present in the source language. Whatever
    punctuation (or lack of it) the reference caption has is kept as-is.
    """
    if not ref or not ref.strip():
        return None
    return ref.strip()


# Columns we actually read. ``image`` (JPEG bytes) and ``embedding`` (a
# 2048-d float64 vector per row) make up most of each row's payload but are
# never used here, so they're pruned before streaming to cut bandwidth and
# time on a ~6.5M-row pass.
_WIT_BASE_COLUMNS = ["image_url", "caption_attribution_description", "wit_features"]


def _load_wit_base(logger: logging.Logger):
    """Stream the full ``wikimedia/wit_base`` dataset (parquet, no script).

    Replaces ``google/WIT``, whose Hub repo only ships a loading script;
    ``datasets>=4.0`` removed script execution (``trust_remote_code`` is no
    longer honoured), so ``google/WIT`` can no longer be loaded at all.
    """
    logger.info("Streaming wikimedia/wit_base (columns: %s)...", _WIT_BASE_COLUMNS)
    ds = load_dataset("wikimedia/wit_base", split="train", streaming=True)
    return ds.select_columns(_WIT_BASE_COLUMNS)


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


def _image_cache_path(cache_dir: str, url: str) -> str:
    cache_key = hashlib.sha1(url.encode()).hexdigest()
    return os.path.join(cache_dir, cache_key + ".jpg")


# Wikimedia throttles/blocks requests whose User-Agent doesn't identify the
# client and a contact point (https://meta.wikimedia.org/wiki/User-Agent_policy).
# A generic UA gets a burst of requests through before being rate-limited --
# which is what caused a 745/28839 success rate in a prior run.
_DOWNLOAD_HEADERS = {
    "User-Agent": "M2-ALIGN/1.0 (https://github.com/MaryamTaj/M2-ALIGN)",
}


# Statuses worth retrying: 429 is Wikimedia's rate limiter (not a dead
# image), 5xx are usually transient. A burst of concurrent requests
# routinely trips 429 even with a compliant User-Agent (see the note on
# _DOWNLOAD_HEADERS) -- backing off and retrying recovers most of these,
# whereas treating 429 as permanent (the previous behaviour) silently
# dropped the pair instead.
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5


def _download_one_image(url: str, cache_dir: str) -> str | None:
    """Fetch *url* into the on-disk image cache if not already there.

    Uses the same ``sha1(url).jpg`` cache key that :class:`WITDataset` in
    ``train.py`` looks up at training time, so once every row has been
    downloaded here, training never needs the network.

    Returns:
        None if the image ends up cached (already present, or freshly
        downloaded and decoded); otherwise a short string describing why it
        failed (e.g. an HTTP status code), for aggregating failure reasons.
    """
    path = _image_cache_path(cache_dir, url)
    if os.path.exists(path):
        return None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=15, headers=_DOWNLOAD_HEADERS)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            tmp_path = path + ".tmp"
            img.save(tmp_path, format="JPEG", quality=85)
            os.replace(tmp_path, path)
            return None
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code
            if status not in _RETRYABLE_STATUSES or attempt == _MAX_RETRIES - 1:
                return f"HTTP {status}"
            retry_after = exc.response.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.isdigit() else min(30, 2 ** attempt)
            time.sleep(wait)
        except Exception as exc:
            return type(exc).__name__
    return "HTTP 429 (exhausted retries)"


def download_images(
    rows: list[dict],
    cache_dir: str,
    max_workers: int,
    logger: logging.Logger,
) -> set[str]:
    """Download every unique ``image_url`` in *rows* into *cache_dir*.

    Runs downloads concurrently across *max_workers* threads. Already-cached
    files are skipped, so re-running after a partial/interrupted run only
    fetches what's still missing.

    Args:
        rows: Pair dicts, each with an ``image_url`` field.
        cache_dir: Directory to save cached JPEGs into.
        max_workers: Number of concurrent download threads.
        logger: Logger instance.

    Returns:
        The subset of URLs that ended up successfully cached.
    """
    os.makedirs(cache_dir, exist_ok=True)
    urls = sorted({row["image_url"] for row in rows})
    logger.info("Downloading %d unique images -> %s", len(urls), cache_dir)

    ok: set[str] = set()
    done = 0
    failure_counts: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one_image, url, cache_dir): url for url in urls}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            done += 1
            reason = future.result()
            if reason is None:
                ok.add(url)
            else:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
            if done % PROGRESS_EVERY == 0 or done == len(urls):
                logger.info(
                    "Downloaded %d/%d images (%d failed so far)",
                    done, len(urls), done - len(ok),
                )

    logger.info("Image download complete: %d/%d succeeded", len(ok), len(urls))
    if failure_counts:
        top = sorted(failure_counts.items(), key=lambda kv: -kv[1])[:10]
        logger.info("Failure reasons (top %d): %s", len(top), dict(top))
    return ok


def _checkpoint_path(output_dir: str, lang_codes: list[str]) -> str:
    tag = "-".join(sorted(lang_codes))
    return os.path.join(output_dir, f".checkpoint_{tag}.json")


def _save_checkpoint(
    path: str,
    ds_state: dict,
    collected: dict[str, list[dict]],
    scanned: int,
    logger: logging.Logger,
) -> None:
    """Overwrite the checkpoint with the current position and results so far.

    Written to a temp file and atomically renamed so a crash mid-write can't
    leave a corrupt/partial checkpoint behind.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"ds_state": ds_state, "collected": collected, "scanned": scanned}, f)
    os.replace(tmp, path)
    logger.info(
        "Checkpoint saved: %d rows scanned | %s",
        scanned, {k: len(v) for k, v in collected.items()},
    )


def _load_checkpoint(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _stream_rows(ds_factory, resume_state: dict | None, logger: logging.Logger):
    """Iterate rows from a streaming dataset, automatically reconnecting and
    resuming from the last successfully read row if the connection drops.

    ``ds_factory`` rebuilds the ``IterableDataset`` from scratch -- once an
    iterator raises, it can't be resumed in place, so a fresh one is opened
    on every reconnect. ``ds.state_dict()`` is pure in-memory bookkeeping (no
    network I/O), so it's captured after every row: a mid-stream error loses
    at most the one row in flight, not a whole checkpoint interval. The
    on-disk checkpoint the caller writes periodically is a separate,
    coarser safety net for when the *process itself* dies (OOM, kill,
    lost SSH session without nohup/tmux) -- this generator only handles
    the connection dropping while the process stays alive.
    """
    ds = ds_factory()
    if resume_state:
        ds.load_state_dict(resume_state)
    it = iter(ds)
    last_state = resume_state
    attempt = 0
    while True:
        try:
            row = next(it)
        except StopIteration:
            return
        except Exception as exc:
            attempt += 1
            wait = min(60, 2 ** attempt)
            logger.warning(
                "Stream error (attempt %d): %s -- reconnecting in %ds",
                attempt, exc, wait,
            )
            time.sleep(wait)
            ds = ds_factory()
            if last_state:
                ds.load_state_dict(last_state)
            it = iter(ds)
            continue
        attempt = 0
        last_state = ds.state_dict()
        yield row, last_state


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_language_pairs(
    lang_codes: list[str],
    n_per_language: int,
    seed: int,
    logger: logging.Logger,
    output_dir: str,
) -> dict[str, list[dict]]:
    """Single pass over WIT: emit English-paired rows for every requested language.

    Each row already carries every language's caption for that image, so all
    target languages are collected in one stream instead of one stream per
    language.

    Progress is logged every :data:`PROGRESS_EVERY` rows, and a resumable
    checkpoint (rows scanned, pairs collected so far, and the stream's own
    read position) is written to *output_dir* every :data:`CHECKPOINT_EVERY`
    rows. If a checkpoint from a previous, interrupted run is found, this
    resumes from it instead of starting over. Dropped connections mid-stream
    are retried automatically (see :func:`_stream_rows`); the on-disk
    checkpoint only matters if the whole process dies.

    Args:
        lang_codes: ISO 639-1 codes to collect (must be keys of :data:`WIT_TO_NLLB`).
        n_per_language: Target number of output pairs per language. A
            value <= 0 means "no cap" — collect every image that has both
            a target-language and an English caption, streaming the whole
            dataset, and skip the down-sampling step.
        seed: Random seed for sampling.
        logger: Logger instance.
        output_dir: Directory to write the checkpoint file into.

    Returns:
        Dict mapping language code to its list of paired dicts.
    """
    wanted = set(lang_codes) | {"en"}

    ckpt_path = _checkpoint_path(output_dir, lang_codes)
    ckpt = _load_checkpoint(ckpt_path)
    if ckpt:
        collected: dict[str, list[dict]] = ckpt["collected"]
        scanned = ckpt["scanned"]
        resume_state = ckpt["ds_state"]
        logger.info(
            "Resuming from checkpoint: %d rows already scanned | %s",
            scanned, {k: len(v) for k, v in collected.items()},
        )
    else:
        collected = {code: [] for code in lang_codes}
        scanned = 0
        resume_state = None

    unlimited = n_per_language <= 0
    overcollect = None if unlimited else n_per_language * 3
    done: set[str] = {
        code for code in lang_codes
        if overcollect is not None and len(collected[code]) >= overcollect
    }

    for row, ds_state in _stream_rows(lambda: _load_wit_base(logger), resume_state, logger):
        if len(done) == len(lang_codes):
            break
        scanned += 1

        if scanned % PROGRESS_EVERY == 0:
            logger.info(
                "Scanned %d rows | %s",
                scanned, {code: len(collected[code]) for code in lang_codes},
            )
        if scanned % CHECKPOINT_EVERY == 0:
            _save_checkpoint(ckpt_path, ds_state, collected, scanned, logger)

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
            if overcollect is not None and len(collected[code]) >= overcollect:
                done.add(code)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        logger.info("Scan complete -- checkpoint removed.")

    sampled: dict[str, list[dict]] = {}
    for code in lang_codes:
        pairs = collected[code]
        if unlimited:
            sampled[code] = pairs
            logger.info("  %s: %d pairs found (unlimited, no sampling)", WIT_CODE_TO_NAME[code], len(pairs))
            continue
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

    total = {code: 0 for code in lang_codes}
    paired = {code: 0 for code in lang_codes}
    seen: dict[str, set[str]] = {code: set() for code in lang_codes}

    scanned = 0
    for row, _ds_state in _stream_rows(lambda: _load_wit_base(logger), None, logger):
        scanned += 1
        if scanned > max_rows:
            break
        if scanned % PROGRESS_EVERY == 0:
            logger.info(
                "Scanned %d / %d rows | %s",
                scanned, max_rows, {code: paired[code] for code in lang_codes},
            )
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
        help=(
            "Target number of image-caption pairs per language. Pass 0 (or "
            "negative) to collect every image with both a target-language "
            "and an English caption, with no cap and no down-sampling."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "data"),
    )
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
    parser.add_argument(
        "--image-cache-dir", type=str, default=None,
        help="Directory to download images into (default: <output-dir>/image_cache, "
             "matching train.py's --image-cache-dir default of ./data/image_cache).",
    )
    parser.add_argument(
        "--download-workers", type=int, default=4,
        help="Number of concurrent threads used to download images. Kept low by default "
             "since Wikimedia's image hosts rate-limit (HTTP 429) aggressive bursts.",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Only write wit_pairs.jsonl; don't download images. Useful when the "
             "images will be fetched separately or are already cached.",
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

    pairs_by_lang = build_language_pairs(
        lang_codes, args.n_per_language, args.seed, logger, args.output_dir,
    )

    all_rows: list[dict] = [row for rows in pairs_by_lang.values() for row in rows]
    random.Random(args.seed).shuffle(all_rows)

    if not args.skip_download:
        cache_dir = args.image_cache_dir or os.path.join(args.output_dir, "image_cache")
        ok_urls = download_images(all_rows, cache_dir, args.download_workers, logger)
        before = len(all_rows)
        all_rows = [row for row in all_rows if row["image_url"] in ok_urls]
        logger.info("Kept %d/%d pairs with a successfully cached image", len(all_rows), before)

    out_path = os.path.join(args.output_dir, "wit_pairs.jsonl")
    _write_jsonl(out_path, all_rows, logger)

    by_lang: dict[str, int] = {}
    for row in all_rows:
        by_lang[row["language_code"]] = by_lang.get(row["language_code"], 0) + 1
    logger.info("Total pairs: %d | per-language: %s", len(all_rows), by_lang)


if __name__ == "__main__":
    main()
