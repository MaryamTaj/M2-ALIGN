"""Download Stage 3b VQA *evaluation* data — real/native benchmarks only.

Hard rule for this file (see the Stage 3 plan): no NLLB-translated data is
ever written here, not even as a fallback. A model scored against its own
translation system's output would measure translation-consistency, not
VQA skill. Every row this script writes comes from a human-authored or
professionally-translated source.

Current active scope (see the Stage 3 language survey), one row per
language across every VQA benchmark this project's Stage 1/2 covers it
for -- policy is to run every benchmark a language qualifies for, not just
one, since xGQA/CVQA/WorldCuisines are independent evaluations, not
interchangeable:

    bn         xGQA + CVQA
    de         xGQA only (no CVQA subset for German at all)
    ru, zh     xGQA + CVQA
    pt, id, ko xGQA + CVQA
    jv         CVQA (+ WorldCuisines, deferred) -- no xGQA coverage
    mn, si, ga CVQA only -- no xGQA or WorldCuisines coverage

CVQA is scored open-ended-via-likelihood -- see evaluate.py -- not as
visible-choice multiple-choice, per the CVQA paper's own open-ended
protocol. Yoruba is dropped: no xGQA/CVQA/WorldCuisines coverage at all.
WorldCuisines is otherwise deferred even where it has coverage (id/jv) --
its loader is kept below, working and unchanged, as ready-to-use
infrastructure if that scope is widened.

Sources
-------
xGQA (Pfeiffer et al. 2022) -- professionally-translated slice of GQA's own
    test-dev set, 8 languages total (bn/de/en/id/ko/pt/ru/zh); we use all 8
    that this project has a checkpoint for: bn/de/ru/zh/pt/id/ko. Not on the
    HF Hub as a `load_dataset` id; distributed as per-language JSON files in the
    adapter-hub/xGQA GitHub repo
    (`data/zero_shot/testdev_balanced_questions_{lang}.json`), each a dict
    keyed by question id with fields `question` (translated), `imageId`
    (GQA/Visual Genome image id), and `answer` (English). Images
    themselves are GQA's own corpus, not re-hosted per-question --
    download+extract https://nlp.stanford.edu/data/gqa/images.zip once and
    pass its path via --gqa-images-dir.

WorldCuisines (Mohamed et al., NAACL 2025) -- `worldcuisines/vqa` on the
    Hub, configs `task1`/`task2`, splits `train`/`test_small`/`test_large`.
    Filtered client-side by the `lang` field (exact tag verified against a
    live sample at load time, logged for a sanity check -- see the Stage 3
    verification plan's "data sanity" step). Covers Indonesian/Javanese
    only -- no Bengali coverage -- deferred, see above.

CVQA (NeurIPS 2024) -- `afaji/cvqa` on the Hub, single `test` split,
    filtered by the `Subset` field (language-country pair string). Covers
    Indonesian and Javanese, plus -- for the newer Stage 1/2 language set
    that has no MGSM/MSVAMP coverage and no active xGQA slot -- Portuguese
    ('Portuguese','Brazil'), Korean ('Korean','South Korea'), Mongolian
    ('Mongolian','Mongolia'), Sinhala ('Sinhala', 'Sri_Lanka'), and Irish
    ('Irish','Ireland'). No Bengali or German coverage (German has no CVQA
    subset at all).

Usage
-----
    # Run from anywhere -- output_dir defaults to $SCRATCH/M2-ALIGN/Stage3/data:
    python Stage3/load_evaluation_data.py --benchmark xgqa --languages bn,de,ru,zh,pt,id,ko

    # bn/ru/zh/pt/id/ko all have BOTH xGQA and CVQA coverage -- run both,
    # they're independent evaluations, not redundant (zh here is
    # mainland/China only, not the separate Singapore subset CVQA also has):
    python Stage3/load_evaluation_data.py --benchmark cvqa --languages bn,ru,zh,pt,id,ko

    # jv/mn/si/ga: CVQA only -- no xGQA coverage for any of these four.
    python Stage3/load_evaluation_data.py --benchmark cvqa --languages jv,mn,si,ga

    # Deferred (Indonesian/Javanese via worldcuisines) -- kept working for
    # later, not run today; CVQA coverage for id/jv is active above:
    python Stage3/load_evaluation_data.py --benchmark worldcuisines --languages id,jv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime

import requests
from datasets import load_dataset

# Data/outputs/logs live on $SCRATCH, not in the git checkout.
SCRATCH_ROOT = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN", "Stage3")

XGQA_RAW_BASE = "https://raw.githubusercontent.com/adapter-hub/xGQA/master/data/zero_shot"
# All 8 of xGQA's languages that this project has a Stage 1/2 checkpoint
# for. Policy: wherever a language has both xGQA and CVQA coverage, run
# both -- they're independent benchmarks (professionally-translated
# multiple-choice-as-open-ended vs. crowd-sourced open-ended-via-
# likelihood), not redundant. jv/mn/si/ga have no xGQA coverage at all
# (outside its 8 languages), so CVQA is their only VQA-eval path.
XGQA_LANGS = {"bn", "de", "en", "ru", "zh", "pt", "id", "ko"}

# Candidate substrings to match against WorldCuisines' `lang` field and
# CVQA's `Subset` field. Verify the actual match against the logged
# unique-value dump before trusting downstream training/eval numbers.
WORLDCUISINES_LANG_CANDIDATES: dict[str, list[str]] = {
    "id": ["ind", "id", "indonesian"],
    "jv": ["jav", "jv", "javanese"],
}
CVQA_SUBSET_CANDIDATES: dict[str, list[str]] = {
    "id": ["indonesian"],
    "jv": ["javanese"],
    # Added to cover Stage 1/2's newer language set (none of these five have
    # MGSM/MSVAMP coverage). pt/ko are also active in xGQA -- run both, per
    # policy above; mn/si/ga have no xGQA coverage, so CVQA is their only
    # VQA-eval path. Subset values are literally "('<Language>', '<Country>')"
    # strings (verified against afaji/cvqa's actual Subset column) -- each of
    # these five names only one country, so a single substring is unambiguous.
    # German has no CVQA subset at all (verified: not among its 39 subsets),
    # so it stays uncovered by this benchmark.
    "pt": ["portuguese"],
    "ko": ["korean"],
    "mn": ["mongolian"],
    "si": ["sinhala"],
    "ga": ["irish"],
    # Also covered by xGQA -- per policy above, run both benchmarks rather
    # than picking one. "zh" deliberately matches "china" (not
    # "chinese"/"zh"): CVQA ships two Chinese subsets,
    # ('Chinese','China') and ('Chinese','Singapore'); xGQA's zh is
    # mainland/simplified Chinese, so only the China subset is the fair
    # comparison -- Singapore stays excluded. No "de" entry: German has no
    # CVQA subset at all (verified against all 39 subsets), so
    # `--languages de` correctly raises rather than silently matching 0 rows.
    "bn": ["bengali"],
    "ru": ["russian"],
    "zh": ["china"],
}


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("stage3b_load_evaluation_data")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"load_evaluation_data_{ts}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
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


# ---------------------------------------------------------------------------
# xGQA
# ---------------------------------------------------------------------------

def load_xgqa(lang: str, logger: logging.Logger) -> list[dict]:
    """Download and parse one xGQA test-dev language file.

    Args:
        lang: ISO 639-1 code, must be in :data:`XGQA_LANGS`.
        logger: Logger instance.

    Returns:
        List of row dicts: ``{id, vg_image_id, query, answer,
        source_language, source_dataset}``.
    """
    if lang not in XGQA_LANGS:
        raise ValueError(f"xGQA has no Stage-3 coverage for {lang!r}; expected one of {sorted(XGQA_LANGS)}")

    url = f"{XGQA_RAW_BASE}/testdev_balanced_questions_{lang}.json"
    logger.info("Downloading xGQA test-dev (%s): %s", lang, url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    logger.info("xGQA %s: %d questions", lang, len(data))

    rows = []
    for qid, entry in data.items():
        question = entry.get("question")
        answer = entry.get("answer")
        image_id = entry.get("imageId")
        if not question or answer is None or not image_id:
            continue
        rows.append({
            "id": qid,
            "vg_image_id": str(image_id),
            "query": question,
            "answer": str(answer),
            "source_language": lang,
            "source_dataset": "xgqa_testdev",
        })
    return rows


# ---------------------------------------------------------------------------
# WorldCuisines
# ---------------------------------------------------------------------------

def _resolve_worldcuisines_lang_tag(ds, candidates: list[str], logger: logging.Logger) -> str:
    """Find the actual `lang` value matching one of *candidates*.

    Logs the full set of unique `lang` values seen (capped) so a human can
    sanity-check the match — see the Stage 3 "data sanity" verification
    step; this loader refuses to guess silently.
    """
    seen: set[str] = set()
    for row in ds:
        seen.add(str(row["lang"]))
        if len(seen) > 200:
            break
    logger.info("WorldCuisines observed lang tags (sample): %s", sorted(seen))
    for cand in candidates:
        for tag in seen:
            if cand.lower() in tag.lower():
                logger.info("Matched candidate %r -> lang tag %r", cand, tag)
                return tag
    raise ValueError(
        f"None of {candidates} matched any observed WorldCuisines `lang` tag: {sorted(seen)}. "
        "Update WORLDCUISINES_LANG_CANDIDATES with the correct tag."
    )


def load_worldcuisines(lang: str, split: str, task: str, logger: logging.Logger) -> list[dict]:
    """Load a WorldCuisines held-out split filtered to one language.

    Args:
        lang: One of the keys in :data:`WORLDCUISINES_LANG_CANDIDATES` (``id``/``jv``).
        split: ``"test_small"`` or ``"test_large"``.
        task: ``"task1"`` or ``"task2"`` (multiple-choice vs. open-ended,
            per the dataset's own task split).
        logger: Logger instance.

    Returns:
        List of row dicts: ``{id, image_url, query, answer,
        source_language, source_dataset}``.
    """
    if lang not in WORLDCUISINES_LANG_CANDIDATES:
        raise ValueError(f"No WorldCuisines coverage configured for {lang!r}")

    logger.info("Loading worldcuisines/vqa [%s/%s] ...", task, split)
    ds = load_dataset("worldcuisines/vqa", task, split=split)
    logger.info("worldcuisines/vqa %s/%s: %d rows total", task, split, len(ds))

    tag = _resolve_worldcuisines_lang_tag(ds, WORLDCUISINES_LANG_CANDIDATES[lang], logger)
    ds_lang = ds.filter(lambda r: str(r["lang"]) == tag)
    logger.info("Filtered to lang=%r: %d rows", tag, len(ds_lang))

    rows = []
    for r in ds_lang:
        query = r.get("multi_choice_prompt") or r.get("open_ended_prompt")
        answer = r.get("multi_choice_answer")
        if answer is None:
            answer = r.get("answer")
        if not query or answer is None:
            continue
        rows.append({
            "id": str(r.get("qa_id")),
            "image_url": r.get("image_url") or r.get("image_path"),
            "query": query,
            "answer": str(answer),
            "source_language": lang,
            "source_dataset": f"worldcuisines_{task}_{split}",
        })
    return rows


# ---------------------------------------------------------------------------
# CVQA
# ---------------------------------------------------------------------------

def _resolve_cvqa_subsets(ds, candidates: list[str], logger: logging.Logger) -> set[str]:
    """Find the `Subset` values matching one of *candidates* (substring match)."""
    seen: set[str] = set()
    for row in ds:
        seen.add(str(row["Subset"]))
    matched = {s for s in seen if any(c.lower() in s.lower() for c in candidates)}
    logger.info("CVQA candidates=%s -> matched Subsets=%s (of %d total)", candidates, sorted(matched), len(seen))
    if not matched:
        raise ValueError(f"None of {candidates} matched any CVQA Subset: {sorted(seen)}")
    return matched


def load_cvqa(lang: str, image_cache_dir: str, logger: logging.Logger) -> list[dict]:
    """Load CVQA test rows filtered to one language, saving each row's image locally.

    Args:
        lang: One of the keys in :data:`CVQA_SUBSET_CANDIDATES` (e.g. ``mn``/``si``/``ga``).
        image_cache_dir: Directory each row's image is saved into, as
            ``<row id>.jpg``.
        logger: Logger instance.

    Returns:
        List of row dicts: ``{id, query, choices, answer_index,
        source_language, source_dataset}``. A row's image lives at
        ``<image_cache_dir>/<id>.jpg``.

    Images come from the dataset's own embedded ``image`` feature (decoded
    locally from the already-downloaded parquet files -- no extra network
    fetch), *not* from the ``Image Source`` field. A live sample of
    ``afaji/cvqa`` showed ``Image Source`` is not usable as a download URL
    for most rows: wherever ``Image Type == "Self"`` (contributor's own,
    unpublished photo), ``Image Source`` is a literal sentinel string
    (``"Self-open"`` / ``"Self-research_only"``), not a URL at all --
    fetching it raises immediately. Many ``"External"`` rows fare no
    better: ``Image Source`` is often a Commons/Flickr *page* URL (e.g.
    ``.../wiki/File:Foo.jpg``), not the raw file, so the response is HTML,
    not an image. Worse, every ``"Self-open"`` row shares the exact same
    sentinel string, so keying a cache by ``sha1(Image Source)`` (the
    pattern used for WIT/WorldCuisines) silently collides multiple rows'
    images onto the same cache entry. The embedded ``image`` feature and
    the row's own unique ``ID`` sidestep both problems.

    Note:
        In ``afaji/cvqa``, ``"Question"`` is the *original* native-language
        question (posed by a native speaker) and ``"Translated Question"``
        is its English translation -- the reverse of what the field names
        suggest. ``query`` must stay in the native language to match every
        other benchmark in this pipeline (xGQA, GQA training data, MGSM):
        the untranslated question feeds both the NLLB encoder and the LLM
        prompt `T`. ``choices`` intentionally keeps preferring the English
        ``"Translated Options"``, matching the project-wide convention that
        answers stay in English (see Stage3/load_translated_data.py's docstring).
    """
    if lang not in CVQA_SUBSET_CANDIDATES:
        raise ValueError(f"No CVQA coverage configured for {lang!r}")

    logger.info("Loading afaji/cvqa [test] ...")
    ds = load_dataset("afaji/cvqa", split="test")
    logger.info("afaji/cvqa test: %d rows total", len(ds))

    subsets = _resolve_cvqa_subsets(ds, CVQA_SUBSET_CANDIDATES[lang], logger)
    ds_lang = ds.filter(lambda r: str(r["Subset"]) in subsets)
    logger.info("Filtered to Subset in %s: %d rows", subsets, len(ds_lang))

    os.makedirs(image_cache_dir, exist_ok=True)
    rows = []
    n_image_failed = 0
    for r in ds_lang:
        query = r.get("Question") or r.get("Translated Question")
        choices = r.get("Translated Options") or r.get("Options")
        label = r.get("Label")
        image = r.get("image")
        row_id = str(r.get("ID"))
        if not query or not choices or label is None or image is None or not row_id:
            continue
        image_path = os.path.join(image_cache_dir, f"{row_id}.jpg")
        if not os.path.exists(image_path):
            try:
                image.convert("RGB").save(image_path, format="JPEG", quality=85)
            except Exception:
                n_image_failed += 1
                continue
        rows.append({
            "id": row_id,
            "query": query,
            "choices": list(choices),
            "answer_index": int(label),
            "source_language": lang,
            "source_dataset": "cvqa_test",
        })
    if n_image_failed:
        logger.info("CVQA %s: %d rows dropped (image decode/save failed)", lang, n_image_failed)
    logger.info("CVQA %s: %d rows with saved images -> %s", lang, len(rows), image_cache_dir)
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download Stage 3b VQA evaluation data (real benchmarks only).")
    parser.add_argument("--benchmark", required=True, choices=["xgqa", "worldcuisines", "cvqa"])
    parser.add_argument("--languages", type=str, required=True,
                        help="Comma-separated ISO codes, e.g. 'bn,id' for xgqa or 'id,jv' for worldcuisines/cvqa.")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Defaults to $SCRATCH/M2-ALIGN/Stage3/data.",
    )
    parser.add_argument("--worldcuisines-split", type=str, default="test_small",
                        choices=["test_small", "test_large"])
    parser.add_argument("--worldcuisines-task", type=str, default="task1", choices=["task1", "task2"])
    parser.add_argument(
        "--cvqa-image-cache-dir", type=str, default=None,
        help="Where each CVQA row's image is saved, as <id>.jpg. Defaults to "
             "<output_dir>/cvqa/images -- pass evaluate.py's --image-cache-dir the same path.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(SCRATCH_ROOT, "data")
    if args.cvqa_image_cache_dir is None:
        args.cvqa_image_cache_dir = os.path.join(args.output_dir, "cvqa", "images")

    logger = setup_logging(os.path.join(SCRATCH_ROOT, "logs"))
    langs = [x.strip() for x in args.languages.split(",") if x.strip()]

    for lang in langs:
        if args.benchmark == "xgqa":
            rows = load_xgqa(lang, logger)
            out_path = os.path.join(args.output_dir, "xgqa", f"{lang}.jsonl")
        elif args.benchmark == "worldcuisines":
            rows = load_worldcuisines(lang, args.worldcuisines_split, args.worldcuisines_task, logger)
            out_path = os.path.join(args.output_dir, "worldcuisines", f"{lang}.jsonl")
        else:
            rows = load_cvqa(lang, args.cvqa_image_cache_dir, logger)
            out_path = os.path.join(args.output_dir, "cvqa", f"{lang}.jsonl")
        write_jsonl(out_path, rows, logger)

    logger.info("Done.")


if __name__ == "__main__":
    main()
