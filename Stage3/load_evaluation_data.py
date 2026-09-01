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

    bn         xGQA + CVQA + WorldCuisines (task1 + task2)
    de         xGQA only (no CVQA or WorldCuisines coverage for German at all)
    ru, zh     xGQA + CVQA + WorldCuisines (task1 + task2)
    pt         xGQA + CVQA (no WorldCuisines coverage for Portuguese)
    id, ko     xGQA + CVQA + WorldCuisines (task1 + task2)
    jv, si     CVQA + WorldCuisines (task1 + task2) -- no xGQA coverage
    mn, ga     CVQA only -- no xGQA or WorldCuisines coverage
    am, ig, om CVQA only -- no xGQA or WorldCuisines coverage (big-headroom
               low-resource languages; see the CVQA docstring below)

CVQA is scored open-ended-via-likelihood -- see evaluate.py -- not as
visible-choice multiple-choice, per the CVQA paper's own open-ended
protocol. Yoruba is dropped: no xGQA/CVQA/WorldCuisines coverage at all.
WorldCuisines task1 (Dish Name Prediction) and task2 (Dish Origin
Prediction) are both run as independent benchmark rows -- see
`--benchmark worldcuisines_task1` / `--benchmark worldcuisines_task2` below --
for every language WorldCuisines covers.

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
    verification plan's "data sanity" step). Covers bn/ru/zh/id/ko/jv/si --
    verified directly against the `task1`/`test_small` JSONL files on the
    Hub (not just the dataset viewer). No pt/de/mn/ga coverage at all (pt
    in particular is absent despite having xGQA+CVQA coverage elsewhere).
    id/ko/ru each ship two registers (`_casual`/`_formal`); jv ships
    `_ngoko`/`_krama` (a real register distinction in Javanese, not just
    formality of phrasing). This project always uses the formal variant
    (`_krama` for Javanese) for consistency with xGQA/CVQA's standard-
    register text -- see `WORLDCUISINES_LANG_TAGS`. si/bn/zh each have
    only one `lang` tag (`si_formal_spoken`/`bn`/`zh_cn`), no register
    choice needed.

    Scoring diverges from xGQA/CVQA -- see evaluate.py's
    `evaluate_worldcuisines_open_ended`/`worldcuisines_correct` -- to match
    the WorldCuisines authors' own protocol (`evaluation/score/score.py` in
    their GitHub repo, not just the paper text): dual-reference (this row's
    own-language answer + the qa_id's English answer, built in
    `load_worldcuisines` below) and case-insensitive substring containment,
    not xGQA's exact-match. Their own model-querying harness
    (`evaluation/src/qwen.py`) also sends no system prompt and no added
    "single word/short phrase" instruction -- just `open_ended_prompt`
    verbatim as the user turn. This project keeps its existing system
    message and English-answer instruction anyway, for consistency with
    xGQA/CVQA (see evaluate.py's module docstring for the rationale).

    Both configs (`task1`/`task2`) share this exact schema -- verified
    directly against `task2`'s own JSONL files, not assumed from the
    dataset card. `task1` is Dish Name Prediction ("what is this food
    called"); `task2` is Dish Origin Prediction ("which country made this
    dish popular") -- a different skill (food-origin world knowledge, not
    visual dish recognition), and a smaller split (100 rows/language in
    `test_small` vs. task1's 300). This project runs both, written to
    `worldcuisines/task1/`/`worldcuisines/task2/` subdirectories under one
    shared `worldcuisines/` output dir (see `main`) -- mirroring CVQA's own
    `cvqa/{lang}.jsonl` + `cvqa/images/` layout -- so they can be evaluated
    as independent benchmark rows while sharing one image cache.

    Empirically, task1's own-language `answer` field is frequently just
    the English name left untranslated -- verified directly against all 7
    active languages' `test_small` files: 100% for `zh_cn`/
    `si_formal_spoken`, 95%/93% for `id_formal`/`bn`, 60-76% for
    `ko_formal`/`ru_formal`/`jv_krama`. Every `qa_id` in every active
    language's file has a matching English row, so `answers` here is
    always `{english}` (the common case) or `{native, english}` (whenever
    a distinct native translation exists) -- never native-only. See
    evaluate.py's module docstring for how this shapes the prompting
    decision (`build_worldcuisines_prompt` keeps the "in English"
    instruction specifically because of this skew).

CVQA (NeurIPS 2024) -- `afaji/cvqa` on the Hub, single `test` split,
    filtered by the `Subset` field (language-country pair string). Covers
    Indonesian and Javanese, plus -- for the newer Stage 1/2 language set
    that has no MGSM/MSVAMP coverage and no active xGQA slot -- Portuguese
    ('Portuguese','Brazil'), Korean ('Korean','South Korea'), Mongolian
    ('Mongolian','Mongolia'), Sinhala ('Sinhala', 'Sri_Lanka'), and Irish
    ('Irish','Ireland'). No Bengali or German coverage (German has no CVQA
    subset at all).

    Amharic ('Amharic','Ethiopia'), Igbo ('Igbo','Nigeria'), and Oromo
    ('Oromo','Ethiopia') were added to target languages where Qwen's own
    multilingual coverage is weakest -- the mapping layer has the most
    headroom to help there. Unlike the entries above, these three
    Subset-string guesses are NOT yet verified against afaji/cvqa's actual
    Subset column -- run with these languages once and check the logged
    `_resolve_cvqa_subsets` output (candidates=...->matched Subsets=...)
    before trusting downstream numbers.

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

    # am/ig/om: CVQA only, big-headroom low-resource languages -- run this
    # once and check the logged matched-Subsets output before trusting
    # downstream numbers (see the CVQA docstring above).
    python Stage3/load_evaluation_data.py --benchmark cvqa --languages am,ig,om

    # WorldCuisines task1 (Dish Name Prediction) -- active for
    # bn/ru/zh/id/ko/jv/si (formal register, krama for Javanese); no
    # pt/de/mn/ga coverage. Writes worldcuisines/task1/{lang}.jsonl plus
    # worldcuisines/images/ (shared with task2 below):
    python Stage3/load_evaluation_data.py --benchmark worldcuisines_task1 --languages bn,ru,zh,id,ko,jv,si

    # WorldCuisines task2 (Dish Origin Prediction) -- same 7 languages,
    # written to worldcuisines/task2/{lang}.jsonl:
    python Stage3/load_evaluation_data.py --benchmark worldcuisines_task2 --languages bn,ru,zh,id,ko,jv,si
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import time
from datetime import datetime

import requests
from datasets import load_dataset
from PIL import Image

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

# Exact WorldCuisines `lang` tag per language -- verified directly against
# the `task1`/`test_small` JSONL files on the Hub (each file's `lang` field
# equals its filename prefix, e.g. `id_formal_small_eval_task1.jsonl` rows
# all have `lang == "id_formal"`). Always the formal register where a
# split exists (krama for Javanese), to match xGQA/CVQA's standard-register
# text -- see the WorldCuisines paragraph in this module's docstring.
WORLDCUISINES_LANG_TAGS: dict[str, str] = {
    "id": "id_formal",
    "jv": "jv_krama",
    "ko": "ko_formal",
    "ru": "ru_formal",
    "si": "si_formal_spoken",
    "bn": "bn",
    "zh": "zh_cn",
}

# Candidate substrings to match against CVQA's `Subset` field. Verify the
# actual match against the logged unique-value dump before trusting
# downstream training/eval numbers.
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
    # Big-headroom, low-resource languages with no xGQA/WorldCuisines
    # coverage -- CVQA is their only VQA-eval path. Subset strings are
    # unverified against afaji/cvqa's actual Subset column (unlike the
    # entries above) -- confirm via _resolve_cvqa_subsets's logged
    # candidates=...->matched Subsets=... output before trusting results.
    "am": ["amharic"],
    "ig": ["igbo"],
    "om": ["oromo"],
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
        # Original GQA question-type annotation, carried straight through
        # from the raw testdev file -- structural (verify/query/choose/
        # logical/compare) and semantic (object/attribute/category/
        # relation/global) tags, used for the per-type accuracy breakdown
        # in Stage3/analysis/aggregate_breakdown.py. Not every entry has a
        # `types` block (a handful of testdev rows omit it), hence `.get`.
        types = entry.get("types") or {}
        rows.append({
            "id": qid,
            "vg_image_id": str(image_id),
            "query": question,
            "answer": str(answer),
            "source_language": lang,
            "source_dataset": "xgqa_testdev",
            "question_type_structural": types.get("structural"),
            "question_type_semantic": types.get("semantic"),
        })
    return rows


# ---------------------------------------------------------------------------
# WorldCuisines
# ---------------------------------------------------------------------------

def _resolve_worldcuisines_lang_tag(ds, expected_tag: str, logger: logging.Logger) -> str:
    """Verify *expected_tag* is an actual `lang` value in *ds*.

    Logs the full set of observed `lang` values so a human can sanity-check
    the match — see the Stage 3 "data sanity" verification step; this
    loader refuses to guess silently. Exact match, not substring: WorldCuisines
    ships casual/formal (and jv's ngoko/krama) register splits under
    distinct `lang` tags, and a substring match risks silently landing on
    the wrong register.
    """
    seen: set[str] = {str(row["lang"]) for row in ds}
    logger.info("WorldCuisines observed lang tags: %s", sorted(seen))
    if expected_tag not in seen:
        raise ValueError(
            f"Expected WorldCuisines lang tag {expected_tag!r} not found among observed tags: "
            f"{sorted(seen)}. Update WORLDCUISINES_LANG_TAGS with the correct tag."
        )
    return expected_tag


def load_worldcuisines(lang: str, split: str, task: str, logger: logging.Logger) -> list[dict]:
    """Load a WorldCuisines held-out split filtered to one language.

    Args:
        lang: One of the keys in :data:`WORLDCUISINES_LANG_TAGS`.
        split: ``"test_small"`` or ``"test_large"``.
        task: ``"task1"`` (Dish Name Prediction, e.g. "what is this food
            called") or ``"task2"`` (Dish Origin Prediction, e.g. "which
            country made this dish popular") -- both configs carry the same
            multi_choice_prompt/open_ended_prompt schema; this project only
            ever loads task1.
        logger: Logger instance.

    Returns:
        List of row dicts: ``{id, image_url, query, answers,
        source_language, source_dataset}``. ``answers`` is a list (dual
        reference: this row's own-language answer + the qa_id's English
        answer), not a single string -- see the "dual reference" note below.

    ``query``/native answer deliberately come from ``open_ended_prompt``/
    ``answer`` (free-text dish name), not ``multi_choice_prompt``/
    ``multi_choice_answer`` (a 5-way index): the latter's prompt embeds
    the gold answer as one of its visible options -- scoring against it
    would leak the answer and evaluate a completely different task than
    the "open-ended, English short answers" this benchmark is documented
    as (see evaluate.py's module docstring), inconsistent with xGQA/CVQA.

    No filtering by `prompt_type` -- every row in the split is kept, so
    task1 rows here are an even 3-way mix of the benchmark's three prompt
    settings (verified against a live sample: 100/100/100 of 300 in
    `test_small`): `prompt_type=1` no-context ("What is this food
    called?"), `=3` contextualized ("What is the local name for this dish
    in <LOCATION>?"), `=4` adversarial ("I like <CUISINE> food. What is
    this dish called?" -- see the source repo's
    `resources/prompt_query_template.csv` for the full template set).
    task2 rows are entirely `prompt_type=2`, no-context only (e.g. "Where
    is this dish from?") -- the source templates have no with_context or
    adversarial variant for the origin-prediction question.

    Dual reference (matches the WorldCuisines authors' own `score_oe`
    `oe_mode="dual"` in their `evaluation/score/score.py`): a row's own
    `answer` field is NOT always English -- e.g. ru_formal's answer for
    the "stuffed eggplant" dish is "Фаршированные баклажаны" (Cyrillic),
    jv_krama's is "Tèrong isi" -- while this project's eval prompt (see
    `build_open_ended_prompt` in evaluate.py) instructs the model to
    answer *in English*. Scoring only against the row's own (possibly
    non-English) answer, as the original single-reference version of this
    loader did, would mark a correct English answer wrong for any language
    whose native dish name differs from English. Joining in the same
    qa_id's `lang == "en"` row's answer as a second reference fixes that,
    while keeping the native answer too in case the model answers natively
    despite the instruction. The `en` filter is free -- `ds` already holds
    the whole split in memory from the `load_dataset` call above, so this
    doesn't trigger a second download.
    """
    if lang not in WORLDCUISINES_LANG_TAGS:
        raise ValueError(f"No WorldCuisines coverage configured for {lang!r}")

    logger.info("Loading worldcuisines/vqa [%s/%s] ...", task, split)
    ds = load_dataset("worldcuisines/vqa", task, split=split)
    logger.info("worldcuisines/vqa %s/%s: %d rows total", task, split, len(ds))

    tag = _resolve_worldcuisines_lang_tag(ds, WORLDCUISINES_LANG_TAGS[lang], logger)
    ds_lang = ds.filter(lambda r: str(r["lang"]) == tag)
    logger.info("Filtered to lang=%r: %d rows", tag, len(ds_lang))

    ds_en = ds.filter(lambda r: str(r["lang"]) == "en")
    en_answer_by_qa_id = {str(r["qa_id"]): r["answer"] for r in ds_en if r.get("answer")}
    logger.info("English reference answers available for %d qa_ids", len(en_answer_by_qa_id))

    rows = []
    for r in ds_lang:
        query = r.get("open_ended_prompt") or r.get("multi_choice_prompt")
        native_answer = r.get("answer")
        if native_answer is None:
            native_answer = r.get("multi_choice_answer")
        qa_id = str(r.get("qa_id")) if r.get("qa_id") is not None else None
        if not query or native_answer is None or not qa_id:
            continue
        answers = {str(native_answer)}
        en_answer = en_answer_by_qa_id.get(qa_id)
        if en_answer:
            answers.add(str(en_answer))
        # `prompt_type` distinguishes the three question framings this loader
        # deliberately keeps mixed together (see this function's docstring):
        # 1=no-context, 3=contextualized (location named in the question),
        # 4=adversarial (a distractor cuisine named in the question). task2
        # rows are always prompt_type=2 (no-context origin question) -- kept
        # as-is rather than remapped, so a raw value of 2 is the signal that
        # a row came from task2 when task1/task2 results get pooled. Used by
        # Stage3/analysis/aggregate_breakdown.py's context-role breakdown.
        prompt_type = r.get("prompt_type")
        rows.append({
            "id": qa_id,
            "image_url": r.get("image_url") or r.get("image_path"),
            "query": query,
            "answers": sorted(answers),
            "source_language": lang,
            "source_dataset": f"worldcuisines_{task}_{split}",
            "prompt_type": int(prompt_type) if prompt_type is not None else None,
        })
    return rows


_WIKIMEDIA_THUMB_RE = re.compile(
    r"^(https?://upload\.wikimedia\.org/wikipedia/commons)/thumb/([^/]+)/([^/]+)/([^/]+)/[^/]+$"
)


def _wikimedia_original_url(url: str) -> str:
    """Strip Wikimedia Commons' thumbnail sizing down to the original file.

    A large share of WorldCuisines' `image_url` values point at a custom-
    width Commons thumbnail, e.g.
    `.../commons/thumb/1/13/Foo.jpg/1629px-Foo.jpg?download`. Verified
    directly against a live run's logs: Wikimedia now rejects most custom
    widths outright (`400 Use thumbnail sizes listed on
    https://w.wiki/GHai` -- only a fixed allow-list of widths is generated
    on demand). The plain, un-thumbed original --
    `.../commons/1/13/Foo.jpg?download` -- carries no such restriction.
    Non-thumb/non-Wikimedia URLs pass through unchanged.
    """
    base, sep, query = url.partition("?")
    m = _WIKIMEDIA_THUMB_RE.match(base)
    if not m:
        return url
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}/{m.group(4)}{sep}{query}"


def _fetch_with_retry(
    session: requests.Session, url: str, logger: logging.Logger,
    max_attempts: int = 5, base_delay: float = 5.0,
) -> bytes | None:
    """GET *url* with exponential backoff on 429 / transient errors.

    Verified necessary against a live run: firing ~9 req/s at
    upload.wikimedia.org with no delay and a generic User-Agent got most
    requests blocked within seconds (`429 ... contact noc@wikimedia.org`)
    -- this is Wikimedia's anti-abuse layer, not a per-request rate limit
    that a single retry clears. `session` should carry a compliant
    User-Agent (see Wikimedia's User-Agent policy) to reduce the odds of
    being blocked in the first place. Non-429 HTTP errors (e.g. the 400
    `_wikimedia_original_url` is meant to avoid) are treated as permanent
    -- no point retrying a request that will fail identically every time.
    """
    delay = base_delay
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, timeout=20)
        except requests.exceptions.RequestException as e:
            if attempt == max_attempts:
                logger.warning("WorldCuisines image download failed for %s: %s", url, e)
                return None
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
            continue
        if resp.status_code == 429:
            if attempt == max_attempts:
                logger.warning("WorldCuisines image download failed for %s: repeated 429 (rate-limited)", url)
                return None
            retry_after = resp.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else delay
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
            continue
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.warning("WorldCuisines image download failed for %s: %s", url, e)
            return None
        return resp.content
    return None


def download_worldcuisines_images(rows: list[dict], cache_dir: str, logger: logging.Logger) -> None:
    """Pre-fetch every row's image into a local cache, on the workstation
    node (which has internet), so evaluate.py's GPU job never needs its
    own network access -- Narval's compute nodes have none. Same reason
    `load_cvqa` above saves images locally instead of leaving them for
    evaluate.py to fetch.

    Cache key is `sha1(image_url).hexdigest() + ".jpg"` -- keyed by the
    *original* `image_url` as it appears in the row (matching evaluate.py's
    `load_image_by_url` cache-path scheme exactly, so its
    `os.path.exists(cache_path)` check hits unconditionally and it never
    falls through to its own `requests.get`), even though the outgoing
    HTTP request itself goes to `_wikimedia_original_url`'s de-thumbed
    URL -- the cache key and the fetch URL are deliberately different
    strings. Idempotent and resumable: an already-cached URL is skipped,
    so re-running after a partial failure only retries what's missing.
    Shared across task1 and task2 (same image pool, since both query the
    same set of dish photos) -- call this with the same `cache_dir` for
    both.

    A small delay between requests plus `_fetch_with_retry`'s backoff
    keeps this well under Wikimedia's abuse threshold (see that
    function's docstring) -- at ~300 unique images per language this adds
    a few minutes, not hours.
    """
    os.makedirs(cache_dir, exist_ok=True)
    urls = sorted({r["image_url"] for r in rows if r.get("image_url")})
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "M2-ALIGN-research/1.0 "
            "(https://github.com/MaryamTaj/M2-ALIGN; academic multilingual-VQA research)"
        ),
    })
    n_cached = n_downloaded = n_failed = 0
    for url in urls:
        cache_path = os.path.join(cache_dir, hashlib.sha1(url.encode()).hexdigest() + ".jpg")
        if os.path.exists(cache_path):
            n_cached += 1
            continue
        content = _fetch_with_retry(session, _wikimedia_original_url(url), logger)
        if content is None:
            n_failed += 1
            continue
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
            img.save(cache_path, format="JPEG", quality=85)
            n_downloaded += 1
        except Exception as e:
            n_failed += 1
            logger.warning("WorldCuisines image decode/save failed for %s: %s", url, e)
        time.sleep(0.5)
    logger.info(
        "WorldCuisines images: %d already cached, %d newly downloaded, %d failed (of %d unique URLs) -> %s",
        n_cached, n_downloaded, n_failed, len(urls), cache_dir,
    )
    if n_failed:
        logger.warning(
            "%d image(s) failed to download -- those rows will resolve to no image "
            "(skipped) during evaluation, since the GPU job has no network to retry them.",
            n_failed,
        )


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
            # afaji/cvqa's own topical/geographic tags (e.g. "Food and
            # Drink", "Sports", "People and everyday life"; `Country` is the
            # subset's country, not necessarily the image's location) --
            # used by Stage3/analysis/aggregate_breakdown.py's per-category
            # breakdown.
            "category": r.get("Category"),
            "country": r.get("Country"),
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
    parser.add_argument("--benchmark", required=True,
                        choices=["xgqa", "worldcuisines_task1", "worldcuisines_task2", "cvqa"])
    parser.add_argument("--languages", type=str, required=True,
                        help="Comma-separated ISO codes, e.g. 'bn,id' for xgqa or 'id,jv' for worldcuisines_task1/cvqa.")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Defaults to $SCRATCH/M2-ALIGN/Stage3/data.",
    )
    parser.add_argument("--worldcuisines-split", type=str, default="test_small",
                        choices=["test_small", "test_large"])
    parser.add_argument(
        "--cvqa-image-cache-dir", type=str, default=None,
        help="Where each CVQA row's image is saved, as <id>.jpg. Defaults to "
             "<output_dir>/cvqa/images -- pass evaluate.py's --image-cache-dir the same path.",
    )
    parser.add_argument(
        "--worldcuisines-image-cache-dir", type=str, default=None,
        help="Where WorldCuisines images are pre-downloaded, as sha1(url).jpg. Defaults to "
             "<output_dir>/worldcuisines/images -- must match evaluate.py's --image-cache-dir "
             "exactly (no internet on Narval's compute nodes, so this has to happen here). "
             "Shared by task1 and task2 -- same image pool.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(SCRATCH_ROOT, "data")
    if args.cvqa_image_cache_dir is None:
        args.cvqa_image_cache_dir = os.path.join(args.output_dir, "cvqa", "images")
    if args.worldcuisines_image_cache_dir is None:
        args.worldcuisines_image_cache_dir = os.path.join(args.output_dir, "worldcuisines", "images")

    logger = setup_logging(os.path.join(SCRATCH_ROOT, "logs"))
    langs = [x.strip() for x in args.languages.split(",") if x.strip()]

    for lang in langs:
        if args.benchmark == "xgqa":
            rows = load_xgqa(lang, logger)
            out_path = os.path.join(args.output_dir, "xgqa", f"{lang}.jsonl")
        elif args.benchmark == "worldcuisines_task1":
            rows = load_worldcuisines(lang, args.worldcuisines_split, "task1", logger)
            out_path = os.path.join(args.output_dir, "worldcuisines", "task1", f"{lang}.jsonl")
            download_worldcuisines_images(rows, args.worldcuisines_image_cache_dir, logger)
        elif args.benchmark == "worldcuisines_task2":
            rows = load_worldcuisines(lang, args.worldcuisines_split, "task2", logger)
            out_path = os.path.join(args.output_dir, "worldcuisines", "task2", f"{lang}.jsonl")
            download_worldcuisines_images(rows, args.worldcuisines_image_cache_dir, logger)
        else:
            rows = load_cvqa(lang, args.cvqa_image_cache_dir, logger)
            out_path = os.path.join(args.output_dir, "cvqa", f"{lang}.jsonl")
        write_jsonl(out_path, rows, logger)

    logger.info("Done.")


if __name__ == "__main__":
    main()
