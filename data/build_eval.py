"""Build the evaluation data: xGQA, CVQA, WorldCuisines, MGSM and MSVAMP.

Nothing here is machine-translated. Every question comes from a
human-written or professionally translated source. 

Needs internet, so run it on the workstation, after build_stage3.py
download (the xGQA images come from the same GQA images.zip):

    python data/build_eval.py [--benchmarks xgqa,cvqa] [--overwrite]

Writes data/eval/{benchmark}/{lang}.jsonl, and the images under data/images/:
    {"id": "201307251", "lang": "bn", "question": "...", "image": "data/images/gqa/n161313.jpg", "answer": "no"}
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import time
import zipfile

ROOT = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN")
DATA_DIR = os.path.join(ROOT, "data")
EVAL_DIR = os.path.join(DATA_DIR, "eval")
GQA_IMAGES_ZIP = os.path.join(DATA_DIR, "raw", "gqa_images.zip")

XGQA_LANGS = ["bn", "de", "id", "ko", "pt", "ru", "zh", "en"]
CVQA_LANGS = ["bn", "ga", "id", "jv", "ko", "mn", "pt", "ru", "si", "zh"]
WORLDCUISINES_LANGS = ["bn", "id", "jv", "ko", "ru", "si", "zh"]
MATH_LANGS = ["bn", "de", "ru", "zh", "en"]

XGQA_URL = "https://raw.githubusercontent.com/adapter-hub/xGQA/master/data/zero_shot/testdev_balanced_questions_{lang}.json"
# How CVQA's "Subset" field names each language (matched as a substring, ignoring case).
CVQA_SUBSETS = {
    "bn": "bengali", "ga": "irish", "id": "indonesian", "jv": "javanese", "ko": "korean",
    "mn": "mongolian", "pt": "portuguese", "ru": "russian", "si": "sinhala", "zh": "china",
}
# WorldCuisines' language tags. The formal register is used wherever there is a choice.
WORLDCUISINES_TAGS = {
    "bn": "bn", "id": "id_formal", "jv": "jv_krama", "ko": "ko_formal", "ru": "ru_formal",
    "si": "si_formal_spoken", "zh": "zh_cn",
}
WORLDCUISINES_SPLIT = "test_small"
USER_AGENT = "M2-ALIGN/1.0 (https://github.com/MaryamTaj/M2-ALIGN; academic multilingual VQA research)"
WIKIMEDIA_THUMB = re.compile(r"^(https?://upload\.wikimedia\.org/wikipedia/commons)/thumb/([^/]+)/([^/]+)/([^/]+)/[^/]+$")


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows):>7} rows -> {path}", flush=True)


# xGQA

def load_xgqa(lang: str) -> list[dict]:
    import requests

    resp = requests.get(XGQA_URL.format(lang=lang), timeout=60)
    resp.raise_for_status()
    rows = []
    for qid, entry in resp.json().items():
        if not entry.get("question") or entry.get("answer") is None or not entry.get("imageId"):
            continue
        types = entry.get("types") or {}
        rows.append({
            "id": qid, "lang": lang, "question": entry["question"],
            "image": f"data/images/gqa/{entry['imageId']}.jpg", "answer": str(entry["answer"]),
            "question_type_structural": types.get("structural"), "question_type_semantic": types.get("semantic"),
        })
    extract_gqa_images({r["image"] for r in rows})
    return rows


def extract_gqa_images(paths: set[str]) -> None:
    """Copy the images at these row paths out of images.zip, which build_stage3.py download fetched."""
    if not os.path.exists(GQA_IMAGES_ZIP):
        raise FileNotFoundError(f"{GQA_IMAGES_ZIP} not found. Run data/build_stage3.py download first.")
    with zipfile.ZipFile(GQA_IMAGES_ZIP) as archive:
        for path in sorted(paths):
            target = os.path.join(ROOT, path)
            if not os.path.exists(target):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with archive.open("images/" + os.path.basename(path)) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)


# CVQA

def load_cvqa(langs: list[str]) -> dict[str, list[dict]]:
    """CVQA's test split, one list of rows per language. The images come inside the dataset."""
    from datasets import load_dataset

    ds = load_dataset("afaji/cvqa", split="test")
    rows = {lang: [] for lang in langs}
    for r in ds:
        lang = next((l for l in langs if CVQA_SUBSETS[l] in str(r["Subset"]).lower()), None)
        question, choices, label = r.get("Question"), r.get("Translated Options") or r.get("Options"), r.get("Label")
        if lang is None or not question or not choices or label is None or r.get("image") is None:
            continue
        path = f"data/images/cvqa/{r['ID']}.jpg"
        target = os.path.join(ROOT, path)
        if not os.path.exists(target):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            try:
                r["image"].convert("RGB").save(target, format="JPEG", quality=85)
            except Exception:
                continue
        row = {
            "id": str(r["ID"]), "lang": lang, "question": question, "image": path,
            # The choices are CVQA's English translations, the same for every approach.
            "choices": list(choices), "answer_index": int(label),
            "category": r.get("Category"), "country": r.get("Country"),
        }
        if r.get("Translated Question"):
            row["question_en"] = r["Translated Question"]
        rows[lang].append(row)
    return rows


# WorldCuisines

def load_worldcuisines(task: str, langs: list[str]) -> dict[str, list[dict]]:
    """One list of rows per language. A row is correct if it matches its own-language or its English answer."""
    from datasets import load_dataset

    ds = load_dataset("worldcuisines/vqa", task, split=WORLDCUISINES_SPLIT)
    english_answer = {str(r["qa_id"]): r["answer"] for r in ds if r["lang"] == "en" and r.get("answer")}
    rows = {lang: [] for lang in langs}
    tag_to_lang = {WORLDCUISINES_TAGS[l]: l for l in langs}
    for r in ds:
        lang = tag_to_lang.get(str(r["lang"]))
        if lang is None or not r.get("open_ended_prompt") or r.get("answer") is None or r.get("qa_id") is None:
            continue
        url = r.get("image_url") or r.get("image_path")
        answers = {str(r["answer"])} | ({str(english_answer[str(r["qa_id"])])} if str(r["qa_id"]) in english_answer else set())
        rows[lang].append({
            "id": str(r["qa_id"]), "lang": lang, "question": r["open_ended_prompt"],
            "image": f"data/images/worldcuisines/{hashlib.sha1(url.encode()).hexdigest()}.jpg",
            "answers": sorted(answers), "prompt_type": r.get("prompt_type"), "url": url,
        })
    ok = download_worldcuisines_images({r["url"]: r["image"] for lang_rows in rows.values() for r in lang_rows})
    return {lang: [{k: v for k, v in r.items() if k != "url"} for r in lang_rows if r["url"] in ok]
            for lang, lang_rows in rows.items()}


def download_worldcuisines_images(url_to_path: dict[str, str]) -> set[str]:
    """Download each image one at a time, gently, since these are Wikimedia urls. Returns the urls now cached."""
    import requests
    from PIL import Image

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    ok = set()
    for url, path in url_to_path.items():
        target = os.path.join(ROOT, path)
        if os.path.exists(target):
            ok.add(url)
            continue
        content = fetch(session, original_wikimedia_url(url))
        if content is None:
            continue
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            Image.open(io.BytesIO(content)).convert("RGB").save(target, format="JPEG", quality=85)
            ok.add(url)
        except Exception as exc:
            print(f"  could not save {url}: {exc}")
        time.sleep(0.5)
    print(f"  WorldCuisines images: {len(ok)}/{len(url_to_path)} cached")
    return ok


def original_wikimedia_url(url: str) -> str:
    """Turn a Wikimedia thumbnail url into the original file's url. Thumbnail sizes are often rejected."""
    base, sep, query = url.partition("?")
    m = WIKIMEDIA_THUMB.match(base)
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}/{m.group(4)}{sep}{query}" if m else url


def fetch(session, url: str, tries: int = 5) -> bytes | None:
    """GET url, backing off on rate limits (429) and connection errors. None if it never succeeds."""
    import requests

    delay = 5.0
    for attempt in range(tries):
        try:
            resp = session.get(url, timeout=20)
        except requests.exceptions.RequestException:
            resp = None
        if resp is not None and resp.status_code != 429:
            return resp.content if resp.ok else None
        if attempt < tries - 1:
            retry_after = resp.headers.get("Retry-After", "") if resp is not None else ""
            time.sleep(float(retry_after) if retry_after.isdigit() else delay)
            delay = min(delay * 2, 60.0)
    print(f"  gave up on {url}")
    return None


# MGSM and MSVAMP

def load_math(benchmark: str, lang: str) -> list[dict]:
    from datasets import load_dataset

    if benchmark == "mgsm":
        ds = load_dataset("juletxara/mgsm", lang, split="test")
        pairs = [(r["question"], r.get("answer") or r.get("answer_number")) for r in ds]
    else:
        ds = load_dataset("Mathoctopus/MSVAMP", lang, split="test")
        pairs = [(r.get("m_query") or r.get("query"), r.get("response") or r.get("answer")) for r in ds]
    return [{"id": str(i), "lang": lang, "question": str(q), "answer": str(a).replace(",", "")}
            for i, (q, a) in enumerate(pairs)]


# Steps

def build(benchmark: str, langs: list[str], overwrite: bool) -> None:
    todo = [l for l in langs if overwrite or not os.path.exists(os.path.join(EVAL_DIR, benchmark, f"{l}.jsonl"))]
    for lang in set(langs) - set(todo):
        print(f"exists, skipped: {os.path.join(EVAL_DIR, benchmark, f'{lang}.jsonl')}")
    if not todo:
        return
    if benchmark == "xgqa":
        rows = {lang: load_xgqa(lang) for lang in todo}
    elif benchmark == "cvqa":
        rows = load_cvqa(todo)
    elif benchmark.startswith("worldcuisines_"):
        rows = load_worldcuisines(benchmark.removeprefix("worldcuisines_"), todo)
    else:
        rows = {lang: load_math(benchmark, lang) for lang in todo}
    for lang, lang_rows in rows.items():
        write_jsonl(os.path.join(EVAL_DIR, benchmark, f"{lang}.jsonl"), lang_rows)


BENCHMARK_LANGS = {
    "xgqa": XGQA_LANGS, "cvqa": CVQA_LANGS,
    "worldcuisines_task1": WORLDCUISINES_LANGS, "worldcuisines_task2": WORLDCUISINES_LANGS,
    "mgsm": MATH_LANGS, "msvamp": MATH_LANGS,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the evaluation data.")
    parser.add_argument("--benchmarks", default=",".join(BENCHMARK_LANGS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for benchmark in args.benchmarks.split(","):
        build(benchmark, BENCHMARK_LANGS[benchmark], args.overwrite)


if __name__ == "__main__":
    main()
