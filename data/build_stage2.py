"""Build Stage 2 data: images with a caption in each language, plus the English caption.

Each language gets rows from two sources:
- WIT: Wikipedia images captioned both in the language and in English.
  Up to WIT_PER_LANGUAGE pairs per language. 
- CC3M: CC3M_SAMPLES English-captioned web images, the same ones for every
  language. NLLB-3.3B translates the captions into each language.

It runs in two steps, on two machines:

    python data/build_stage2.py download     # workstation: needs internet
    # copy data/ to Narval (Globus), then submit as a GPU job:
    sbatch --gres=gpu:1 --mem=48G --time=24:00:00 job.sh python data/build_stage2.py translate

download writes data/raw/stage2/wit/{lang}.jsonl (finished WIT rows),
data/raw/stage2/cc3m_english.jsonl, and the images under data/images/.
translate writes data/stage2/{lang}.jsonl:
    {"task": "caption", "lang": "bn", "source": "<caption>", "target": "<English caption>",
     "image": "data/images/wit/<sha1 of url>.jpg"}
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN")
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "stage2")
HF_CACHE = os.path.join(os.environ.get("SCRATCH", "."), "huggingface", "hub")
SEED = 42
TRAIN_LANGS = ["bn", "ru", "de", "zh", "pt", "id", "ko", "jv", "mn", "si", "ga"]
NLLB_TAGS = {
    "en": "eng_Latn", "bn": "ben_Beng", "ru": "rus_Cyrl", "de": "deu_Latn", "zh": "zho_Hans",
    "pt": "por_Latn", "id": "ind_Latn", "ko": "kor_Hang", "jv": "jav_Latn", "mn": "khk_Cyrl",
    "si": "sin_Sinh", "ga": "gle_Latn",
}

WIT_PER_LANGUAGE = 1236
CC3M_SAMPLES = 10_000
CC3M_BATCH = 20_000  

TRANSLATOR = "facebook/nllb-200-3.3B"
TRANSLATE_BATCH_SIZE = 32
TRANSLATE_BEAMS = 4

DOWNLOAD_WORKERS = 4  # Wikimedia rate-limits bigger bursts
# Wikimedia blocks clients that do not name themselves (meta.wikimedia.org/wiki/User-Agent_policy).
USER_AGENT = "M2-ALIGN/1.0 (https://github.com/MaryamTaj/M2-ALIGN)"
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_TRIES = 5
DOWNLOAD_DEADLINE_S = 20  # a hard cap per try: requests' timeout misses servers that send bytes slowly
MAX_ASPECT_RATIO = 200  # Qwen3-VL's processor rejects images narrower than this


# Files

def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows):>7} rows -> {path}", flush=True)


# Images

def image_path(source: str, url: str) -> str:
    """Where an image is cached, relative to ROOT. Rows store this path."""
    return f"data/images/{source}/{hashlib.sha1(url.encode()).hexdigest()}.jpg"


def download_image(url: str, path: str) -> str | None:
    """Fetch url into path, unless it is there already. Returns None on success, else the reason it failed."""
    import requests
    from PIL import Image

    full_path = os.path.join(ROOT, path)
    if os.path.exists(full_path):
        return None
    for attempt in range(MAX_TRIES):
        try:
            deadline = time.monotonic() + DOWNLOAD_DEADLINE_S
            with requests.get(url, timeout=15, headers={"User-Agent": USER_AGENT}, stream=True) as resp:
                resp.raise_for_status()
                chunks = []
                for chunk in resp.iter_content(chunk_size=65536):
                    if time.monotonic() > deadline:
                        raise TimeoutError("download took too long")
                    chunks.append(chunk)
            image = Image.open(io.BytesIO(b"".join(chunks))).convert("RGB")
            if max(image.size) / min(image.size) > MAX_ASPECT_RATIO:
                return "aspect ratio"
            image.save(full_path + ".tmp", format="JPEG", quality=85)
            os.replace(full_path + ".tmp", full_path)
            return None
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code
            if status not in RETRY_STATUSES or attempt == MAX_TRIES - 1:
                return f"HTTP {status}"
            retry_after = exc.response.headers.get("Retry-After", "")
            time.sleep(float(retry_after) if retry_after.isdigit() else min(30, 2 ** attempt))
        except Exception as exc:
            return type(exc).__name__
    return "too many retries"


def download_images(urls: list[str], source: str) -> set[str]:
    """Download every url into data/images/{source}/. Returns the urls that are now cached.

    Urls that failed before are listed in .failed_urls.txt and skipped, so
    a rerun does not wait on the same dead links again.
    """
    folder = os.path.join(DATA_DIR, "images", source)
    os.makedirs(folder, exist_ok=True)
    failed_list = os.path.join(folder, ".failed_urls.txt")
    known_bad = set(read_lines(failed_list))
    todo = [u for u in dict.fromkeys(urls) if u not in known_bad]
    with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
        reasons = list(pool.map(lambda u: download_image(u, image_path(source, u)), todo))
    new_bad = [u for u, reason in zip(todo, reasons) if reason]
    with open(failed_list, "a", encoding="utf-8") as f:
        f.writelines(u + "\n" for u in new_bad)
    print(f"  {source} images: {len(todo) - len(new_bad)}/{len(todo)} cached", flush=True)
    return {u for u, reason in zip(todo, reasons) if reason is None}


def read_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# WIT
def stream_wit():
    """Rows of wikimedia/wit_base, reconnecting from the last row read if the connection drops."""
    from datasets import load_dataset

    def open_stream(state):
        # Leave out the image bytes and embeddings: most of each row's size, and never used here.
        ds = load_dataset("wikimedia/wit_base", split="train", streaming=True)
        ds = ds.select_columns(["image_url", "wit_features"])
        if state:
            ds.load_state_dict(state)
        return ds, iter(ds)

    ds, rows = open_stream(None)
    state, failures = None, 0
    while True:
        try:
            row = next(rows)
        except StopIteration:
            return
        except Exception as exc:
            failures += 1
            print(f"  stream error ({exc}); reconnecting in {min(60, 2 ** failures)}s", flush=True)
            time.sleep(min(60, 2 ** failures))
            ds, rows = open_stream(state)
            continue
        failures = 0
        state = ds.state_dict()
        yield row


def wit_captions(row: dict) -> dict[str, str]:
    """{language: caption} for one WIT image, from the first entry per language.

    Only caption_reference_description is used. The attribution text is
    shared by all languages and is often left in English.
    """
    features = row.get("wit_features") or {}
    captions = {}
    for i, lang in enumerate(features.get("language") or []):
        if lang not in captions:
            captions[lang] = (features["caption_reference_description"][i] or "").strip()
    return {lang: text for lang, text in captions.items() if text}


def collect_wit(langs: list[str]) -> dict[str, list[dict]]:
    """One pass over WIT, collecting up to WIT_PER_LANGUAGE (url, caption, English caption) per language."""
    found = {lang: [] for lang in langs}
    for scanned, row in enumerate(stream_wit(), start=1):
        if all(len(found[lang]) >= WIT_PER_LANGUAGE for lang in langs):
            break
        if scanned % 200_000 == 0:
            print(f"  WIT: {scanned:,} rows scanned | " + " ".join(f"{k}={len(v)}" for k, v in found.items()), flush=True)
        captions = wit_captions(row)
        if not row.get("image_url") or "en" not in captions:
            continue
        for lang in langs:
            if lang in captions and len(found[lang]) < WIT_PER_LANGUAGE:
                found[lang].append({"url": row["image_url"], "source": captions[lang], "target": captions["en"]})
    return found


# CC3M
def cc3m_candidates():
    """CC3M (url, English caption) pairs in a fixed random order, without repeated urls."""
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/conceptual_captions", "unlabeled", split="train")
    order = list(range(len(ds)))
    random.Random(SEED).shuffle(order)
    seen = set()
    for i in order:
        url, caption = (ds[i]["image_url"] or "").strip(), (ds[i]["caption"] or "").strip()
        if url and caption and url not in seen:
            seen.add(url)
            yield {"url": url, "caption": caption}


def sample_cc3m() -> list[dict]:
    """CC3M_SAMPLES pairs whose image downloaded. Many CC3M links are dead, so this tries candidates in rounds."""
    candidates, kept = cc3m_candidates(), []
    while len(kept) < CC3M_SAMPLES:
        batch = list(itertools.islice(candidates, CC3M_BATCH))
        if not batch:
            print(f"  CC3M ran out of candidates at {len(kept)} images")
            break
        ok = download_images([c["url"] for c in batch], "cc3m")
        kept += [c for c in batch if c["url"] in ok]
    return [{"image": image_path("cc3m", c["url"]), "caption": c["caption"]} for c in kept[:CC3M_SAMPLES]]


# Steps
def download(langs: list[str], overwrite: bool) -> None:
    todo = [lang for lang in langs if overwrite or not os.path.exists(os.path.join(RAW_DIR, "wit", f"{lang}.jsonl"))]
    if todo:
        for lang, pairs in collect_wit(todo).items():
            ok = download_images([p["url"] for p in pairs], "wit")
            rows = [
                {"task": "caption", "lang": lang, "source": p["source"], "target": p["target"],
                 "image": image_path("wit", p["url"])}
                for p in pairs if p["url"] in ok
            ]
            write_jsonl(os.path.join(RAW_DIR, "wit", f"{lang}.jsonl"), rows)
    cc3m_path = os.path.join(RAW_DIR, "cc3m_english.jsonl")
    if overwrite or not os.path.exists(cc3m_path):
        write_jsonl(cc3m_path, sample_cc3m())


def load_translator():
    import torch
    from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

    tokenizer = NllbTokenizer.from_pretrained(TRANSLATOR, cache_dir=HF_CACHE, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR, cache_dir=HF_CACHE, dtype=torch.float16)
    return tokenizer, model.to("cuda").eval()


def translate(texts: list[str], tgt_tag: str, translator) -> list[str]:
    """Translate English texts into tgt_tag, in order."""
    import torch

    tokenizer, model = translator
    out = []
    with torch.inference_mode():
        for start in range(0, len(texts), TRANSLATE_BATCH_SIZE):
            enc = tokenizer(texts[start:start + TRANSLATE_BATCH_SIZE], return_tensors="pt", truncation=True,
                            max_length=256, padding=True).to("cuda")
            ids = model.generate(**enc, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_tag),
                                 max_new_tokens=256, num_beams=TRANSLATE_BEAMS)
            out += tokenizer.batch_decode(ids, skip_special_tokens=True)
    return out


def translate_step(langs: list[str], overwrite: bool) -> None:
    cc3m = read_jsonl(os.path.join(RAW_DIR, "cc3m_english.jsonl"))
    translator = None
    for lang in langs:
        path = os.path.join(DATA_DIR, "stage2", f"{lang}.jsonl")
        if os.path.exists(path) and not overwrite:
            print(f"exists, skipped: {path}")
            continue
        translator = translator or load_translator()
        captions = translate([c["caption"] for c in cc3m], NLLB_TAGS[lang], translator)
        cc3m_rows = [
            {"task": "caption", "lang": lang, "source": caption, "target": c["caption"], "image": c["image"]}
            for c, caption in zip(cc3m, captions)
        ]
        write_jsonl(path, read_jsonl(os.path.join(RAW_DIR, "wit", f"{lang}.jsonl")) + cc3m_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 2 caption data.")
    parser.add_argument("step", choices=["download", "translate"])
    parser.add_argument("--langs", default=",".join(TRAIN_LANGS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    langs = args.langs.split(",")
    if args.step == "download":
        download(langs, args.overwrite)
    else:
        translate_step(langs, args.overwrite)


if __name__ == "__main__":
    main()
