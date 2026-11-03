"""Build Stage 3 data: GQA questions in English and NLLB-translated into each language.

The same N_SAMPLES English questions are translated into every language,
and the answers stay in English. 

It runs in two steps, on two machines:

    python data/build_stage3.py download     # workstation: needs internet
    # copy data/ to Narval (Globus), then submit as a GPU job:
    sbatch --gres=gpu:1 --mem=48G --time=24:00:00 job.sh python data/build_stage3.py translate

download samples GQA into data/stage3/en.jsonl and extracts the images
those questions need from GQA's images.zip into data/images/gqa/.
translate writes data/stage3/{lang}.jsonl:
    {"task": "vqa", "lang": "bn", "source": "<question>", "target": "no",
     "image": "data/images/gqa/2375767.jpg"}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import zipfile
from collections import defaultdict

ROOT = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN")
DATA_DIR = os.path.join(ROOT, "data")
HF_CACHE = os.path.join(os.environ.get("SCRATCH", "."), "huggingface", "hub")
SEED = 42
TRAIN_LANGS = ["bn", "ru", "de", "zh", "pt", "id", "ko", "jv", "mn", "si", "ga"]
NLLB_TAGS = {
    "en": "eng_Latn", "bn": "ben_Beng", "ru": "rus_Cyrl", "de": "deu_Latn", "zh": "zho_Hans",
    "pt": "por_Latn", "id": "ind_Latn", "ko": "kor_Hang", "jv": "jav_Latn", "mn": "khk_Cyrl",
    "si": "sin_Sinh", "ga": "gle_Latn",
}

N_SAMPLES = 30_000
MAX_QUESTIONS_PER_IMAGE = 4  # so a few heavily annotated images do not dominate the sample
GQA_IMAGES_URL = "https://nlp.stanford.edu/data/gqa/images.zip"
GQA_IMAGES_ZIP = os.path.join(DATA_DIR, "raw", "gqa_images.zip")

TRANSLATOR = "facebook/nllb-200-3.3B"
TRANSLATE_BATCH_SIZE = 16
TRANSLATE_BEAMS = 4


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows):>7} rows -> {path}", flush=True)


def sample_gqa() -> list[dict]:
    """N_SAMPLES English rows from GQA's balanced train split, in a fixed random order."""
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/GQA", "train_balanced_instructions", split="train")
    order = list(range(len(ds)))
    random.Random(SEED).shuffle(order)
    per_image = defaultdict(int)
    rows = []
    for i in order:
        r = ds[i]
        # GQA copies on the Hub name the image id differently.
        image_id = str(r.get("imageId") or r.get("image_id") or "")
        question, answer = (r.get("question") or "").strip(), (r.get("answer") or "").strip()
        if not image_id or not question or not answer or per_image[image_id] >= MAX_QUESTIONS_PER_IMAGE:
            continue
        per_image[image_id] += 1
        rows.append({"task": "vqa", "lang": "en", "source": question, "target": answer,
                     "image": f"data/images/gqa/{image_id}.jpg"})
        if len(rows) >= N_SAMPLES:
            break
    return rows


def download_gqa_zip() -> None:
    """Download images.zip, resuming a partial download if one is there."""
    import requests

    if os.path.exists(GQA_IMAGES_ZIP):
        return
    part = GQA_IMAGES_ZIP + ".part"
    os.makedirs(os.path.dirname(part), exist_ok=True)
    have = os.path.getsize(part) if os.path.exists(part) else 0
    with requests.get(GQA_IMAGES_URL, headers={"Range": f"bytes={have}-"}, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(part, "ab" if resp.status_code == 206 else "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    os.replace(part, GQA_IMAGES_ZIP)


def extract_gqa_images(paths: set[str]) -> None:
    """Copy the images at these row paths (data/images/gqa/{id}.jpg) out of images.zip."""
    download_gqa_zip()
    missing = 0
    with zipfile.ZipFile(GQA_IMAGES_ZIP) as archive:
        names = set(archive.namelist())
        for path in sorted(paths):
            target = os.path.join(ROOT, path)
            member = "images/" + os.path.basename(path)
            if os.path.exists(target):
                continue
            if member not in names:
                missing += 1
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with archive.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"  GQA images: {len(paths) - missing}/{len(paths)} present, {missing} not in images.zip")


def download(overwrite: bool) -> None:
    path = os.path.join(DATA_DIR, "stage3", "en.jsonl")
    if os.path.exists(path) and not overwrite:
        print(f"exists, skipped: {path}")
    else:
        write_jsonl(path, sample_gqa())
    extract_gqa_images({r["image"] for r in read_jsonl(path)})


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
    english = read_jsonl(os.path.join(DATA_DIR, "stage3", "en.jsonl"))
    translator = None
    for lang in langs:
        path = os.path.join(DATA_DIR, "stage3", f"{lang}.jsonl")
        if os.path.exists(path) and not overwrite:
            print(f"exists, skipped: {path}")
            continue
        translator = translator or load_translator()
        questions = translate([r["source"] for r in english], NLLB_TAGS[lang], translator)
        write_jsonl(path, [{**r, "lang": lang, "source": q} for r, q in zip(english, questions)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 3 VQA data.")
    parser.add_argument("step", choices=["download", "translate"])
    parser.add_argument("--langs", default=",".join(TRAIN_LANGS), help="translate only.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.step == "download":
        download(args.overwrite)
    else:
        translate_step(args.langs.split(","), args.overwrite)


if __name__ == "__main__":
    main()
