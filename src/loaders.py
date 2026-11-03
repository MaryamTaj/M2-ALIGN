"""Training data: read JSONL rows and turn them into batches for MappedVLM.

Every training file has the same row format, for example:

    {"task": "vqa", "lang": "bn", "source": "স্যুপের ডানদিকে রুটি আছে?",
     "target": "no", "image": "Stage3/data/gqa/images/2375767.jpg"}

- task: what the row trains. One of:
    translation  Stage 1 and translation replay. No image, no chat.
    caption      Stage 2. The image goes in its own block, before the mapped caption.
    vqa          Stage 3. The image goes inside the chat, with the question.
    math         Math replay. A chat with no image.
- lang: the ISO 639-1 code of the source text.
- source: the text the encoder reads. 
- target: the English text the backbone should write.
- image: a path under ROOT. 

Also contains training prompts.
"""
from __future__ import annotations

import json
import os
import random
from functools import partial

import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN")
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_LANGS = ["bn", "ru", "de", "zh", "pt", "id", "ko", "jv", "mn", "si", "ga"]
STAGE_LANGS = {1: TRAIN_LANGS, 2: TRAIN_LANGS, 3: TRAIN_LANGS + ["en"]}
VAL_FRACTION = 0.05
MAX_VAL_ROWS = 3000
NUM_WORKERS = 4

# Token limits for the encoder's input and the backbone's target, by task.
# A batch mixing tasks (replay) uses the larger limit.
MAX_SOURCE_LEN = {"translation": 256, "caption": 512, "vqa": 256, "math": 256}
MAX_TARGET_LEN = {"translation": 256, "caption": 512, "vqa": 16, "math": 512}
MAX_VQA_CHAT_LEN = 256
MAX_MATH_QUESTION_LEN = 1024

# Prompts
VQA_SYSTEM = "You are a helpful assistant that answers questions about images."
MATH_SYSTEM = "You are a helpful assistant that solves math problems step by step."


def open_ended_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer with a single word or short phrase, in English."


def math_prompt(question: str) -> str:
    return f"{question}\n\nLet's think step by step."


def vqa_messages(question: str) -> list[dict]:
    """The chat for a VQA question. backbone.render_chat puts the image before the question."""
    return [{"role": "system", "content": VQA_SYSTEM}, {"role": "user", "content": open_ended_prompt(question)}]


def math_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": MATH_SYSTEM}, {"role": "user", "content": math_prompt(question)}]


# Files
def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_image(path: str) -> Image.Image:
    return Image.open(os.path.join(ROOT, path)).convert("RGB")


# Batches
def tokenize_targets(tokenizer, texts: list[str], max_len: int) -> dict[str, torch.Tensor]:
    """Tokenize English targets, end each with EOS, and pad on the right.

    EOS is added by hand. The Qwen tokenizer adds no special tokens to
    plain text, and without EOS the backbone never learns to stop.
    """
    ids = [
        tokenizer(t, add_special_tokens=False, truncation=True, max_length=max_len - 1)["input_ids"]
        + [tokenizer.eos_token_id]
        for t in texts
    ]
    longest = max(len(x) for x in ids)
    return {
        "input_ids": torch.tensor([x + [tokenizer.pad_token_id] * (longest - len(x)) for x in ids]),
        "attention_mask": torch.tensor([[1] * len(x) + [0] * (longest - len(x)) for x in ids]),
    }


def render_row_chat(row: dict, backbone) -> dict:
    """The chat part of one row. Translation rows get an empty chat, so they can share a batch with math rows."""
    if row["task"] == "vqa":
        return backbone.render_chat(vqa_messages(row["source"]), load_image(row["image"]), max_len=MAX_VQA_CHAT_LEN)
    if row["task"] == "math":
        ids = backbone.tokenizer(row["source"], add_special_tokens=False)["input_ids"]
        question = backbone.tokenizer.decode(ids[:MAX_MATH_QUESTION_LEN])
        return backbone.render_chat(math_messages(question))
    empty = torch.zeros(1, 0, dtype=torch.long)
    return {"input_ids": empty, "attention_mask": empty}


def collate(rows: list[dict], model) -> dict:
    """Turn rows into the batch MappedVLM reads: source, target, and image or chat when the task needs one.
    """
    tasks = {r["task"] for r in rows}
    if len(tasks) > 1 and not tasks <= {"translation", "math"}:
        raise ValueError(f"These tasks cannot share a batch: {sorted(tasks)}")
    encoder, backbone = model.encoder, model.backbone

    batch = {
        "source": encoder.tokenize(
            [r["source"] for r in rows], [r["lang"] for r in rows], max(MAX_SOURCE_LEN[t] for t in tasks),
        ),
        "target": tokenize_targets(
            backbone.tokenizer, [r["target"] for r in rows], max(MAX_TARGET_LEN[t] for t in tasks),
        ),
    }
    if "caption" in tasks:
        batch["image"] = backbone.process_images([load_image(r["image"]) for r in rows])
    if tasks & {"vqa", "math"}:
        batch["chat"] = backbone.batch_chats([render_row_chat(r, backbone) for r in rows])
    return batch


# Loaders
def stage_rows(stage: int) -> list[dict]:
    """All rows for one training stage, from data/stage{N}/{lang}.jsonl."""
    rows = []
    for lang in STAGE_LANGS[stage]:
        rows.extend(read_jsonl(os.path.join(DATA_DIR, f"stage{stage}", f"{lang}.jsonl")))
    return rows


def replay_rows(translation_replay: bool) -> list[dict]:
    """Math replay in every trained language plus English. Translation replay only when asked.

    launch.py turns translation replay off when Stage 1 is not in the run.
    """
    rows = []
    for lang in TRAIN_LANGS + ["en"]:
        rows.extend(read_jsonl(os.path.join(DATA_DIR, "replay", f"math_{lang}.jsonl")))
    if translation_replay:
        for lang in TRAIN_LANGS:
            rows.extend(read_jsonl(os.path.join(DATA_DIR, "replay", f"translation_{lang}.jsonl")))
    return rows


def make_loader(rows: list[dict], model, batch_size: int, shuffle: bool, drop_last: bool = False) -> DataLoader:
    # A plain list works as a dataset. Images load inside collate, in the worker processes.
    return DataLoader(
        rows, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        collate_fn=partial(collate, model=model), num_workers=NUM_WORKERS,
    )


def make_train_loaders(
    stage: int, model, batch_size: int, seed: int, rank: int = 0, world_size: int = 1,
) -> tuple[DataLoader, DataLoader]:
    """The train and validation loaders for one stage. The split depends only on the seed.

    With several GPUs, each one gets its own slice of the rows. Every
    training slice has the same length, so all GPUs run the same number
    of batches and stay in step.
    """
    rows = stage_rows(stage)
    random.Random(seed).shuffle(rows)
    n_val = min(int(len(rows) * VAL_FRACTION), MAX_VAL_ROWS)
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    per_gpu = len(train_rows) // world_size
    return (
        make_loader(train_rows[rank * per_gpu:(rank + 1) * per_gpu], model, batch_size, shuffle=True),
        make_loader(val_rows[rank::world_size], model, batch_size, shuffle=False),
    )


def make_replay_loader(model, translation_replay: bool, batch_size: int) -> DataLoader:
    """Replay batches, shuffled. train.py restarts this loader whenever it runs out."""
    return make_loader(replay_rows(translation_replay), model, batch_size, shuffle=True, drop_last=True)
