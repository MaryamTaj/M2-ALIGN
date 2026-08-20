"""Baseline evaluation of raw Qwen3-VL 8B Instruct on multilingual VQA benchmarks.

No NLLB encoder, no Mapping layer -- the image and the question (in its
native low-resource language, exactly as fed to Stage 3b's `T` term) go
straight into Qwen3-VL's own chat template. This is the "before" reference
point for the primary hypothesis, parallel to how `Baseline/evaluate_text.py`
already gives the raw-model reference point for text reasoning. Comparing
this against Stage 2 (vision-mapping only, zero-shot) and the final
Stage 3b checkpoint answers: does the mapping help at all beyond what
Qwen3-VL can already do zero-shot in these languages?

Supported benchmarks (same real/non-synthetic sources as Stage3/evaluate.py)
--------------------------------------------------------------------------------
Policy: wherever a language has coverage in more than one benchmark below,
run all of them -- they're independent evaluations, not interchangeable.

xgqa           -- open-ended, English short answers. Active languages:
                  bn/de/ru/zh/pt/id/ko.
worldcuisines_task1/_task2 -- open-ended. Two independent benchmark rows,
                  both active for bn/ru/zh/id/ko/jv/si (no pt/de/mn/ga
                  coverage):
                    worldcuisines_task1 -- task1, Dish Name Prediction.
                    worldcuisines_task2 -- task2, Dish Origin Prediction.
                  Must match Stage3/evaluate.py's
                  `evaluate_worldcuisines_open_ended`/`worldcuisines_correct`/
                  `build_worldcuisines_prompt` exactly: dual-reference
                  (native + English answer) substring-containment scoring,
                  with the "single word/short phrase, in English"
                  instruction kept -- see that module's docstring for why.
cvqa           -- multiple-choice dataset, scored **open-ended via
                  answer-choice log-likelihood** -- must match
                  Stage3/evaluate.py's `evaluate_cvqa_open_ended`
                  exactly so Baseline/Stage2/Stage3 are comparable. Active
                  languages: am/ig/om/mn/si/ga, plus bn/ru/zh/pt/id/ko
                  (also run through xGQA, per the policy note). No German
                  coverage -- CVQA has no German subset at all.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Data/outputs/logs live on $SCRATCH, not in the git checkout.
SCRATCH_ROOT = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN", "Baseline")

_VQA_SYSTEM = "You are a helpful assistant that answers questions about images."


def build_open_ended_prompt(question: str) -> str:
    """Must match Stage3/evaluate.py's `build_open_ended_prompt` exactly."""
    return f"Question: {question}\nAnswer with a single word or short phrase, in English."


def build_cvqa_open_ended_prompt(question: str) -> str:
    """Must match Stage3/evaluate.py's `build_cvqa_open_ended_prompt` exactly."""
    return f"Question: {question}"


def build_worldcuisines_prompt(question: str) -> str:
    """Must match Stage3/evaluate.py's `build_worldcuisines_prompt` exactly --
    keeps the "single word/short phrase, in English" instruction, see that
    function's docstring."""
    return f"Question: {question}\nAnswer with a single word or short phrase, in English."


# ─── Logging ────────────────────────────────────────────────────────────────

def setup_logging(benchmark: str) -> logging.Logger:
    """Create a logger that writes to stdout.

    The job script's SLURM ``--output`` file is the single log file for a
    run, so this only needs to format stdout consistently -- it must not
    also write its own file, or every run ends up with two logs.
    """
    logger = logging.getLogger(f"baseline_eval_vqa_{benchmark}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model(
    model_id: str,
    local_files_only: bool,
) -> tuple[Qwen3VLForConditionalGeneration, AutoProcessor, torch.device]:
    assert torch.cuda.is_available(), "CUDA not available - request a GPU node."
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto",
        low_cpu_mem_usage=True, local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model.eval()
    return model, processor, device


# ─── Image resolution (same conventions as Stage3/evaluate.py) ─────────

def load_image_by_id(images_dir: str, image_id: str) -> Image.Image | None:
    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(images_dir, f"{image_id}{ext}")
        if os.path.exists(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                return None
    return None


def load_image_by_url(url: str, cache_dir: str) -> Image.Image | None:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, hashlib.sha1(url.encode()).hexdigest() + ".jpg")
    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            pass
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "M2-ALIGN/1.0"})
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(cache_path, format="JPEG", quality=85)
        return img
    except Exception:
        return None


# ─── Generation ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_answer(
    image: Image.Image,
    prompt: str,
    system_message: str,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = model.generate(
        **inputs, do_sample=False, max_new_tokens=max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id, eos_token_id=processor.tokenizer.eos_token_id,
    )
    gen_only = generated_ids[:, prompt_len:]
    return processor.tokenizer.decode(gen_only[0], skip_special_tokens=True).strip()


@torch.inference_mode()
def score_choice_loglikelihood(
    image: Image.Image,
    prompt: str,
    choice_text: str,
    system_message: str,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    device: torch.device,
) -> float:
    """Length-normalized log-likelihood of *choice_text* as a continuation of
    *prompt*. Must match Stage3/evaluate.py's `score_choice_loglikelihood`
    (same protocol, applied to the raw model instead of the custom mapping
    pipeline): one forward pass per candidate choice, no generation."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    choice_ids = processor.tokenizer(choice_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    n_choice_tok = choice_ids.shape[1]
    if n_choice_tok == 0:
        return float("-inf")

    full_ids = torch.cat([inputs["input_ids"], choice_ids], dim=1)
    full_mask = torch.cat([inputs["attention_mask"], torch.ones_like(choice_ids)], dim=1)
    model_kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    if "mm_token_type_ids" in model_kwargs:
        # Appended choice tokens are plain text (type 0) -- extend to match
        # full_ids/full_mask's length or get_rope_index's per-token indexing
        # (input_token_type[attention_mask[...].bool()]) shape-mismatches.
        model_kwargs["mm_token_type_ids"] = torch.cat(
            [model_kwargs["mm_token_type_ids"], torch.zeros_like(choice_ids)], dim=1,
        )
    logits = model(input_ids=full_ids, attention_mask=full_mask, **model_kwargs).logits

    # Position i's logits predict token i+1, so the logit that predicts
    # choice token k sits at (prompt_len - 1 + k).
    choice_logits = logits[0, prompt_len - 1 : prompt_len - 1 + n_choice_tok, :].float()
    log_probs = torch.log_softmax(choice_logits, dim=-1)
    token_logprobs = log_probs.gather(1, choice_ids[0].unsqueeze(-1)).squeeze(-1)
    return (token_logprobs.sum() / n_choice_tok).item()


# ─── Scoring (identical to Stage3/evaluate.py) ─────────────────────────

def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def open_ended_correct(pred_text: str, target: str) -> bool:
    return normalize_answer(pred_text) == normalize_answer(target)


def worldcuisines_correct(pred_text: str, answers: list[str]) -> bool:
    """Must match Stage3/evaluate.py's `worldcuisines_correct` exactly --
    case-insensitive substring containment against any dual-reference
    answer, not exact match."""
    pred_lower = pred_text.lower()
    return any(ans.lower() in pred_lower for ans in answers if ans)


# ─── Evaluation loops ────────────────────────────────────────────────────────

def evaluate_open_ended(
    rows: list[dict], lang: str, image_resolver,
    model, processor, device, max_examples: int | None, logger: logging.Logger,
) -> float:
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    for idx, row in enumerate(tqdm(rows, desc=f"open-ended/{lang}")):
        image = image_resolver(row)
        if image is None:
            continue
        prompt = build_open_ended_prompt(row["query"])
        pred = generate_answer(image, prompt, _VQA_SYSTEM, model, processor, device, max_new_tokens=16)
        ok = open_ended_correct(pred, row["answer"])
        correct += int(ok)
        if idx < 5:
            logger.info("idx=%d question=%r pred=%r target=%r ok=%s", idx, row["query"][:80], pred, row["answer"], ok)
    acc = correct / len(rows) * 100 if rows else 0.0
    logger.info("lang=%s accuracy=%.2f%% (n=%d)", lang, acc, len(rows))
    return acc


def evaluate_worldcuisines_open_ended(
    rows: list[dict], lang: str, image_resolver,
    model, processor, device, max_examples: int | None, logger: logging.Logger,
) -> float:
    """Must match Stage3/evaluate.py's `evaluate_worldcuisines_open_ended`
    exactly (generation path aside -- no NLLB/mapping here): scores against
    `row["answers"]` (dual-reference list) with `worldcuisines_correct`,
    prompted with `build_worldcuisines_prompt`."""
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    for idx, row in enumerate(tqdm(rows, desc=f"worldcuisines/{lang}")):
        image = image_resolver(row)
        if image is None:
            continue
        prompt = build_worldcuisines_prompt(row["query"])
        pred = generate_answer(image, prompt, _VQA_SYSTEM, model, processor, device, max_new_tokens=16)
        ok = worldcuisines_correct(pred, row["answers"])
        correct += int(ok)
        if idx < 5:
            logger.info(
                "idx=%d question=%r pred=%r targets=%r ok=%s",
                idx, row["query"][:80], pred, row["answers"], ok,
            )
    acc = correct / len(rows) * 100 if rows else 0.0
    logger.info("lang=%s accuracy=%.2f%% (n=%d)", lang, acc, len(rows))
    return acc


def evaluate_cvqa_open_ended(
    rows: list[dict], lang: str, image_resolver,
    model, processor, device, max_examples: int | None, logger: logging.Logger,
) -> float:
    """CVQA's own open-ended protocol: no options shown in the prompt; score
    each candidate answer choice by log-likelihood and take the argmax."""
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    for idx, row in enumerate(tqdm(rows, desc=f"cvqa-open-ended/{lang}")):
        image = image_resolver(row)
        if image is None:
            continue
        choices = row["choices"]
        prompt = build_cvqa_open_ended_prompt(row["query"])
        scores = [
            score_choice_loglikelihood(image, prompt, choice, _VQA_SYSTEM, model, processor, device)
            for choice in choices
        ]
        pred_idx = max(range(len(scores)), key=lambda i: scores[i])
        target_idx = int(row["answer_index"])
        ok = pred_idx == target_idx
        correct += int(ok)
        if idx < 5:
            logger.info(
                "idx=%d question=%r choices=%r scores=%r pred=%d target=%d ok=%s",
                idx, row["query"][:80], choices, [round(s, 3) for s in scores], pred_idx, target_idx, ok,
            )
    acc = correct / len(rows) * 100 if rows else 0.0
    logger.info("lang=%s accuracy=%.2f%% (n=%d)", lang, acc, len(rows))
    return acc


# ─── Entry point ─────────────────────────────────────────────────────────────

def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline Qwen3-VL evaluation on multilingual VQA benchmarks (no mapping layer)."
    )
    parser.add_argument("--benchmark", required=True,
                         choices=["xgqa", "worldcuisines_task1", "worldcuisines_task2", "cvqa"])
    parser.add_argument("--lang", required=True,
                         help="ISO code (bn/de/ru/zh/pt/id/ko for xgqa; am/ig/om/pt/ko/mn/si/ga/bn/ru/zh for cvqa; "
                              "bn/ru/zh/id/ko/jv/si for worldcuisines_task1/worldcuisines_task2).")
    parser.add_argument("--eval-data", required=True, help="JSONL from Stage3/load_evaluation_data.py.")
    parser.add_argument("--images-dir", default=None, help="Local GQA images dir (required for --benchmark xgqa).")
    parser.add_argument("--image-cache-dir",
                        default=os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN", "Stage3", "data", "cvqa", "images"),
                        help="Image cache dir. For cvqa: pre-populated by Stage3/load_evaluation_data.py's "
                             "--cvqa-image-cache-dir, looked up by row id (no network access needed). "
                             "For worldcuisines_task1/_task2: a URL download cache, populated on the fly.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run 5 examples only.")
    args = parser.parse_args()

    if args.benchmark == "xgqa" and not args.images_dir:
        parser.error("--images-dir is required for --benchmark xgqa")

    logger = setup_logging(args.benchmark)
    max_examples = 5 if args.smoke else args.max_examples

    rows = _read_jsonl(args.eval_data)
    logger.info("Loaded %d rows from %s", len(rows), args.eval_data)
    logger.info("Model: %s | Benchmark: %s | Lang: %s", args.model_id, args.benchmark, args.lang)

    model, processor, device = load_model(args.model_id, local_files_only=args.local_files_only)

    if args.benchmark == "xgqa":
        resolver = lambda row: load_image_by_id(args.images_dir, row["vg_image_id"])
    elif args.benchmark == "cvqa":
        resolver = lambda row: load_image_by_id(args.image_cache_dir, row["id"])
    else:
        resolver = lambda row: load_image_by_url(row["image_url"], args.image_cache_dir)

    if args.benchmark == "cvqa":
        evaluate_cvqa_open_ended(rows, args.lang, resolver, model, processor, device, max_examples, logger)
    elif args.benchmark in ("worldcuisines_task1", "worldcuisines_task2"):
        evaluate_worldcuisines_open_ended(rows, args.lang, resolver, model, processor, device, max_examples, logger)
    else:
        evaluate_open_ended(rows, args.lang, resolver, model, processor, device, max_examples, logger)


if __name__ == "__main__":
    main()
