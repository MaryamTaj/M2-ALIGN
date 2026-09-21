"""Baseline evaluation of raw Qwen3-VL 8B Instruct on multilingual VQA and text-reasoning benchmarks.

No NLLB encoder, no Mapping layer -- the image and the question (in its
native low-resource language, exactly as fed to Stage 3b's `T` term) go
straight into Qwen3-VL's own chat template. This is the "before" reference
point for the primary hypothesis; mgsm/msvamp (below) give the raw-model
reference point for text reasoning with the same script. Comparing
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
cvqa_generation -- diagnostic variant of cvqa: same eval data and prompt,
                  scored via true generation
                  (`evaluate_cvqa_generation`/`cvqa_generation_correct`,
                  substring match against the gold choice text) instead of
                  `score_choice_loglikelihood`'s choice-ranking -- must
                  match Stage3/evaluate.py's `evaluate_cvqa_generation`
                  exactly. Tests whether part of the cvqa gap vs. Stage3 is
                  a log-likelihood/generation eval-format artifact.
cvqa_mcq       -- closed-ended variant of cvqa: lettered options shown in
                  the prompt, prediction = option letter with the highest
                  next-token logit, plus `letter_mass` -- must match
                  Stage3/evaluate.py's `evaluate_cvqa_mcq` exactly.
mgsm / msvamp  -- text-only math (no image): question in its own language,
                  greedy chain-of-thought (512 new tokens), numeric answer
                  extraction. --eval-data is Stage3/data/{mgsm,msvamp}/<lang>.jsonl
                  (Stage3/load_text_evaluation.py). Prompt, system message,
                  decoding and scoring must match Stage3/evaluate.py (mgsm/msvamp)
                  exactly. (Formerly Baseline/evaluate_text.py.)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import re

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Data/outputs/logs live on $SCRATCH, not in the git checkout.
SCRATCH_ROOT = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN", "Baseline")

_VQA_SYSTEM = "You are a helpful assistant that answers questions about images."
_MATH_SYSTEM = "You are a helpful assistant that solves math problems step by step."

TEXT_BENCHMARKS = ("mgsm", "msvamp")
TEXT_MAX_NEW_TOKENS = 512


def build_open_ended_prompt(question: str) -> str:
    """Must match Stage3/evaluate.py's `build_open_ended_prompt` exactly."""
    return f"Question: {question}\nAnswer with a single word or short phrase, in English."


def build_cvqa_open_ended_prompt(question: str) -> str:
    """Must match Stage3/evaluate.py's `build_cvqa_open_ended_prompt` exactly."""
    return f"Question: {question}"


_OPTION_LETTERS = "ABCDEFGH"


def build_cvqa_mcq_prompt(question: str, choices: list[str]) -> str:
    """Must match Stage3/evaluate.py's `build_cvqa_mcq_prompt` exactly."""
    options = "\n".join(f"{_OPTION_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"Question: {question}\n{options}\n"
        "Answer with the option's letter from the given choices directly."
    )


def option_letter_token_ids(tokenizer, n_choices: int) -> list[int]:
    """Must match Stage3/evaluate.py's `option_letter_token_ids` exactly."""
    ids = []
    for letter in _OPTION_LETTERS[:n_choices]:
        tok = tokenizer(letter, add_special_tokens=False)["input_ids"]
        if len(tok) != 1:
            raise ValueError(f"Option letter {letter!r} tokenizes to {tok}, expected a single token")
        ids.append(tok[0])
    return ids


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

def _pixel_kwargs(visual_pixels: int | None) -> dict:
    """`apply_chat_template` kwargs pinning the image pixel budget.

    None keeps the processor defaults (native resolution, up to 16.7M px).
    An int sets min_pixels = max_pixels, matching Stage3/evaluate.py's
    `--visual-pixels` (default 256*256) so Baseline and Stage3 see the same
    number of visual tokens per image.
    """
    if visual_pixels is None:
        return {}
    return {"processor_kwargs": {"min_pixels": visual_pixels, "max_pixels": visual_pixels}}


@torch.inference_mode()
def generate_answer(
    image: Image.Image,
    prompt: str,
    system_message: str,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    device: torch.device,
    max_new_tokens: int,
    visual_pixels: int | None = None,
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
        return_dict=True, return_tensors="pt", **_pixel_kwargs(visual_pixels),
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
    visual_pixels: int | None = None,
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
        return_dict=True, return_tensors="pt", **_pixel_kwargs(visual_pixels),
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


@torch.inference_mode()
def score_mcq_letters(
    image: Image.Image,
    question: str,
    choices: list[str],
    system_message: str,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    device: torch.device,
    visual_pixels: int | None = None,
) -> tuple[list[float], float]:
    """Next-token logits of the option letters after the options-shown MCQ
    prompt. Must match Stage3/evaluate.py's `score_mcq_letters` (same
    protocol, raw model instead of the mapping pipeline): one forward pass,
    read at the last prompt position.

    Returns:
        ``(letter_logits, letter_mass)``.
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": build_cvqa_mcq_prompt(question, choices)},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", **_pixel_kwargs(visual_pixels),
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    next_logits = model(**inputs).logits[0, -1, :].float()
    letter_ids = torch.tensor(option_letter_token_ids(processor.tokenizer, len(choices)), device=next_logits.device)
    letter_logits = next_logits[letter_ids]
    letter_mass = torch.softmax(next_logits, dim=-1)[letter_ids].sum().item()
    return letter_logits.tolist(), letter_mass


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


def cvqa_generation_correct(pred_text: str, target: str) -> bool:
    """Must match Stage3/evaluate.py's `cvqa_generation_correct` exactly --
    correct if the gold choice text appears anywhere in the generated text,
    case-insensitive."""
    return target.lower() in pred_text.lower()


# ─── Evaluation loops ────────────────────────────────────────────────────────

def _open_results_writer(results_path: str | None):
    """Must match Stage3/evaluate.py's `_open_results_writer` exactly."""
    if results_path is None:
        return contextlib.nullcontext()
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    return open(results_path, "w", encoding="utf-8")


def _write_result(rf, row: dict, **fields) -> None:
    """Must match Stage3/evaluate.py's `_write_result` exactly."""
    if rf is None:
        return
    record = dict(row)
    record.update(fields)
    rf.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_open_ended(
    rows: list[dict], lang: str, image_resolver,
    model, processor, device, max_examples: int | None, logger: logging.Logger,
    results_path: str | None = None, visual_pixels: int | None = None,
) -> float:
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    with _open_results_writer(results_path) as rf:
        for idx, row in enumerate(tqdm(rows, desc=f"open-ended/{lang}")):
            image = image_resolver(row)
            if image is None:
                continue
            prompt = build_open_ended_prompt(row["query"])
            pred = generate_answer(image, prompt, _VQA_SYSTEM, model, processor, device, max_new_tokens=16,
                                   visual_pixels=visual_pixels)
            ok = open_ended_correct(pred, row["answer"])
            correct += int(ok)
            _write_result(rf, row, pred=pred, correct=bool(ok))
            if idx < 5:
                logger.info("idx=%d question=%r pred=%r target=%r ok=%s", idx, row["query"][:80], pred, row["answer"], ok)
    acc = correct / len(rows) * 100 if rows else 0.0
    logger.info("lang=%s accuracy=%.2f%% (n=%d)", lang, acc, len(rows))
    return acc


def evaluate_worldcuisines_open_ended(
    rows: list[dict], lang: str, image_resolver,
    model, processor, device, max_examples: int | None, logger: logging.Logger,
    results_path: str | None = None, visual_pixels: int | None = None,
) -> float:
    """Must match Stage3/evaluate.py's `evaluate_worldcuisines_open_ended`
    exactly (generation path aside -- no NLLB/mapping here): scores against
    `row["answers"]` (dual-reference list) with `worldcuisines_correct`,
    prompted with `build_worldcuisines_prompt`."""
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    with _open_results_writer(results_path) as rf:
        for idx, row in enumerate(tqdm(rows, desc=f"worldcuisines/{lang}")):
            image = image_resolver(row)
            if image is None:
                continue
            prompt = build_worldcuisines_prompt(row["query"])
            pred = generate_answer(image, prompt, _VQA_SYSTEM, model, processor, device, max_new_tokens=16,
                                   visual_pixels=visual_pixels)
            ok = worldcuisines_correct(pred, row["answers"])
            correct += int(ok)
            _write_result(rf, row, pred=pred, correct=bool(ok))
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
    results_path: str | None = None, visual_pixels: int | None = None,
) -> float:
    """CVQA's own open-ended protocol: no options shown in the prompt; score
    each candidate answer choice by log-likelihood and take the argmax."""
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    with _open_results_writer(results_path) as rf:
        for idx, row in enumerate(tqdm(rows, desc=f"cvqa-open-ended/{lang}")):
            image = image_resolver(row)
            if image is None:
                continue
            choices = row["choices"]
            prompt = build_cvqa_open_ended_prompt(row["query"])
            scores = [
                score_choice_loglikelihood(image, prompt, choice, _VQA_SYSTEM, model, processor, device,
                                           visual_pixels=visual_pixels)
                for choice in choices
            ]
            pred_idx = max(range(len(scores)), key=lambda i: scores[i])
            target_idx = int(row["answer_index"])
            ok = pred_idx == target_idx
            correct += int(ok)
            _write_result(
                rf, row, pred=choices[pred_idx], pred_index=pred_idx,
                choice_scores=[round(s, 4) for s in scores], correct=bool(ok),
            )
            if idx < 5:
                logger.info(
                    "idx=%d question=%r choices=%r scores=%r pred=%d target=%d ok=%s",
                    idx, row["query"][:80], choices, [round(s, 3) for s in scores], pred_idx, target_idx, ok,
                )
    acc = correct / len(rows) * 100 if rows else 0.0
    logger.info("lang=%s accuracy=%.2f%% (n=%d)", lang, acc, len(rows))
    return acc


def evaluate_cvqa_generation(
    rows: list[dict], lang: str, image_resolver,
    model, processor, device, max_examples: int | None, logger: logging.Logger,
    results_path: str | None = None, visual_pixels: int | None = None,
) -> float:
    """Must match Stage3/evaluate.py's `evaluate_cvqa_generation` exactly
    (generation path aside -- no NLLB/mapping here): same no-options-shown
    prompt, scored via true generation against the gold choice text with
    `cvqa_generation_correct` instead of log-likelihood ranking."""
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    with _open_results_writer(results_path) as rf:
        for idx, row in enumerate(tqdm(rows, desc=f"cvqa-generation/{lang}")):
            image = image_resolver(row)
            if image is None:
                continue
            choices = row["choices"]
            target_idx = int(row["answer_index"])
            target = choices[target_idx]
            prompt = build_cvqa_open_ended_prompt(row["query"])
            pred = generate_answer(image, prompt, _VQA_SYSTEM, model, processor, device, max_new_tokens=16,
                                   visual_pixels=visual_pixels)
            ok = cvqa_generation_correct(pred, target)
            correct += int(ok)
            _write_result(rf, row, pred=pred, target=target, correct=bool(ok))
            if idx < 5:
                logger.info(
                    "idx=%d question=%r pred=%r target=%r ok=%s",
                    idx, row["query"][:80], pred, target, ok,
                )
    acc = correct / len(rows) * 100 if rows else 0.0
    logger.info("lang=%s accuracy=%.2f%% (n=%d)", lang, acc, len(rows))
    return acc


def log_mcq_summary(
    pred_letters: list[str], gold_letters: list[str], letter_masses: list[float], logger: logging.Logger,
) -> None:
    """Must match Stage3/evaluate.py's `log_mcq_summary` exactly."""
    if not pred_letters:
        return
    letters = sorted(set(gold_letters) | set(pred_letters))
    n = len(pred_letters)
    pred_dist = {l: round(100 * pred_letters.count(l) / n, 1) for l in letters}
    gold_dist = {l: round(100 * gold_letters.count(l) / n, 1) for l in letters}
    logger.info("mcq predicted letter %%: %s | gold letter %%: %s", pred_dist, gold_dist)
    logger.info(
        "mcq letter_mass mean=%.4f min=%.4f (share < 0.5: %.1f%%)",
        sum(letter_masses) / n, min(letter_masses), 100 * sum(m < 0.5 for m in letter_masses) / n,
    )


def evaluate_cvqa_mcq(
    rows: list[dict], lang: str, image_resolver,
    model, processor, device, max_examples: int | None, logger: logging.Logger,
    results_path: str | None = None, visual_pixels: int | None = None,
) -> float:
    """Must match Stage3/evaluate.py's `evaluate_cvqa_mcq` exactly (no
    NLLB/mapping here): lettered options in the prompt, prediction = option
    letter with the highest next-token logit."""
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    pred_letters, gold_letters, letter_masses = [], [], []
    with _open_results_writer(results_path) as rf:
        for idx, row in enumerate(tqdm(rows, desc=f"cvqa-mcq/{lang}")):
            image = image_resolver(row)
            if image is None:
                continue
            choices = row["choices"]
            letter_logits, letter_mass = score_mcq_letters(
                image, row["query"], choices, _VQA_SYSTEM, model, processor, device,
                visual_pixels=visual_pixels,
            )
            pred_idx = max(range(len(letter_logits)), key=lambda i: letter_logits[i])
            target_idx = int(row["answer_index"])
            ok = pred_idx == target_idx
            correct += int(ok)
            pred_letters.append(_OPTION_LETTERS[pred_idx])
            gold_letters.append(_OPTION_LETTERS[target_idx])
            letter_masses.append(letter_mass)
            _write_result(
                rf, row, pred=choices[pred_idx], pred_index=pred_idx, pred_letter=_OPTION_LETTERS[pred_idx],
                letter_logits=[round(s, 4) for s in letter_logits], letter_mass=round(letter_mass, 4),
                correct=bool(ok),
            )
            if idx < 5:
                logger.info(
                    "idx=%d question=%r choices=%r letter_logits=%r letter_mass=%.3f pred=%d target=%d ok=%s",
                    idx, row["query"][:80], choices, [round(s, 3) for s in letter_logits], letter_mass,
                    pred_idx, target_idx, ok,
                )
    log_mcq_summary(pred_letters, gold_letters, letter_masses, logger)
    acc = correct / len(rows) * 100 if rows else 0.0
    logger.info("lang=%s accuracy=%.2f%% (n=%d)", lang, acc, len(rows))
    return acc


# ─── Entry point ─────────────────────────────────────────────────────────────

# ─── Text-only math: mgsm / msvamp (identical to Stage3/evaluate.py mgsm/msvamp) ─

def build_math_prompt(question: str) -> str:
    return f"{question}\n\nLet's think step by step."


def extract_math_answer(text: str) -> str | None:
    m = re.search(r"(?:the\s+)?answer\s+is[:\s]+(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def math_correct(pred_text: str, target: str) -> bool:
    pred = extract_math_answer(pred_text)
    if pred is None:
        return False
    try:
        return float(pred) == float(target.replace(",", ""))
    except ValueError:
        return pred.strip() == target.strip()


@torch.inference_mode()
def generate_text(question: str, model, tokenizer, device, max_new_tokens: int) -> str:
    """Text-only chat generation: system + user(prompt), no image."""
    messages = [
        {"role": "system", "content": _MATH_SYSTEM},
        {"role": "user", "content": build_math_prompt(question)},
    ]
    enc = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]
    generated_ids = model.generate(
        **enc, do_sample=False, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True).strip()


def evaluate_math(
    rows: list[dict], lang: str, task: str,
    model, tokenizer, device, max_examples: int | None, logger: logging.Logger,
    results_path: str | None = None,
) -> float:
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    with _open_results_writer(results_path) as rf:
        for idx, row in enumerate(tqdm(rows, desc=f"{task}/{lang}")):
            pred = generate_text(row["question"], model, tokenizer, device, TEXT_MAX_NEW_TOKENS)
            ok = math_correct(pred, row["answer"])
            correct += int(ok)
            # Same per-item fields as Stage3/evaluate.py mgsm/msvamp --results-jsonl.
            _write_result(rf, {"idx": idx, "question": row["question"], "answer": row["answer"]},
                          pred=pred, extracted=extract_math_answer(pred), correct=ok)
            if idx < 5:
                logger.info("idx=%d question=%r pred=%r extracted=%r target=%s ok=%s",
                            idx, row["question"][:80], pred[:200], extract_math_answer(pred), row["answer"], ok)
    acc = correct / len(rows) * 100 if rows else 0.0
    logger.info("lang=%s accuracy=%.2f%% (n=%d)", lang, acc, len(rows))
    return acc


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
        description="Baseline Qwen3-VL evaluation on multilingual VQA and text benchmarks (no mapping layer)."
    )
    parser.add_argument("--benchmark", required=True,
                         choices=["xgqa", "worldcuisines_task1", "worldcuisines_task2", "cvqa", "cvqa_generation",
                                  "cvqa_mcq", *TEXT_BENCHMARKS])
    parser.add_argument("--lang", required=True,
                         help="ISO code (bn/de/ru/zh/pt/id/ko for xgqa; am/ig/om/pt/ko/mn/si/ga/bn/ru/zh for "
                              "cvqa/cvqa_generation/cvqa_mcq; bn/ru/zh/id/ko/jv/si for worldcuisines_task1/worldcuisines_task2).")
    parser.add_argument("--eval-data", required=True,
                        help="JSONL from Stage3/load_evaluation_data.py (VQA) or "
                             "Stage3/load_text_evaluation.py (mgsm/msvamp).")
    parser.add_argument("--images-dir", default=None, help="Local GQA images dir (required for --benchmark xgqa).")
    parser.add_argument("--image-cache-dir", default=None,
                        help="Image cache dir. Defaults to <Stage3 data>/cvqa/images (cvqa) or "
                             "<Stage3 data>/worldcuisines/images (worldcuisines_task1/_task2), matching "
                             "Stage3/load_evaluation_data.py's --cvqa-image-cache-dir / "
                             "--worldcuisines-image-cache-dir. Both are pre-populated on the workstation "
                             "node before this job runs -- this GPU job has no network access (Narval's "
                             "compute nodes have none), so it only ever reads the local cache.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--visual-pixels", type=int, default=None,
                        help="Fixed per-image pixel budget (min_pixels = max_pixels). Default: processor "
                             "defaults (native resolution). Pass 65536 (256*256) to match Stage3/evaluate.py.")
    parser.add_argument("--query-field", default="query",
                        help="Row field holding the question to ask. Default 'query' (native language). "
                             "'translated_query' = CVQA's human English question (English upper bound, "
                             "added by Baseline/load_english_upper_bound.py); the original 'query' is kept "
                             "as 'query_native' in --results-jsonl.")
    parser.add_argument("--smoke", action="store_true", help="Run 5 examples only.")
    parser.add_argument("--results-jsonl", default=None,
                         help="If given, write one JSON line per example (full row plus pred/correct, "
                              "and pred_index/choice_scores for cvqa) to this path -- must match "
                              "Stage3/evaluate.py's --results-jsonl schema so qualitative_cases.py can "
                              "diff a Baseline run against a Stage3 run. Parent dir is created if missing.")
    args = parser.parse_args()

    if args.benchmark in TEXT_BENCHMARKS and (args.visual_pixels is not None or args.query_field != "query"):
        parser.error(f"--visual-pixels/--query-field do not apply to {args.benchmark}")
    if args.benchmark == "xgqa" and not args.images_dir:
        parser.error("--images-dir is required for --benchmark xgqa")
    if args.image_cache_dir is None:
        stage3_data = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN", "Stage3", "data")
        if args.benchmark in ("cvqa", "cvqa_generation", "cvqa_mcq"):
            args.image_cache_dir = os.path.join(stage3_data, "cvqa", "images")
        elif args.benchmark in ("worldcuisines_task1", "worldcuisines_task2"):
            args.image_cache_dir = os.path.join(stage3_data, "worldcuisines", "images")

    logger = setup_logging(args.benchmark)
    max_examples = 5 if args.smoke else args.max_examples

    rows = _read_jsonl(args.eval_data)
    logger.info("Loaded %d rows from %s", len(rows), args.eval_data)
    if args.query_field != "query":
        missing = [r["id"] for r in rows if not r.get(args.query_field)]
        if missing:
            raise ValueError(f"{len(missing)} rows lack --query-field {args.query_field!r}, e.g. {missing[:3]}")
        for r in rows:
            r["query_native"] = r["query"]
            r["query"] = r[args.query_field]
        logger.info("Question from field %r, e.g. %r -> %r",
                    args.query_field, rows[0]["query_native"], rows[0]["query"])
    logger.info("Model: %s | Benchmark: %s | Lang: %s | visual_pixels: %s",
                args.model_id, args.benchmark, args.lang, args.visual_pixels or "processor default")

    model, processor, device = load_model(args.model_id, local_files_only=args.local_files_only)

    if args.benchmark in TEXT_BENCHMARKS:
        # Plain tokenizer (not processor.tokenizer), exactly as the former
        # Baseline/evaluate_text.py loaded it, so earlier runs stay comparable.
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
        evaluate_math(rows, args.lang, args.benchmark, model, tokenizer, device, max_examples, logger,
                      results_path=args.results_jsonl)
        return

    if args.benchmark == "xgqa":
        resolver = lambda row: load_image_by_id(args.images_dir, row["vg_image_id"])
    elif args.benchmark in ("cvqa", "cvqa_generation", "cvqa_mcq"):
        resolver = lambda row: load_image_by_id(args.image_cache_dir, row["id"])
    else:
        resolver = lambda row: load_image_by_url(row["image_url"], args.image_cache_dir)

    if args.benchmark == "cvqa":
        evaluate_cvqa_open_ended(
            rows, args.lang, resolver, model, processor, device, max_examples, logger,
            results_path=args.results_jsonl, visual_pixels=args.visual_pixels,
        )
    elif args.benchmark == "cvqa_generation":
        evaluate_cvqa_generation(
            rows, args.lang, resolver, model, processor, device, max_examples, logger,
            results_path=args.results_jsonl, visual_pixels=args.visual_pixels,
        )
    elif args.benchmark == "cvqa_mcq":
        evaluate_cvqa_mcq(
            rows, args.lang, resolver, model, processor, device, max_examples, logger,
            results_path=args.results_jsonl, visual_pixels=args.visual_pixels,
        )
    elif args.benchmark in ("worldcuisines_task1", "worldcuisines_task2"):
        evaluate_worldcuisines_open_ended(
            rows, args.lang, resolver, model, processor, device, max_examples, logger,
            results_path=args.results_jsonl, visual_pixels=args.visual_pixels,
        )
    else:
        evaluate_open_ended(
            rows, args.lang, resolver, model, processor, device, max_examples, logger,
            results_path=args.results_jsonl, visual_pixels=args.visual_pixels,
        )


if __name__ == "__main__":
    main()
