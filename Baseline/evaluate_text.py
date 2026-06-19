"""Baseline evaluation of Qwen3-VL 8B Instruct on text-only multilingual benchmarks.

Supported tasks
---------------
mgsm     – Multilingual Grade School Math    (juletxara/mgsm)
msvamp   – Multilingual SVAMP               (Mathoctopus/MSVAMP)
x-csqa   – Cross-lingual Commonsense QA     (INK-USC/xcsr)
xnli     – Cross-lingual NLI                (xnli)

"""
from __future__ import annotations

import argparse
import logging
import os
import re
from datetime import datetime

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

TASK_DEFAULT_LANGS: dict[str, list[str]] = {
    "mgsm":   ["en"],
    "msvamp": ["en"],
    "x-csqa": ["en"],
    "xnli":   ["en"],
}

TASK_MAX_NEW_TOKENS: dict[str, int] = {
    "mgsm": 512,
    "msvamp": 512,
    "x-csqa": 20,
    "xnli": 20,
}

_MATH_SYSTEM  = "You are a helpful assistant that solves math problems step by step."
_XCSQA_SYSTEM = "You are a helpful assistant that answers commonsense questions."
_XNLI_SYSTEM  = "You are a helpful assistant that determines textual entailment."

_XNLI_INT_TO_STR = {0: "entailment", 1: "neutral", 2: "contradiction"}
_NLI_LABEL_STRINGS = ["entailment", "neutral", "contradiction"]


# ─── Logging ────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str, task: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"baseline_eval_{task}_{timestamp}.log")
    logger = logging.getLogger(f"baseline_eval_{task}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger.addHandler(logging.FileHandler(log_path, encoding="utf-8"))
    logger.addHandler(logging.StreamHandler())
    for h in logger.handlers:
        h.setFormatter(fmt)
    return logger


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model(
    model_id: str,
    local_files_only: bool,
) -> tuple[Qwen3VLForConditionalGeneration, AutoTokenizer, torch.device]:
    assert torch.cuda.is_available(), "CUDA not available – request a GPU node."
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto",
        low_cpu_mem_usage=True, local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model.eval()
    return model, tokenizer, device


# ─── Generation ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_text(
    prompt: str,
    model: Qwen3VLForConditionalGeneration,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_new_tokens: int,
    system_message: str | None = None,
) -> str:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    enc = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]
    generated_ids = model.generate(
        **enc, do_sample=False, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    gen_only = generated_ids[:, prompt_len:]
    return tokenizer.decode(gen_only[0], skip_special_tokens=True).strip()


# ─── Dataset loaders ────────────────────────────────────────────────────────

def _hf_load(hf_id: str, config: str, split: str) -> list[dict]:
    try:
        ds = load_dataset(hf_id, config, split=split, download_mode="reuse_dataset_if_exists")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load '{hf_id}' config='{config}' split='{split}': {exc}"
        ) from exc
    return list(ds)


def load_mgsm(lang: str) -> list[dict]:
    rows = _hf_load("juletxara/mgsm", lang, "test")
    return [
        {"question": str(r["question"]),
         "answer": str(r.get("answer") or r.get("answer_number") or "").replace(",", "")}
        for r in rows
    ]


def load_msvamp(lang: str) -> list[dict]:
    rows = _hf_load("Mathoctopus/MSVAMP", lang, "test")
    return [
        {"question": str(r.get("m_query") or r.get("query") or ""),
         "answer": str(r.get("response") or r.get("answer") or "").replace(",", "")}
        for r in rows
    ]


def load_xcsqa(lang: str) -> list[dict]:
    config = f"X-CSQA-{lang}"
    for split in ("validation", "test"):
        try:
            rows = _hf_load("INK-USC/xcsr", config, split)
        except RuntimeError:
            continue
        if rows and str(rows[0].get("answerKey", "")).strip():
            return rows
    raise RuntimeError(f"Could not load X-CSQA for '{lang}' from INK-USC/xcsr.")


def load_xnli(lang: str) -> list[dict]:
    rows = _hf_load("xnli", lang, "test")
    return [
        {"premise": str(r["premise"]), "hypothesis": str(r["hypothesis"]),
         "label": _XNLI_INT_TO_STR.get(int(r["label"]), "entailment")}
        for r in rows
    ]


_LOADERS = {"mgsm": load_mgsm, "msvamp": load_msvamp, "x-csqa": load_xcsqa, "xnli": load_xnli}


# ─── Prompt builders (identical to Stage2/eval_multilingual.py) ─────────────

def build_math_prompt(question: str) -> str:
    return f"{question}\n\nLet's think step by step."


def build_xcsqa_prompt(sample: dict) -> tuple[str, list[str]]:
    stem: str = sample["question"]["stem"]
    choices_dict: dict = sample["question"]["choices"]
    labels: list[str] = choices_dict["label"]
    texts: list[str] = choices_dict["text"]
    choice_lines = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    prompt = (
        f"Question: {stem}\n"
        f"Choices:\n{choice_lines}\n\n"
        "Answer with the letter corresponding to the correct choice (e.g., A, B, C, D, ...)."
    )
    return prompt, labels


def build_xnli_prompt(sample: dict) -> str:
    return (
        "Determine the relationship between the premise and the hypothesis. "
        "Respond with exactly one word: entailment, neutral, or contradiction.\n\n"
        f"Premise: {sample['premise']}\n"
        f"Hypothesis: {sample['hypothesis']}\n"
        "Answer:"
    )


# ─── Scoring (identical to Stage2/eval_multilingual.py) ────────────────────

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


def classify_nli(text: str) -> str:
    lower = text.lower()
    for label in _NLI_LABEL_STRINGS:
        if label in lower:
            return label
    return text.strip().lower()


def classify_xcsqa(text: str, valid_letters: list[str]) -> str:
    valid_set = set(valid_letters)
    matches = [ch for ch in text if ch in valid_set]
    return matches[-1] if matches else valid_letters[0]


# ─── Evaluation loops ────────────────────────────────────────────────────────

def _log_summary(
    task: str,
    results: dict[str, float],
    total_correct: int,
    total_examples: int,
    logger: logging.Logger,
) -> None:
    if not results:
        return
    macro = sum(results.values()) / len(results)
    micro = total_correct / total_examples * 100 if total_examples else 0.0
    logger.info("=== %s Baseline Summary ===", task.upper())
    for lang, acc in results.items():
        logger.info("  %-8s %.2f%%", lang, acc)
    logger.info("  Macro-avg (%d langs): %.2f%%", len(results), macro)
    logger.info("  Micro-avg (%d examples): %.2f%%", total_examples, micro)


def evaluate_math(
    task: str,
    langs: list[str],
    model: Qwen3VLForConditionalGeneration,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_examples: int | None,
    logger: logging.Logger,
) -> dict[str, float]:
    loader = _LOADERS[task]
    results: dict[str, float] = {}
    total_correct_all = 0
    total_all = 0

    for lang in langs:
        samples = loader(lang)
        if max_examples is not None:
            samples = samples[:max_examples]

        correct = 0
        logger.info("[%s] lang=%s | %d examples", task, lang, len(samples))

        for idx, sample in enumerate(tqdm(samples, desc=f"{task}/{lang}")):
            prompt = build_math_prompt(sample["question"])
            raw_out = generate_text(prompt, model, tokenizer, device, TASK_MAX_NEW_TOKENS[task], _MATH_SYSTEM)
            ok = math_correct(raw_out, sample["answer"])
            if ok:
                correct += 1
            if idx < 5:
                logger.info(
                    "lang=%s idx=%d | question=%r | raw_out=%r | extracted=%r | target=%s | ok=%s",
                    lang, idx, sample["question"][:80], raw_out[:200],
                    extract_math_answer(raw_out), sample["answer"], ok,
                )

        acc = correct / len(samples) * 100 if samples else 0.0
        results[lang] = acc
        total_correct_all += correct
        total_all += len(samples)
        logger.info("[%s] lang=%s accuracy=%.2f%%", task, lang, acc)

    _log_summary(task, results, total_correct_all, total_all, logger)
    return results


def evaluate_xcsqa(
    langs: list[str],
    model: Qwen3VLForConditionalGeneration,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_examples: int | None,
    logger: logging.Logger,
) -> dict[str, float]:
    results: dict[str, float] = {}
    total_correct_all = 0
    total_all = 0

    for lang in langs:
        samples = load_xcsqa(lang)
        if max_examples is not None:
            samples = samples[:max_examples]

        correct = 0
        logger.info("[x-csqa] lang=%s | %d examples", lang, len(samples))

        for idx, sample in enumerate(tqdm(samples, desc=f"x-csqa/{lang}")):
            prompt, valid_letters = build_xcsqa_prompt(sample)
            raw_out = generate_text(prompt, model, tokenizer, device, TASK_MAX_NEW_TOKENS["x-csqa"], _XCSQA_SYSTEM)
            pred = classify_xcsqa(raw_out, valid_letters)
            raw_key = sample["answerKey"]
            if isinstance(raw_key, int):
                expected = chr(ord("A") + raw_key)
            elif isinstance(raw_key, str) and raw_key.isdigit():
                expected = chr(ord("A") + int(raw_key))
            else:
                expected = str(raw_key).upper()
            ok = pred == expected
            if ok:
                correct += 1
            if idx < 5:
                logger.info(
                    "lang=%s idx=%d | stem=%r | raw_out=%r | pred=%s | target=%s | ok=%s",
                    lang, idx, sample["question"]["stem"][:80], raw_out, pred, expected, ok,
                )

        acc = correct / len(samples) * 100 if samples else 0.0
        results[lang] = acc
        total_correct_all += correct
        total_all += len(samples)
        logger.info("[x-csqa] lang=%s accuracy=%.2f%%", lang, acc)

    _log_summary("x-csqa", results, total_correct_all, total_all, logger)
    return results


def evaluate_nli(
    task: str,
    langs: list[str],
    model: Qwen3VLForConditionalGeneration,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_examples: int | None,
    logger: logging.Logger,
) -> dict[str, float]:
    loader = _LOADERS[task]
    results: dict[str, float] = {}
    total_correct_all = 0
    total_all = 0

    for lang in langs:
        samples = loader(lang)
        if max_examples is not None:
            samples = samples[:max_examples]

        correct = 0
        logger.info("[%s] lang=%s | %d examples", task, lang, len(samples))

        for idx, sample in enumerate(tqdm(samples, desc=f"{task}/{lang}")):
            prompt = build_xnli_prompt(sample)
            raw_out = generate_text(prompt, model, tokenizer, device, TASK_MAX_NEW_TOKENS[task], _XNLI_SYSTEM)
            pred = classify_nli(raw_out)
            ok = pred == sample["label"]
            if ok:
                correct += 1
            if idx < 5:
                logger.info(
                    "lang=%s idx=%d | premise=%r | raw_out=%r | pred=%s | target=%s | ok=%s",
                    lang, idx, sample["premise"][:80], raw_out, pred, sample["label"], ok,
                )

        acc = correct / len(samples) * 100 if samples else 0.0
        results[lang] = acc
        total_correct_all += correct
        total_all += len(samples)
        logger.info("[%s] lang=%s accuracy=%.2f%%", task, lang, acc)

    _log_summary(task, results, total_correct_all, total_all, logger)
    return results


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline Qwen3-VL evaluation on multilingual benchmarks (no mapping layer)."
    )
    parser.add_argument(
        "--task", required=True, choices=["mgsm", "msvamp", "x-csqa", "xnli"],
        help="Benchmark to evaluate.",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Cap on examples per language (None = all).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run 5 examples on the first language only.",
    )
    args = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    logger = setup_logging(log_dir, args.task)

    langs = TASK_DEFAULT_LANGS[args.task]
    if args.smoke:
        langs = langs[:1]
        args.max_examples = args.max_examples or 5

    logger.info("Task: %s | Model: %s | Languages: %s | max_examples=%s",
                args.task, args.model_id, langs, args.max_examples)

    model, tokenizer, device = load_model(args.model_id, local_files_only=args.local_files_only)

    if args.task in ("mgsm", "msvamp"):
        evaluate_math(args.task, langs, model, tokenizer, device, args.max_examples, logger)
    elif args.task == "x-csqa":
        evaluate_xcsqa(langs, model, tokenizer, device, args.max_examples, logger)
    elif args.task == "xnli":
        evaluate_nli(args.task, langs, model, tokenizer, device, args.max_examples, logger)


if __name__ == "__main__":
    main()
