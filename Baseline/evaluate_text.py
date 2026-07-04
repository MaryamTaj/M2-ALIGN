"""Baseline evaluation of Qwen3-VL 8B Instruct on text-only multilingual benchmarks.

No NLLB encoder, no Mapping layer -- this is the raw-model reference point
for the secondary hypothesis, parallel to how Baseline/evaluate_vqa.py is
the raw-model reference point for the primary (VQA) hypothesis. Compare
against Stage 3a's evaluate_text.py numbers (same benchmarks, same
languages) to see what the mapping/augmentation training actually buys.

Supported tasks
---------------
mgsm     – Multilingual Grade School Math    (juletxara/mgsm)
msvamp   – Multilingual SVAMP               (Mathoctopus/MSVAMP)

x-csqa and xnli are not supported here (dropped to match Stage 3a): standard
XNLI's language list is ar/bg/de/el/en/es/fr/hi/ru/sw/th/tr/ur/vi/zh -- "bg"
is Bulgarian, not Bengali -- and INK-USC/xcsr (X-CSQA) has the same gap.
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

# Default languages match Stage 3a's evaluation set: English as a
# high-resource reference, Swahili and Bengali as the languages Stage 3a
# actually trains/evaluates on.
TASK_DEFAULT_LANGS: dict[str, list[str]] = {
    "mgsm":   ["en", "sw", "bn"],
    "msvamp": ["en", "sw", "bn"],
}

TASK_MAX_NEW_TOKENS: dict[str, int] = {
    "mgsm": 512,
    "msvamp": 512,
}

_MATH_SYSTEM = "You are a helpful assistant that solves math problems step by step."


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


_LOADERS = {"mgsm": load_mgsm, "msvamp": load_msvamp}


# ─── Prompt builders (identical to Stage3/evaluate_text.py) ────────────────

def build_math_prompt(question: str) -> str:
    return f"{question}\n\nLet's think step by step."


# ─── Scoring (identical to Stage3/evaluate_text.py) ────────────────────────

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


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline Qwen3-VL evaluation on multilingual benchmarks (no mapping layer)."
    )
    parser.add_argument(
        "--task", required=True, choices=["mgsm", "msvamp"],
        help="Benchmark to evaluate.",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--langs", nargs="+", default=None,
        help="Language codes to evaluate (e.g. --langs sw bn en). Defaults to task default.",
    )
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

    langs = args.langs if args.langs else TASK_DEFAULT_LANGS[args.task]
    if args.smoke:
        langs = langs[:1]
        args.max_examples = args.max_examples or 5

    logger.info("Task: %s | Model: %s | Languages: %s | max_examples=%s",
                args.task, args.model_id, langs, args.max_examples)

    model, tokenizer, device = load_model(args.model_id, local_files_only=args.local_files_only)

    evaluate_math(args.task, langs, model, tokenizer, device, args.max_examples, logger)


if __name__ == "__main__":
    main()
