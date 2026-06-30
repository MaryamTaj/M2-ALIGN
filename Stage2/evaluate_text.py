"""Stage 2 evaluation on text-only multilingual benchmarks.

Loads the trained AugmentedMindMerger (NLLB encoder → mapping → frozen
Qwen3-VL + LLM-side query prefix) and evaluates on the same text benchmarks
as Baseline/evaluate_text.py.  Both the source-language text (via NLLB) and
the chat-formatted task prompt (via the LLM tokenizer) are fed as the
generation prefix, matching the Stage 2 training setup.

Supported tasks
---------------
mgsm     – Multilingual Grade School Math    (juletxara/mgsm)
msvamp   – Multilingual SVAMP               (Mathoctopus/MSVAMP)
x-csqa   – Cross-lingual Commonsense QA     (INK-USC/xcsr)
xnli     – Cross-lingual NLI                (xnli)
"""
from __future__ import annotations

import argparse
import inspect
import logging
import os
import re
from datetime import datetime

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model import AugmentedMindMerger


# ─── NLLB language codes (ISO 639-1 → NLLB flores code) ────────────────────

NLLB_CODES: dict[str, str] = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "th": "tha_Thai",
    "sw": "swh_Latn",
    "bn": "ben_Beng",
    "te": "tel_Telu",
    "ar": "arb_Arab",
    "bg": "bul_Cyrl",
    "el": "ell_Grek",
    "hi": "hin_Deva",
    "hu": "hun_Latn",
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "tr": "tur_Latn",
    "ur": "urd_Arab",
    "vi": "vie_Latn",
}

TASK_DEFAULT_LANGS: dict[str, list[str]] = {
    "mgsm":   ["de", "es", "fr", "ru", "zh", "ja", "th", "sw", "bn", "te", "en"],
    "msvamp": ["de", "es", "fr", "ru", "zh", "ja", "th", "sw", "bn", "en"],
    "x-csqa": [
        "ar", "de", "en", "es", "fr", "hi", "it", "ja",
        "nl", "pl", "pt", "ru", "sw", "ur", "vi", "zh",
    ],
    "xnli": ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"],
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
    log_path = os.path.join(log_dir, f"stage2_eval_{task}_{timestamp}.log")
    logger = logging.getLogger(f"stage2_eval_{task}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger.addHandler(logging.FileHandler(log_path, encoding="utf-8"))
    logger.addHandler(logging.StreamHandler())
    for h in logger.handlers:
        h.setFormatter(fmt)
    return logger


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model(
    llm_path: str,
    mt_path: str,
    mapping_ckpt: str,
    local_files_only: bool,
    max_gen_len: int,
    logger: logging.Logger,
) -> tuple[AugmentedMindMerger, AutoTokenizer, AutoTokenizer, torch.device, torch.dtype]:
    assert torch.cuda.is_available(), "CUDA not available – request a GPU node."
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    tokenizer_llm = AutoTokenizer.from_pretrained(
        llm_path, use_fast=False, local_files_only=local_files_only
    )
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"
    tokenizer_llm.truncation_side = "left"

    tokenizer_mt = AutoTokenizer.from_pretrained(mt_path, local_files_only=local_files_only)

    sig = inspect.signature(AugmentedMindMerger.__init__)
    extra = {"local_files_only": local_files_only} if "local_files_only" in sig.parameters else {}
    model = AugmentedMindMerger(
        mt_path=mt_path,
        llm_path=llm_path,
        max_gen_len=max_gen_len,
        llm_bos_token_id=tokenizer_llm.bos_token_id,
        llm_pad_token_id=tokenizer_llm.pad_token_id,
        **extra,
    )

    ckpt = torch.load(mapping_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.mapping.load_state_dict(state_dict, strict=False)
    logger.info("Loaded mapping from: %s", mapping_ckpt)
    if missing:
        logger.warning("  missing keys (%d): %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("  unexpected keys (%d): %s", len(unexpected), unexpected[:5])

    model.model_mt.to(device)
    model.model_llm.to(device)
    model.mapping.to(device)
    model.eval()

    return model, tokenizer_mt, tokenizer_llm, device, amp_dtype


# ─── Tokenisation helpers ────────────────────────────────────────────────────

def mt_tokenize(
    text: str,
    nllb_code: str,
    tokenizer_mt: AutoTokenizer,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer_mt.src_lang = nllb_code
    enc = tokenizer_mt(text, truncation=True, max_length=max_seq_len, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def llm_tokenize(
    prompt: str,
    tokenizer_llm: AutoTokenizer,
    max_seq_len: int,
    device: torch.device,
    system_message: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    formatted = tokenizer_llm.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer_llm(
        formatted,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ─── Generation ─────────────────────────────────────────────────────────────

def generate_text(
    prompt: str,
    nllb_text: str,
    nllb_code: str,
    model: AugmentedMindMerger,
    tokenizer_mt: AutoTokenizer,
    tokenizer_llm: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_new_tokens: int,
    max_mt_seq_len: int,
    max_llm_seq_len: int,
    system_message: str | None = None,
) -> str:
    input_ids_mt, mask_mt = mt_tokenize(nllb_text, nllb_code, tokenizer_mt, max_mt_seq_len, device)
    input_ids_llm, mask_llm = llm_tokenize(
        prompt, tokenizer_llm, max_llm_seq_len, device, system_message
    )
    gen_kw = dict(do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=tokenizer_llm.eos_token_id)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        out_texts = model.generate(
            input_ids_mt=input_ids_mt,
            attention_mask_mt=mask_mt,
            input_ids_query_llm=input_ids_llm,
            mask_query_llm=mask_llm,
            tokenizer_llm=tokenizer_llm,
            generation_kwargs=gen_kw,
        )
    return (out_texts[0] if out_texts else "").strip()


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


# ─── Prompt builders (identical to Baseline/evaluate_text.py) ───────────────

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


# ─── NLLB text builders (raw source text for the MT encoder) ─────────────────

def build_math_nllb_text(question: str) -> str:
    return question


def build_xcsqa_nllb_text(sample: dict) -> str:
    stem: str = sample["question"]["stem"]
    choices_dict: dict = sample["question"]["choices"]
    labels: list[str] = choices_dict["label"]
    texts: list[str] = choices_dict["text"]
    options_block = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    return f"{stem}\n{options_block}"


def build_xnli_nllb_text(sample: dict) -> str:
    return f"{sample['premise']}\n{sample['hypothesis']}"


# ─── Scoring ────────────────────────────────────────────────────────────────

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


# ─── Summary logging ─────────────────────────────────────────────────────────

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
    logger.info("=== %s Stage2 Summary ===", task.upper())
    for lang, acc in results.items():
        logger.info("  %-8s %.2f%%", lang, acc)
    logger.info("  Macro-avg (%d langs): %.2f%%", len(results), macro)
    logger.info("  Micro-avg (%d examples): %.2f%%", total_examples, micro)


# ─── Evaluation loops ────────────────────────────────────────────────────────

def evaluate_math(
    task: str,
    langs: list[str],
    model: AugmentedMindMerger,
    tokenizer_mt: AutoTokenizer,
    tokenizer_llm: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_examples: int | None,
    max_mt_seq_len: int,
    max_llm_seq_len: int,
    logger: logging.Logger,
) -> dict[str, float]:
    loader = _LOADERS[task]
    results: dict[str, float] = {}
    total_correct_all = 0
    total_all = 0

    for lang in langs:
        nllb_code = NLLB_CODES.get(lang)
        if nllb_code is None:
            logger.warning("No NLLB code for '%s'; skipping.", lang)
            continue

        samples = loader(lang)
        if max_examples is not None:
            samples = samples[:max_examples]

        correct = 0
        logger.info("[%s] lang=%s | %d examples", task, lang, len(samples))

        for idx, sample in enumerate(tqdm(samples, desc=f"{task}/{lang}")):
            prompt = build_math_prompt(sample["question"])
            nllb_text = build_math_nllb_text(sample["question"])
            raw_out = generate_text(
                prompt, nllb_text, nllb_code, model, tokenizer_mt, tokenizer_llm,
                device, amp_dtype, TASK_MAX_NEW_TOKENS[task], max_mt_seq_len, max_llm_seq_len,
                _MATH_SYSTEM,
            )
            ok = math_correct(raw_out, sample["answer"])
            if ok:
                correct += 1
            if idx < 5:
                logger.info(
                    "lang=%s idx=%d | question=%r | prompt=%r | raw_out=%r | extracted=%r | target=%s | ok=%s",
                    lang, idx, sample["question"][:80], prompt[:200], raw_out[:200],
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
    model: AugmentedMindMerger,
    tokenizer_mt: AutoTokenizer,
    tokenizer_llm: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_examples: int | None,
    max_mt_seq_len: int,
    max_llm_seq_len: int,
    logger: logging.Logger,
) -> dict[str, float]:
    results: dict[str, float] = {}
    total_correct_all = 0
    total_all = 0

    for lang in langs:
        nllb_code = NLLB_CODES.get(lang)
        if nllb_code is None:
            logger.warning("No NLLB code for '%s'; skipping.", lang)
            continue

        samples = load_xcsqa(lang)
        if max_examples is not None:
            samples = samples[:max_examples]

        correct = 0
        logger.info("[x-csqa] lang=%s | %d examples", lang, len(samples))

        for idx, sample in enumerate(tqdm(samples, desc=f"x-csqa/{lang}")):
            prompt, valid_letters = build_xcsqa_prompt(sample)
            nllb_text = build_xcsqa_nllb_text(sample)
            raw_out = generate_text(
                prompt, nllb_text, nllb_code, model, tokenizer_mt, tokenizer_llm,
                device, amp_dtype, TASK_MAX_NEW_TOKENS["x-csqa"], max_mt_seq_len, max_llm_seq_len,
                _XCSQA_SYSTEM,
            )
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
                    "lang=%s idx=%d | stem=%r | prompt=%r | raw_out=%r | pred=%s | target=%s | ok=%s",
                    lang, idx, sample["question"]["stem"][:80], prompt[:200], raw_out, pred, expected, ok,
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
    model: AugmentedMindMerger,
    tokenizer_mt: AutoTokenizer,
    tokenizer_llm: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_examples: int | None,
    max_mt_seq_len: int,
    max_llm_seq_len: int,
    logger: logging.Logger,
) -> dict[str, float]:
    loader = _LOADERS[task]
    results: dict[str, float] = {}
    total_correct_all = 0
    total_all = 0

    for lang in langs:
        nllb_code = NLLB_CODES.get(lang)
        if nllb_code is None:
            logger.warning("No NLLB code for '%s'; skipping.", lang)
            continue

        samples = loader(lang)
        if max_examples is not None:
            samples = samples[:max_examples]

        correct = 0
        logger.info("[%s] lang=%s | %d examples", task, lang, len(samples))

        for idx, sample in enumerate(tqdm(samples, desc=f"{task}/{lang}")):
            prompt = build_xnli_prompt(sample)
            nllb_text = build_xnli_nllb_text(sample)
            raw_out = generate_text(
                prompt, nllb_text, nllb_code, model, tokenizer_mt, tokenizer_llm,
                device, amp_dtype, TASK_MAX_NEW_TOKENS[task], max_mt_seq_len, max_llm_seq_len,
                _XNLI_SYSTEM,
            )
            pred = classify_nli(raw_out)
            ok = pred == sample["label"]
            if ok:
                correct += 1
            if idx < 5:
                logger.info(
                    "lang=%s idx=%d | premise=%r | prompt=%r | raw_out=%r | pred=%s | target=%s | ok=%s",
                    lang, idx, sample["premise"][:80], prompt[:200], raw_out, pred, sample["label"], ok,
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
        description="Stage 2 AugmentedMindMerger evaluation on multilingual text benchmarks."
    )
    parser.add_argument(
        "--task", required=True, choices=["mgsm", "msvamp", "x-csqa", "xnli"],
        help="Benchmark to evaluate.",
    )
    parser.add_argument("--llm-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--mt-path", default="facebook/nllb-200-distilled-600M")
    parser.add_argument(
        "--mapping-ckpt",
        default="./outputs/augmentation/mapping/pytorch_model.bin",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--langs", nargs="+", default=None,
        help="ISO 639-1 language codes to evaluate (e.g. --langs sw en zh). Defaults to task default.",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Cap on examples per language (None = all).",
    )
    parser.add_argument("--max-mt-seq-len", type=int, default=256)
    parser.add_argument("--max-llm-seq-len", type=int, default=2048)
    parser.add_argument("--max-gen-len", type=int, default=512)
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

    logger.info(
        "Task: %s | MT: %s | LLM: %s | Languages: %s | max_examples=%s",
        args.task, args.mt_path, args.llm_path, langs, args.max_examples,
    )

    model, tokenizer_mt, tokenizer_llm, device, amp_dtype = load_model(
        llm_path=args.llm_path,
        mt_path=args.mt_path,
        mapping_ckpt=args.mapping_ckpt,
        local_files_only=args.local_files_only,
        max_gen_len=args.max_gen_len,
        logger=logger,
    )

    common_kw = dict(
        model=model,
        tokenizer_mt=tokenizer_mt,
        tokenizer_llm=tokenizer_llm,
        device=device,
        amp_dtype=amp_dtype,
        max_examples=args.max_examples,
        max_mt_seq_len=args.max_mt_seq_len,
        max_llm_seq_len=args.max_llm_seq_len,
        logger=logger,
    )

    if args.task in ("mgsm", "msvamp"):
        evaluate_math(args.task, langs, **common_kw)
    elif args.task == "x-csqa":
        evaluate_xcsqa(langs, **common_kw)
    elif args.task == "xnli":
        evaluate_nli(args.task, langs, **common_kw)


if __name__ == "__main__":
    main()
