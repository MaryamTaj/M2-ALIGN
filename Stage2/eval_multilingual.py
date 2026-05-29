"""Evaluate Stage 2 AugmentedMindMerger on multilingual benchmarks.

Supported tasks
---------------
mgsm     – Multilingual Grade School Math    (juletxara/mgsm)
msvamp   – Multilingual SVAMP               (juletxara/msvamp)
x-csqa   – Cross-lingual Commonsense QA     (xcsr/x_csqa)
xnli     – Cross-lingual NLI                (xnli)
afrimgsm – African-language MGSM            (masakhane/afrimgsm)
afrixnli – African-language XNLI            (masakhane/afrixnli)

Each task uses the same AugmentedMindMerger prefix as evaluate.py:
    [BOS] + X_m + [end_boundary] + T
where X_m encodes the source-language text via NLLB and T is the
chat-formatted task prompt.

Usage
-----
    python eval_multilingual.py --task mgsm \\
        --llm-path  Qwen/Qwen3-VL-8B-Instruct \\
        --mt-path   facebook/nllb-200-distilled-600M \\
        --mapping-ckpt Stage2/outputs/augmentation/mapping/pytorch_model.bin

    # Specific languages only:
    python eval_multilingual.py --task xnli --langs de,fr,zh ...

    # Fast smoke test (5 examples, one language):
    python eval_multilingual.py --task mgsm --smoke ...
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import re
from datetime import datetime
from typing import Iterable

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, NllbTokenizer

from model import AugmentedMindMerger


# ─── NLLB language codes (ISO 639-1 → NLLB flores code) ────────────────────

NLLB_CODES: dict[str, str] = {
    # High-resource / MGSM / XNLI / X-CSQA
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
    # African languages (AfriMGSM / AfriXNLI)
    "am": "amh_Ethi",
    "ee": "ewe_Latn",
    "ha": "hau_Latn",
    "ig": "ibo_Latn",
    "rw": "kin_Latn",
    "lg": "lug_Latn",
    "om": "gaz_Latn",
    "sn": "sna_Latn",
    "so": "som_Latn",
    "st": "sot_Latn",
    "tw": "twi_Latn",
    "wo": "wol_Latn",
    "xh": "xho_Latn",
    "yo": "yor_Latn",
    "zu": "zul_Latn",
}

# ─── Default language lists per task ────────────────────────────────────────

TASK_DEFAULT_LANGS: dict[str, list[str]] = {
    "mgsm": ["de", "es", "fr", "ru", "zh", "ja", "th", "sw", "bn", "te", "en"],
    "msvamp": ["de", "es", "fr", "ru", "zh", "ja", "th", "sw", "bn", "en"],
    "x-csqa": [
        "ar", "de", "en", "es", "fr", "hi", "it", "ja",
        "nl", "pl", "pt", "ru", "sw", "ur", "vi", "zh",
    ],
    "xnli": ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"],
    "afrimgsm": ["am", "ha", "ig", "rw", "sn", "sw", "wo", "xh", "yo", "zu"],
    "afrixnli": ["am", "ee", "ha", "ig", "rw", "sn", "st", "sw", "tw", "wo", "xh", "yo", "zu"],
}

# ─── Maximum new tokens per task ────────────────────────────────────────────

TASK_MAX_NEW_TOKENS: dict[str, int] = {
    "mgsm": 512,
    "msvamp": 512,
    "x-csqa": 6,
    "xnli": 20,
    "afrimgsm": 512,
    "afrixnli": 20,
}

# ─── NLI label constants ────────────────────────────────────────────────────

# Integer label → string used as both the generation target and for scoring.
# The model was trained on MultiNLI which uses Entailment/Neutral/Contradiction
# (capitalised), so we match case-insensitively at scoring time.
_XNLI_INT_TO_STR = {0: "entailment", 1: "neutral", 2: "contradiction"}
_NLI_LABEL_STRINGS = ["entailment", "neutral", "contradiction"]


# ─── Logging ────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str, task: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"eval_{task}_{timestamp}.log")

    logger = logging.getLogger(f"stage2_eval_{task}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ─── Prompt builders ────────────────────────────────────────────────────────

def build_math_prompt(question: str) -> tuple[str, str]:
    """Return (prompt, nllb_text) for math tasks (MGSM / MSVAMP / AfriMGSM).

    The prompt encourages chain-of-thought; NLLB receives only the question.
    """
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{question}\n\n"
        "### Response: Let's think step by step."
    )
    return prompt, question


def build_xcsqa_prompt(sample: dict) -> tuple[str, str, list[str]]:
    """Return (prompt, nllb_text, valid_letters) for X-CSQA.

    Expects the INK-USC/xcsr schema:
        sample["question"]["stem"]              – question text
        sample["question"]["choices"]["label"]  – list of letters ["A", "B", ...]
        sample["question"]["choices"]["text"]   – parallel list of option strings
    """
    stem: str = sample["question"]["stem"]
    choices_dict: dict = sample["question"]["choices"]
    labels: list[str] = choices_dict["label"]
    texts: list[str] = choices_dict["text"]
    choice_lines = "\t".join(f"{l}. {t}" for l, t in zip(labels, texts))
    valid_letters = labels
    prompt = (
        f"{stem}\n"
        f"Options: {choice_lines}\n"
        "Answer:"
    )
    nllb_text = f"{stem}\n" + " ".join(texts)
    return prompt, nllb_text, valid_letters


def build_xnli_prompt(sample: dict) -> tuple[str, str]:
    """Return (prompt, nllb_text) for NLI tasks (XNLI / AfriXNLI).

    Expects sample["premise"] and sample["hypothesis"].
    Mirrors the MultiNLI training format used in Stage 2.
    """
    premise: str = sample["premise"]
    hypothesis: str = sample["hypothesis"]
    prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel:"
    nllb_text = f"{premise} {hypothesis}"
    return prompt, nllb_text


# ─── Dataset loaders ────────────────────────────────────────────────────────

def _hf_load(hf_id: str, config: str, split: str) -> list[dict]:
    """Thin wrapper around load_dataset that surfaces the HF id on error."""
    try:
        ds = load_dataset(hf_id, config, split=split, download_mode="reuse_dataset_if_exists")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load HuggingFace dataset '{hf_id}' config='{config}' split='{split}'.\n"
            f"  Verify the dataset ID and your network/cache. Original error: {exc}"
        ) from exc
    return list(ds)


def load_mgsm(lang: str) -> list[dict]:
    rows = _hf_load("juletxara/mgsm", lang, "test")
    out = []
    for r in rows:
        answer = r.get("answer") or r.get("answer_number") or ""
        out.append({"question": str(r["question"]), "answer": str(answer).replace(",", "")})
    return out


def load_msvamp(lang: str) -> list[dict]:
    # Mathoctopus/MSVAMP uses ISO-2 config codes (sw, de, es, …).
    # Fields: query (question text), response (numeric answer string).
    rows = _hf_load("Mathoctopus/MSVAMP", lang, "test")
    out = []
    for r in rows:
        question = r.get("query") or r.get("m_query") or ""
        answer = r.get("response") or r.get("answer") or ""
        out.append({"question": str(question), "answer": str(answer).replace(",", "")})
    return out


# AfriMGSM / AfriXNLI use full FLORES-style config codes, not ISO-2.
_AFRI_CONFIG = {"sw": "swa", "wo": "wol", "yo": "yor"}


def load_xcsqa(lang: str) -> list[dict]:
    # INK-USC/xcsr uses config names of the form "X-CSQA-{lang}".
    # The test split has 1000 labelled examples; validation has 300.
    config = f"X-CSQA-{lang}"
    for split in ("test", "validation"):
        try:
            return _hf_load("INK-USC/xcsr", config, split)
        except RuntimeError:
            continue
    raise RuntimeError(
        f"Could not load X-CSQA for language '{lang}' from 'INK-USC/xcsr' "
        f"(tried config '{config}'). Check that '{lang}' is a supported language code."
    )


def load_xnli(lang: str) -> list[dict]:
    rows = _hf_load("xnli", lang, "test")
    out = []
    for r in rows:
        label_int = int(r["label"])
        out.append({
            "premise": str(r["premise"]),
            "hypothesis": str(r["hypothesis"]),
            "label": _XNLI_INT_TO_STR.get(label_int, "entailment"),
        })
    return out


def load_afrimgsm(lang: str) -> list[dict]:
    # masakhane/afrimgsm uses full config codes: swa, wol, yor (not sw, wo, yo).
    config = _AFRI_CONFIG.get(lang, lang)
    rows = _hf_load("masakhane/afrimgsm", config, "test")
    out = []
    for r in rows:
        answer = r.get("answer") or r.get("answer_number") or ""
        out.append({"question": str(r["question"]), "answer": str(answer).replace(",", "")})
    return out


def load_afrixnli(lang: str) -> list[dict]:
    # masakhane/afrixnli uses full config codes: swa, wol, yor (not sw, wo, yo).
    config = _AFRI_CONFIG.get(lang, lang)
    rows = _hf_load("masakhane/afrixnli", config, "test")
    out = []
    for r in rows:
        label_raw = r.get("label", 0)
        if isinstance(label_raw, int):
            label_str = _XNLI_INT_TO_STR.get(label_raw, "entailment")
        else:
            label_str = str(label_raw).lower()
        out.append({
            "premise": str(r["premise"]),
            "hypothesis": str(r["hypothesis"]),
            "label": label_str,
        })
    return out


_LOADERS = {
    "mgsm": load_mgsm,
    "msvamp": load_msvamp,
    "x-csqa": load_xcsqa,
    "xnli": load_xnli,
    "afrimgsm": load_afrimgsm,
    "afrixnli": load_afrixnli,
}


# ─── Answer extraction / scoring ────────────────────────────────────────────

def extract_math_answer(text: str) -> str | None:
    """Extract the final numeric answer from a chain-of-thought response."""
    # Prefer explicit "The answer is X" or "answer: X" patterns.
    m = re.search(
        r"(?:the\s+)?answer\s+is[:\s]+(-?[\d,]+(?:\.\d+)?)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).replace(",", "")
    # #### N  (common GSM8K / MetaMathQA convention)
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    # Fall back: last standalone number in the text.
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
    """Pick the first NLI label word that appears in *text* (case-insensitive)."""
    lower = text.lower()
    for label in _NLI_LABEL_STRINGS:
        if label in lower:
            return label
    return "entailment"  # safe fallback


def classify_xcsqa(text: str, valid_letters: list[str]) -> str:
    """Return the first valid option letter found in *text*."""
    for ch in text.strip():
        if ch in valid_letters:
            return ch
    return valid_letters[0]  # safe fallback


# ─── Tokenisation helpers (mirrors Stage2/evaluate.py) ──────────────────────

def mt_input_features(
    texts: Iterable[str],
    nllb_codes: Iterable[str],
    tokenizer_mt: NllbTokenizer,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids: list[list[int]] = []
    masks: list[list[int]] = []
    for text, code in zip(texts, nllb_codes):
        tokenizer_mt.src_lang = code
        enc = tokenizer_mt(text, truncation=True, max_length=max_seq_len, padding=False)
        ids.append(enc["input_ids"])
        masks.append(enc["attention_mask"])
    max_len = max(len(x) for x in ids)
    pad_id = tokenizer_mt.pad_token_id
    for i in range(len(ids)):
        while len(ids[i]) < max_len:
            ids[i].append(pad_id)
            masks[i].append(0)
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def llm_input_features(
    texts: Iterable[str],
    tokenizer_llm: AutoTokenizer,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer_llm(
        list(texts),
        truncation=True,
        max_length=max_seq_len,
        padding=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ─── Per-example inference ───────────────────────────────────────────────────

@torch.inference_mode()
def run_inference(
    model: AugmentedMindMerger,
    tokenizer_mt: NllbTokenizer,
    tokenizer_llm: AutoTokenizer,
    prompt: str,
    nllb_text: str,
    nllb_code: str,
    max_mt_seq_len: int,
    max_llm_seq_len: int,
    max_new_tokens: int,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> str:
    """Run one forward+generate step and return the decoded output string."""
    input_ids_mt, mask_mt = mt_input_features(
        [nllb_text], [nllb_code], tokenizer_mt, max_mt_seq_len, device
    )
    formatted = tokenizer_llm.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids_query_llm, mask_query_llm = llm_input_features(
        [formatted], tokenizer_llm, max_llm_seq_len, device
    )
    gen_kw = dict(do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=tokenizer_llm.eos_token_id)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        out_texts = model.generate(
            input_ids_mt=input_ids_mt,
            attention_mask_mt=mask_mt,
            input_ids_query_llm=input_ids_query_llm,
            mask_query_llm=mask_query_llm,
            tokenizer_llm=tokenizer_llm,
            generation_kwargs=gen_kw,
        )
    return (out_texts[0] if out_texts else "").strip()


# ─── Per-task evaluation loop ────────────────────────────────────────────────

def evaluate_math(
    task: str,
    langs: list[str],
    model: AugmentedMindMerger,
    tokenizer_mt: NllbTokenizer,
    tokenizer_llm: AutoTokenizer,
    *,
    max_mt_seq_len: int,
    max_llm_seq_len: int,
    max_examples: int | None,
    amp_dtype: torch.dtype,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, float]:
    loader = _LOADERS[task]
    results: dict[str, float] = {}
    total_correct_all = 0
    total_all = 0

    for lang in langs:
        nllb_code = NLLB_CODES.get(lang)
        if nllb_code is None:
            logger.warning("No NLLB code for language '%s'; skipping.", lang)
            continue

        samples = loader(lang)
        if max_examples is not None:
            samples = samples[:max_examples]

        correct = 0
        logger.info("[%s] lang=%s | %d examples", task, lang, len(samples))

        for idx, sample in enumerate(tqdm(samples, desc=f"{task}/{lang}")):
            prompt, nllb_text = build_math_prompt(sample["question"])
            raw_out = run_inference(
                model, tokenizer_mt, tokenizer_llm,
                prompt, nllb_text, nllb_code,
                max_mt_seq_len, max_llm_seq_len,
                TASK_MAX_NEW_TOKENS[task], amp_dtype, device,
            )
            ok = math_correct(raw_out, sample["answer"])
            if ok:
                correct += 1
            if idx < 8:
                logger.debug(
                    "lang=%s idx=%d | question=%r | raw_out=%r | extracted=%r | target=%s | ok=%s",
                    lang, idx, sample["question"][:100], raw_out[:200],
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
    tokenizer_mt: NllbTokenizer,
    tokenizer_llm: AutoTokenizer,
    *,
    max_mt_seq_len: int,
    max_llm_seq_len: int,
    max_examples: int | None,
    amp_dtype: torch.dtype,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, float]:
    results: dict[str, float] = {}
    total_correct_all = 0
    total_all = 0

    for lang in langs:
        nllb_code = NLLB_CODES.get(lang)
        if nllb_code is None:
            logger.warning("No NLLB code for language '%s'; skipping.", lang)
            continue

        samples = load_xcsqa(lang)
        if max_examples is not None:
            samples = samples[:max_examples]

        correct = 0
        logger.info("[x-csqa] lang=%s | %d examples", lang, len(samples))

        for idx, sample in enumerate(tqdm(samples, desc=f"x-csqa/{lang}")):
            prompt, nllb_text, valid_letters = build_xcsqa_prompt(sample)
            raw_out = run_inference(
                model, tokenizer_mt, tokenizer_llm,
                prompt, nllb_text, nllb_code,
                max_mt_seq_len, max_llm_seq_len,
                TASK_MAX_NEW_TOKENS["x-csqa"], amp_dtype, device,
            )
            pred = classify_xcsqa(raw_out, valid_letters)
            ok = pred == sample["answerKey"]
            if ok:
                correct += 1
            if idx < 8:
                logger.debug(
                    "lang=%s idx=%d | stem=%r | raw_out=%r | pred=%s | target=%s | ok=%s",
                    lang, idx, sample["question"]["stem"][:80], raw_out, pred, sample["answerKey"], ok,
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
    tokenizer_mt: NllbTokenizer,
    tokenizer_llm: AutoTokenizer,
    *,
    max_mt_seq_len: int,
    max_llm_seq_len: int,
    max_examples: int | None,
    amp_dtype: torch.dtype,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, float]:
    loader = _LOADERS[task]
    results: dict[str, float] = {}
    total_correct_all = 0
    total_all = 0

    for lang in langs:
        nllb_code = NLLB_CODES.get(lang)
        if nllb_code is None:
            logger.warning("No NLLB code for language '%s'; skipping.", lang)
            continue

        samples = loader(lang)
        if max_examples is not None:
            samples = samples[:max_examples]

        correct = 0
        logger.info("[%s] lang=%s | %d examples", task, lang, len(samples))

        for idx, sample in enumerate(tqdm(samples, desc=f"{task}/{lang}")):
            prompt, nllb_text = build_xnli_prompt(sample)
            raw_out = run_inference(
                model, tokenizer_mt, tokenizer_llm,
                prompt, nllb_text, nllb_code,
                max_mt_seq_len, max_llm_seq_len,
                TASK_MAX_NEW_TOKENS[task], amp_dtype, device,
            )
            pred = classify_nli(raw_out)
            ok = pred == sample["label"]
            if ok:
                correct += 1
            if idx < 8:
                logger.debug(
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


# ─── Summary logging ─────────────────────────────────────────────────────────

def _log_summary(
    task: str,
    results: dict[str, float],
    total_correct: int,
    total_examples: int,
    logger: logging.Logger,
) -> None:
    if not results:
        logger.warning("[%s] No results to summarise.", task)
        return
    macro = sum(results.values()) / len(results)
    micro = total_correct / total_examples * 100 if total_examples else 0.0
    logger.info("=== %s Summary ===", task.upper())
    for lang, acc in results.items():
        logger.info("  %-8s %.2f%%", lang, acc)
    logger.info("  Macro-avg (n=%d langs): %.2f%%", len(results), macro)
    logger.info("  Micro-avg (%d examples): %.2f%%", total_examples, micro)


# ─── Model loading (mirrors Stage2/evaluate.py) ──────────────────────────────

def load_model(
    llm_path: str,
    mt_path: str,
    mapping_ckpt: str,
    local_files_only: bool,
    max_gen_len: int,
    logger: logging.Logger,
) -> tuple[AugmentedMindMerger, NllbTokenizer, AutoTokenizer, torch.device, torch.dtype]:
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path, use_fast=False, local_files_only=local_files_only)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"
    tokenizer_llm.truncation_side = "left"

    tokenizer_mt = NllbTokenizer.from_pretrained(mt_path, local_files_only=local_files_only)

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


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 2 AugmentedMindMerger on multilingual benchmarks."
    )
    parser.add_argument(
        "--task", required=True,
        choices=["mgsm", "msvamp", "x-csqa", "xnli", "afrimgsm", "afrixnli"],
        help="Benchmark to evaluate.",
    )
    parser.add_argument("--llm-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--mt-path", default="facebook/nllb-200-distilled-600M")
    parser.add_argument(
        "--mapping-ckpt",
        default="Stage2/outputs/augmentation/mapping/pytorch_model.bin",
    )
    parser.add_argument(
        "--langs", default=None,
        help="Comma-separated ISO 639-1 codes to evaluate (default: all for the task).",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Cap on examples per language (None = all).",
    )
    parser.add_argument("--max-mt-seq-len", type=int, default=256)
    parser.add_argument("--max-llm-seq-len", type=int, default=2048)
    parser.add_argument("--max-gen-len", type=int, default=512)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run 5 examples on the first language only for a quick sanity check.",
    )
    args = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    logger = setup_logging(log_dir, args.task)

    langs = TASK_DEFAULT_LANGS[args.task]
    if args.langs:
        langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    if args.smoke:
        langs = langs[:1]
        args.max_examples = args.max_examples or 5

    for lang in langs:
        if lang not in NLLB_CODES:
            raise ValueError(
                f"Language '{lang}' has no NLLB code entry. "
                f"Add it to NLLB_CODES or choose from: {sorted(NLLB_CODES)}"
            )

    logger.info("Task: %s | Languages: %s | max_examples=%s", args.task, langs, args.max_examples)

    model, tokenizer_mt, tokenizer_llm, device, amp_dtype = load_model(
        llm_path=args.llm_path,
        mt_path=args.mt_path,
        mapping_ckpt=args.mapping_ckpt,
        local_files_only=args.local_files_only,
        max_gen_len=args.max_gen_len,
        logger=logger,
    )

    common_kw = dict(
        max_mt_seq_len=args.max_mt_seq_len,
        max_llm_seq_len=args.max_llm_seq_len,
        max_examples=args.max_examples,
        amp_dtype=amp_dtype,
        device=device,
        logger=logger,
    )

    if args.task in ("mgsm", "msvamp", "afrimgsm"):
        evaluate_math(args.task, langs, model, tokenizer_mt, tokenizer_llm, **common_kw)
    elif args.task == "x-csqa":
        evaluate_xcsqa(langs, model, tokenizer_mt, tokenizer_llm, **common_kw)
    elif args.task in ("xnli", "afrixnli"):
        evaluate_nli(args.task, langs, model, tokenizer_mt, tokenizer_llm, **common_kw)
    else:
        raise ValueError(f"Unknown task: {args.task!r}")


if __name__ == "__main__":
    main()
