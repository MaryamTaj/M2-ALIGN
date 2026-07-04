"""Stage 3a evaluation on text-only multilingual benchmarks.

Loads the trained AugmentedMindMerger (NLLB encoder → mapping → frozen
Qwen3-VL + LLM-side query prefix) and evaluates on the same text benchmarks
as Baseline/evaluate_text.py.  Both the source-language text (via NLLB) and
the chat-formatted task prompt (via the LLM tokenizer) are fed as the
generation prefix, matching the Stage 3a training setup.

Run once right after Stage 3a training (reference point for the secondary
hypothesis) and again after Stage 3b (VQA augmentation) to check for
regression — see the Stage 3 plan for the before/after comparison.

Supported tasks
---------------
mgsm     – Multilingual Grade School Math    (juletxara/mgsm)
msvamp   – Multilingual SVAMP               (Mathoctopus/MSVAMP)

x-csqa and xnli were dropped for now: standard XNLI's language list is
ar/bg/de/el/en/es/fr/hi/ru/sw/th/tr/ur/vi/zh -- "bg" is Bulgarian, not
Bengali -- and INK-USC/xcsr (X-CSQA) has the same gap. Both were removed
from Stage 3a training and evaluation entirely rather than relying on a
substitute for one language only.
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
    log_path = os.path.join(log_dir, f"stage3a_eval_{task}_{timestamp}.log")
    logger = logging.getLogger(f"stage3a_eval_{task}")
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


_LOADERS = {"mgsm": load_mgsm, "msvamp": load_msvamp}


# ─── Prompt builders (identical to Baseline/evaluate_text.py) ───────────────

def build_math_prompt(question: str) -> str:
    return f"{question}\n\nLet's think step by step."


# ─── NLLB text builders (raw source text for the MT encoder) ─────────────────

def build_math_nllb_text(question: str) -> str:
    return question


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
    logger.info("=== %s Stage3a Summary ===", task.upper())
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


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3a AugmentedMindMerger evaluation on multilingual text benchmarks."
    )
    parser.add_argument(
        "--task", required=True, choices=["mgsm", "msvamp"],
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

    evaluate_math(args.task, langs, **common_kw)


if __name__ == "__main__":
    main()
