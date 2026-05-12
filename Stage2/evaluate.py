# coding=utf-8
"""Evaluate the Stage 2 AugmentedMindMerger checkpoint on MMLU-ProX.

Unlike Stage1/run_evaluating.py, this script feeds the LLM exactly the
prefix it was trained on at Stage 2:

    [BOS] + X_m + [end_boundary] + T

where:
    X_m = mapping(NLLB_encoder(question_in_source_language))
    T   = LLM_token_embedding(prompt)   # same prompt, LLM-tokenised

This matches the augmentation stage of the MindMerger paper (NeurIPS 2024)
and avoids the train/eval mismatch that occurs when Stage 1's eval path
(which omits T) is used with a Stage 2 checkpoint.

Decoding hyperparameters are kept consistent with Baseline/mmlu_prox.py and
Stage1/run_evaluating.py for direct comparability.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import string
from datetime import datetime
from typing import Iterable

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, NllbTokenizer

from model import AugmentedMindMerger


TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 40
PRESENCE_PENALTY = 2.0
N_SHOT = 5
FEWSHOT_SEED = 1234
MAX_NEW_TOKENS_FOR_MCQ = 6

MMLU_TO_SOURCE_LANGUAGE = {
    "sw": "Swahili",
    "wo": "Wolof",
    "yo": "Yoruba",
    "fr": "French",
}

NLLB_LANG_MAP = {
    "English": "eng_Latn",
    "Swahili": "swh_Latn",
    "Yoruba": "yor_Latn",
    "Wolof": "wol_Latn",
    "French": "fra_Latn",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> logging.Logger:
    """Create a logger that writes to stdout and a timestamped file in *log_dir*.

    Args:
        log_dir: Directory in which to create the log file (created if absent).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"eval_{timestamp}.log")

    logger = logging.getLogger("stage2_eval")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Prompt construction (kept in sync with Stage1/run_evaluating.py)
# ---------------------------------------------------------------------------

def extract_options(sample: dict) -> tuple[list[str], list[str]]:
    """Extract answer options from an MMLU-ProX sample dict.

    Reads keys of the form ``"option_N"`` (N integer), sorts by N, and
    returns parallel lists of option letters and option texts.

    Args:
        sample: An MMLU-ProX sample dict with ``"option_1"``, ``"option_2"``…

    Returns:
        A 2-tuple ``(letters, texts)`` of equal-length lists.
    """
    option_items = []
    for k, v in sample.items():
        if k.startswith("option_") and v is not None:
            idx = int(k.split("_")[1])
            option_items.append((idx, v))
    option_items.sort(key=lambda x: x[0])
    texts = [v for _, v in option_items]
    letters = list(string.ascii_uppercase[: len(texts)])
    return letters, texts


def format_options_block(letters: list[str], texts: list[str]) -> str:
    """Format option letters and texts into one ``"L. text"`` line per option.

    Args:
        letters: Option letters, e.g. ``["A", "B", "C", "D"]``.
        texts: Corresponding option text strings.

    Returns:
        A newline-separated string of ``"L. text"`` entries.
    """
    return "\n".join(f"{L}. {t}" for L, t in zip(letters, texts))


def qwen_eval_block(question: str, options_block: str, answer_letter: str | None = None) -> str:
    """Build a single MCQ block in the Qwen3-VL evaluation style.

    Args:
        question: The question text.
        options_block: Pre-formatted option lines from
            :func:`format_options_block`.
        answer_letter: Correct letter for few-shot demos; ``None`` to leave
            the block open for the model to complete.

    Returns:
        A formatted prompt block string.
    """
    base = (
        "Respond with only the letter of the correct option.\n"
        f"Question: {question} Possible answer choices:\n"
        f"{options_block}\n"
        "The best answer is:"
    )
    if answer_letter is not None:
        return base + f" {answer_letter}\n"
    return base


def build_fewshot_prompt(demo_samples: list[dict], test_sample: dict) -> tuple[str, list[str]]:
    """Construct a 5-shot MCQ prompt for one test example.

    Args:
        demo_samples: Few-shot demonstration sample dicts.
        test_sample: The test question to be answered.

    Returns:
        A 2-tuple ``(prompt, letters)`` where *letters* are the valid
        answer letters for *test_sample*.
    """
    blocks = []
    for s in demo_samples:
        letters, texts = extract_options(s)
        blocks.append(qwen_eval_block(s["question"], format_options_block(letters, texts), s["answer"]))

    letters, texts = extract_options(test_sample)
    blocks.append(qwen_eval_block(test_sample["question"], format_options_block(letters, texts)))
    return "\n\n".join(blocks), letters


def build_nllb_test_text(test_sample: dict) -> str:
    """Build the short target-language string for NLLB encoding.

    The NLLB encoder receives only the test question and its options (both
    in the target language).  Feeding it the full 5-shot prompt would
    mismatch the ``src_lang`` setting and right-truncate the test question
    under ``max_seq_len=256``.

    Args:
        test_sample: An MMLU-ProX test sample dict.

    Returns:
        A string with the question and options separated by a newline.
    """
    letters, texts = extract_options(test_sample)
    return f"{test_sample['question']}\n{format_options_block(letters, texts)}"


# ---------------------------------------------------------------------------
# Tokenisation helpers (kept in sync with Stage2/run_augmentation.py)
# ---------------------------------------------------------------------------

def mt_input_features(
    texts: Iterable[str],
    langs: Iterable[str],
    tokenizer_mt: NllbTokenizer,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise source-language texts with the NLLB tokenizer.

    Sets ``tokenizer_mt.src_lang`` per example and right-pads the batch.

    Args:
        texts: Source-language strings to tokenize.
        langs: Human-readable language names (keys in :data:`NLLB_LANG_MAP`).
        tokenizer_mt: An instantiated :class:`NllbTokenizer`.
        max_seq_len: Maximum sequence length; longer sequences are truncated.
        device: Target device for the returned tensors.

    Returns:
        A 2-tuple ``(input_ids, attention_mask)`` on *device*.
    """
    ids, masks = [], []
    for text, lang in zip(texts, langs):
        tokenizer_mt.src_lang = NLLB_LANG_MAP[lang]
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
    add_bos: bool,
    add_eos: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise texts with the LLM tokenizer.

    Matches the training-time helper exactly so that ``T`` is identical
    to the token stream the mapping was trained against.

    Args:
        texts: Strings to tokenize.
        tokenizer_llm: An instantiated LLM tokenizer.
        max_seq_len: Maximum sequence length; longer sequences are truncated.
        add_bos: Whether to prepend a BOS token.
        add_eos: Whether to append an EOS token.
        device: Target device for the returned tensors.

    Returns:
        A 2-tuple ``(input_ids, attention_mask)`` on *device*.
    """
    if hasattr(tokenizer_llm, "add_bos_token"):
        tokenizer_llm.add_bos_token = add_bos
    if hasattr(tokenizer_llm, "add_eos_token"):
        tokenizer_llm.add_eos_token = add_eos
    enc = tokenizer_llm(
        list(texts),
        truncation=True,
        max_length=max_seq_len,
        padding=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ---------------------------------------------------------------------------
# Per-example MCQ scoring
# ---------------------------------------------------------------------------

@torch.inference_mode()
def pick_choice(
    model: AugmentedMindMerger,
    tokenizer_mt: NllbTokenizer,
    tokenizer_llm: AutoTokenizer,
    prompt: str,
    nllb_text: str,
    source_language: str,
    max_mt_seq_len: int,
    max_llm_seq_len: int,
    choices: list[str],
    amp_dtype: torch.dtype,
    device: torch.device,
) -> tuple[str, str]:
    """Run one MCQ inference step with the Stage 2 prefix and return the predicted letter.

    The NLLB encoder receives only the test question + options.  The LLM
    receives the full 5-shot prompt via left-truncation so that the test
    question is always preserved if the prompt overflows *max_llm_seq_len*.

    Args:
        model: A loaded :class:`AugmentedMindMerger` in eval mode.
        tokenizer_mt: NLLB tokenizer.
        tokenizer_llm: Qwen3-VL tokenizer (``truncation_side`` must be
            ``"left"`` before calling this function).
        prompt: Full 5-shot MCQ prompt string for the LLM prefix ``T``.
        nllb_text: Target-language question + options for the MT encoder.
        source_language: Human-readable language name (key in
            :data:`NLLB_LANG_MAP`).
        max_mt_seq_len: Maximum sequence length for the NLLB encoder.
        max_llm_seq_len: Maximum sequence length for the LLM prompt.
        choices: Valid answer letters for this question.
        amp_dtype: Dtype for ``torch.autocast`` (bf16 or fp16).
        device: Device on which the model lives.

    Returns:
        A 2-tuple ``(predicted_letter, raw_output_string)``.
    """
    input_ids_mt, mask_mt = mt_input_features(
        [nllb_text], [source_language], tokenizer_mt, max_mt_seq_len, device
    )
    input_ids_query_llm, mask_query_llm = llm_input_features(
        [prompt], tokenizer_llm, max_llm_seq_len, add_bos=False, add_eos=False, device=device
    )
    gen_kw = dict(
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_new_tokens=MAX_NEW_TOKENS_FOR_MCQ,
        eos_token_id=tokenizer_llm.eos_token_id,
    )
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        out_texts = model.generate(
            input_ids_mt=input_ids_mt,
            attention_mask_mt=mask_mt,
            input_ids_query_llm=input_ids_query_llm,
            mask_query_llm=mask_query_llm,
            tokenizer_llm=tokenizer_llm,
            generation_kwargs=gen_kw,
            presence_penalty=PRESENCE_PENALTY,
        )
    out_text = (out_texts[0] if out_texts else "").strip()
    for ch in out_text:
        if ch in choices:
            return ch, out_text
    return choices[0], out_text


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def evaluate(
    *,
    llm_path: str,
    mt_path: str,
    mapping_ckpt: str,
    langs: list[str],
    local_files_only: bool,
    max_mt_seq_len: int,
    max_llm_seq_len: int,
    max_gen_len: int,
    max_test_examples: int | None,
    max_val_examples: int | None,
    print_sample_count: int = 20,
    logger: logging.Logger,
) -> tuple[dict[str, float], float, float]:
    """Evaluate a Stage 2 AugmentedMindMerger checkpoint on MMLU-ProX.

    Loads the model, iterates over *langs*, and computes per-language and
    aggregate accuracy.  Diagnostic information (tokenizer special tokens,
    checkpoint load summary, per-sample outputs) is captured by *logger*.

    Args:
        llm_path: HF id or local path for Qwen3-VL.
        mt_path: HF id or local path for the NLLB encoder.
        mapping_ckpt: Path to the Stage 2 mapping checkpoint.
        langs: MMLU-ProX language codes to evaluate (e.g. ``["sw", "fr"]``).
        local_files_only: If ``True``, do not attempt Hub downloads.
        max_mt_seq_len: Maximum NLLB sequence length.
        max_llm_seq_len: Maximum LLM prompt sequence length.
        max_gen_len: Maximum new tokens for generation.
        max_test_examples: Cap on test examples per language (``None`` = all).
        max_val_examples: Cap on validation examples per language.
        print_sample_count: Number of (prompt, output, answer) triples to
            print per language; 0 disables.
        logger: Logger for progress and diagnostic messages.

    Returns:
        A 3-tuple ``(results, macro_avg, micro_avg)`` where *results* maps
        language code → accuracy and the averages are percentages.
    """
    device = torch.device("cuda")
    cap_major = torch.cuda.get_device_capability(0)[0]
    amp_dtype = torch.bfloat16 if cap_major >= 8 else torch.float16

    tokenizer_llm = AutoTokenizer.from_pretrained(
        llm_path, use_fast=False, local_files_only=local_files_only
    )
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"
    # Left-truncate so that if a 5-shot prompt still exceeds max_llm_seq_len
    # we drop demonstrations from the front rather than the test question.
    tokenizer_llm.truncation_side = "left"

    tokenizer_mt = NllbTokenizer.from_pretrained(mt_path, local_files_only=local_files_only)

    logger.info(
        "LLM tokenizer special tokens: bos=%s, eos=%s, pad=%s",
        tokenizer_llm.bos_token_id, tokenizer_llm.eos_token_id, tokenizer_llm.pad_token_id,
    )

    sig = inspect.signature(AugmentedMindMerger.__init__)
    extra_kwargs = {}
    if "local_files_only" in sig.parameters:
        extra_kwargs["local_files_only"] = local_files_only

    model = AugmentedMindMerger(
        mt_path=mt_path,
        llm_path=llm_path,
        max_gen_len=max_gen_len,
        llm_bos_token_id=tokenizer_llm.bos_token_id,
        llm_pad_token_id=tokenizer_llm.pad_token_id,
        **extra_kwargs,
    )

    ckpt = torch.load(mapping_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.mapping.load_state_dict(state_dict, strict=False)
    logger.info("Loaded Stage 2 mapping from: %s", mapping_ckpt)
    if missing:
        logger.warning("  missing keys (%d): %s%s",
                       len(missing), missing[:5], " ..." if len(missing) > 5 else "")
    if unexpected:
        logger.warning("  unexpected keys (%d): %s%s",
                       len(unexpected), unexpected[:5], " ..." if len(unexpected) > 5 else "")

    logger.debug(
        "Checkpoint meta: step=%s, loss=%s | end_boundary_norm=%.4f, linear1_w_norm=%.4f",
        ckpt.get("step") if isinstance(ckpt, dict) else "n/a",
        ckpt.get("loss") if isinstance(ckpt, dict) else "n/a",
        float(model.mapping.end_boundary.detach().float().norm().item()),
        float(model.mapping.mlp.linear1.weight.detach().float().norm().item()),
    )

    model.model_mt.to(device)
    model.model_llm.to(device)
    model.mapping.to(device)
    model.eval()

    # Log a few rows of training data to help diagnose label-format mismatches.
    _peek_training_data(logger)

    results: dict[str, float] = {}
    total_correct_all = 0
    total_all = 0
    rng = random.Random(FEWSHOT_SEED)

    for lang in langs:
        if lang not in MMLU_TO_SOURCE_LANGUAGE:
            raise ValueError(f"Unsupported language {lang!r}; add to MMLU_TO_SOURCE_LANGUAGE.")
        source_language = MMLU_TO_SOURCE_LANGUAGE[lang]

        test_ds = load_dataset("li-lab/MMLU-ProX", lang, split="test",
                               download_mode="reuse_dataset_if_exists")
        if max_test_examples is not None:
            test_ds = test_ds.select(range(min(max_test_examples, len(test_ds))))

        val_ds = load_dataset("li-lab/MMLU-ProX", lang, split="validation",
                              download_mode="reuse_dataset_if_exists")
        if max_val_examples is not None:
            val_ds = val_ds.select(range(min(max_val_examples, len(val_ds))))

        idxs = list(range(len(val_ds)))
        rng.shuffle(idxs)
        demo_samples = [val_ds[i] for i in idxs[:N_SHOT]]

        correct = 0
        total = len(test_ds)
        logger.info("Evaluating language: %s (%s, %d examples) with %d-shot",
                    lang, source_language, total, N_SHOT)

        debug_rows: list[tuple[str, str, str, str, bool]] = []
        for sample_idx, sample in enumerate(tqdm(test_ds)):
            prompt, choice_letters = build_fewshot_prompt(demo_samples, sample)
            nllb_text = build_nllb_test_text(sample)
            pred, raw_out = pick_choice(
                model, tokenizer_mt, tokenizer_llm,
                prompt, nllb_text, source_language,
                max_mt_seq_len, max_llm_seq_len,
                choice_letters, amp_dtype, device,
            )
            ok = pred == sample["answer"]
            if ok:
                correct += 1
            if print_sample_count > 0 and len(debug_rows) < print_sample_count:
                debug_rows.append((prompt, raw_out, pred, sample["answer"], ok))

            # Log first 8 samples per language at DEBUG level for diagnosis.
            if sample_idx < 8:
                logger.debug(
                    "lang=%s sample=%d | nllb_text_head=%r | raw_out=%r | pred=%s | target=%s | ok=%s",
                    lang, sample_idx, nllb_text[:200], raw_out, pred, sample.get("answer"), ok,
                )

        if debug_rows:
            logger.info("=== Printed examples (n=%d) for MMLU-ProX lang=%s ===", len(debug_rows), lang)
            for i, (prompt, raw_out, pred, target, correct_i) in enumerate(debug_rows, 1):
                logger.info(
                    "--- example %d/%d ---\nINPUT:\n%s\n\nOUTPUT (raw): %r\n"
                    "PREDICTED: %s  TARGET: %s  CORRECT: %s",
                    i, len(debug_rows), prompt, raw_out, pred, target, correct_i,
                )

        acc = correct / total * 100
        results[lang] = acc
        total_correct_all += correct
        total_all += total
        logger.info("Accuracy for %s: %.2f%%", lang, acc)

    macro_avg = sum(results.values()) / len(results)
    micro_avg = total_correct_all / total_all * 100
    logger.info("=== MMLU-ProX Summary (Stage 2 / AugmentedMindMerger) ===")
    logger.info("Macro-average over %d languages: %.2f%%", len(results), macro_avg)
    logger.info("Micro-average over all examples: %.2f%%", micro_avg)
    return results, macro_avg, micro_avg


def _peek_training_data(logger: logging.Logger) -> None:
    """Log the first 5 rows of task_specialization_en.jsonl at DEBUG level.

    Useful for diagnosing label-format mismatches between training data
    and the MCQ evaluation format.

    Args:
        logger: Logger to which debug lines are written.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    peek_path = os.path.join(script_dir, "data", "task_specialization_en.jsonl")
    if not os.path.isfile(peek_path):
        return
    try:
        with open(peek_path, "r", encoding="utf-8") as f:
            for i, raw in enumerate(f):
                if i >= 5:
                    break
                raw = raw.strip().lstrip("﻿")
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                    logger.debug(
                        "task_specialization_en[%d]: query_head=%r answer=%r source_dataset=%s",
                        i, (row.get("query") or "")[:100],
                        (row.get("answer") or "")[:100],
                        row.get("source_dataset"),
                    )
                except Exception:
                    logger.debug("task_specialization_en[%d]: parse error, raw=%r", i, raw[:100])
    except Exception as exc:
        logger.debug("Could not peek at training data: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run Stage 2 evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 AugmentedMindMerger on MMLU-ProX.")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--mt-path", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument(
        "--mapping-ckpt", type=str,
        default="Stage2/outputs/augmentation/mapping/pytorch_model.bin",
    )
    parser.add_argument("--langs", nargs="*", default=["sw", "wo", "yo"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    parser.add_argument(
        "--max-seq-len", type=int, default=256,
        help="Max sequence length for the NLLB encoder (X_m).",
    )
    parser.add_argument(
        "--max-llm-seq-len", type=int, default=4096,
        help="Max sequence length for the Qwen3-VL prompt (T); truncation_side='left' is used.",
    )
    parser.add_argument("--max-gen-len", type=int, default=256)
    parser.add_argument(
        "--print-sample-count", type=int, default=20,
        help="Print this many (input, raw output, target) triples per language; 0 disables.",
    )
    args = parser.parse_args()

    logger = setup_logging(os.path.join(os.path.dirname(__file__), "logs"))

    max_test = args.max_test_examples
    max_val = args.max_val_examples
    if args.smoke:
        args.langs = ["sw"]
        max_test = 5 if max_test is None else max_test
        max_val = 20 if max_val is None else max_val

    evaluate(
        llm_path=args.llm_path,
        mt_path=args.mt_path,
        mapping_ckpt=args.mapping_ckpt,
        langs=args.langs,
        local_files_only=args.local_files_only,
        max_mt_seq_len=args.max_seq_len,
        max_llm_seq_len=args.max_llm_seq_len,
        max_gen_len=args.max_gen_len,
        max_test_examples=max_test,
        max_val_examples=max_val,
        print_sample_count=args.print_sample_count,
        logger=logger,
    )


if __name__ == "__main__":
    main()
