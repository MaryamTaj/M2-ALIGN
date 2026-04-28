# coding=utf-8
"""
Evaluate the Stage 2 augmented MindMerger checkpoint on MMLU-ProX.

Unlike `Stage1/run_evaluating.py`, this script feeds the LLM exactly the
prefix it was trained on at Stage 2:

    [BOS] + X_m + [end_boundary] + T

where
    X_m = mapping(NLLB_encoder(prompt_in_source_language))
    T   = LLM_token_embedding(prompt)        # same prompt, LLM-tokenised

This matches the augmentation stage of the MindMerger paper (NeurIPS 2024)
and avoids the train/eval mismatch that occurs when Stage 1's eval path
(which omits T) is used with a Stage 2 checkpoint.

Decoding hyperparameters are kept consistent with Baseline/mmlu_prox.py and
Stage1/run_evaluating.py for direct comparability.
"""
from __future__ import annotations

import argparse
import inspect
import random
import string
from typing import Iterable

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, NllbTokenizer

from modeling_augmentation import AugmentedMindMerger


TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 40
PRESENCE_PENALTY = 2.0
N_SHOT = 5
FEWSHOT_SEED = 1234
MAX_NEW_TOKENS_FOR_MCQ = 6


# MMLU-ProX config name -> human-readable language key used by the NLLB lang map.
MMLU_TO_SOURCE_LANGUAGE = {
    "sw": "Swahili",
    "wo": "Wolof",
    "yo": "Yoruba",
}

NLLB_LANG_MAP = {
    "English": "eng_Latn",
    "Swahili": "swh_Latn",
    "Yoruba": "yor_Latn",
    "Wolof": "wol_Latn",
}


# ---------------------------------------------------------------------------
# Prompt construction (kept in sync with Stage1/run_evaluating.py).
# ---------------------------------------------------------------------------

def extract_options(sample: dict) -> tuple[list[str], list[str]]:
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
    return "\n".join([f"{L}. {t}" for L, t in zip(letters, texts)])


def qwen_eval_block(question: str, options_block: str, answer_letter: str | None = None) -> str:
    base = (
        "Respond with only the letter of the correct option.\n"
        f"Question: {question} Possible answer choices:\n"
        f"{options_block}\n"
        "The best answer is:"
    )
    if answer_letter is not None:
        return base + f" {answer_letter}\n"
    return base


def build_fewshot_prompt(demo_samples: list, test_sample: dict) -> tuple[str, list[str]]:
    blocks = []
    for s in demo_samples:
        letters, texts = extract_options(s)
        options_block = format_options_block(letters, texts)
        blocks.append(qwen_eval_block(s["question"], options_block, s["answer"]))

    letters, texts = extract_options(test_sample)
    options_block = format_options_block(letters, texts)
    blocks.append(qwen_eval_block(test_sample["question"], options_block, answer_letter=None))
    return "\n\n".join(blocks), letters


# ---------------------------------------------------------------------------
# Tokenisation helpers (kept in sync with Stage2/run_augmentation.py so that
# eval-time token streams match what the model saw during training).
# ---------------------------------------------------------------------------

def mt_input_features(
    texts: Iterable[str],
    langs: Iterable[str],
    tokenizer_mt: NllbTokenizer,
    max_seq_len: int,
    device: torch.device,
):
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
):
    # Match the training-time helper exactly so that `T` is identical to the
    # token stream the mapping was trained against.
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
# Per-example MCQ scoring.
# ---------------------------------------------------------------------------

@torch.inference_mode()
def pick_choice(
    model: AugmentedMindMerger,
    tokenizer_mt: NllbTokenizer,
    tokenizer_llm: AutoTokenizer,
    prompt: str,
    source_language: str,
    max_seq_len: int,
    choices: list[str],
    amp_dtype: torch.dtype,
    device: torch.device,
) -> str:
    input_ids_mt, mask_mt = mt_input_features(
        [prompt], [source_language], tokenizer_mt, max_seq_len, device
    )
    input_ids_query_llm, mask_query_llm = llm_input_features(
        [prompt], tokenizer_llm, max_seq_len, add_bos=False, add_eos=False, device=device
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
            return ch
    return choices[0]


# ---------------------------------------------------------------------------
# Evaluation driver.
# ---------------------------------------------------------------------------

def evaluate(
    *,
    llm_path: str,
    mt_path: str,
    mapping_ckpt: str,
    langs: list[str],
    local_files_only: bool,
    max_seq_len: int,
    max_gen_len: int,
    max_test_examples: int | None,
    max_val_examples: int | None,
):
    device = torch.device("cuda")
    cap_major = torch.cuda.get_device_capability(0)[0]
    amp_dtype = torch.bfloat16 if cap_major >= 8 else torch.float16

    tokenizer_llm = AutoTokenizer.from_pretrained(
        llm_path, use_fast=False, local_files_only=local_files_only
    )
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"

    tokenizer_mt = NllbTokenizer.from_pretrained(mt_path, local_files_only=local_files_only)

    # Stay forward-compatible with model defs that may not yet accept
    # `local_files_only` as a keyword argument.
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
    print(f"Loaded Stage 2 mapping from: {mapping_ckpt}")
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    model.model_mt.to(device)
    model.model_llm.to(device)
    model.mapping.to(device)
    model.eval()

    results: dict[str, float] = {}
    total_correct_all = 0
    total_all = 0
    rng = random.Random(FEWSHOT_SEED)

    for lang in langs:
        if lang not in MMLU_TO_SOURCE_LANGUAGE:
            raise ValueError(f"Unsupported language {lang!r}; add to MMLU_TO_SOURCE_LANGUAGE.")

        source_language = MMLU_TO_SOURCE_LANGUAGE[lang]

        test_ds = load_dataset(
            "li-lab/MMLU-ProX",
            lang,
            split="test",
            download_mode="reuse_dataset_if_exists",
        )
        if max_test_examples is not None:
            test_ds = test_ds.select(range(min(max_test_examples, len(test_ds))))

        val_ds = load_dataset(
            "li-lab/MMLU-ProX",
            lang,
            split="validation",
            download_mode="reuse_dataset_if_exists",
        )
        if max_val_examples is not None:
            val_ds = val_ds.select(range(min(max_val_examples, len(val_ds))))

        idxs = list(range(len(val_ds)))
        rng.shuffle(idxs)
        demo_samples = [val_ds[i] for i in idxs[:N_SHOT]]

        correct = 0
        total = len(test_ds)
        print(f"\nEvaluating language: {lang} ({source_language}, {total} examples) with {N_SHOT}-shot")

        for sample in tqdm(test_ds):
            prompt, choice_letters = build_fewshot_prompt(demo_samples, sample)
            pred = pick_choice(
                model,
                tokenizer_mt,
                tokenizer_llm,
                prompt,
                source_language,
                max_seq_len,
                choice_letters,
                amp_dtype,
                device,
            )
            if pred == sample["answer"]:
                correct += 1

        acc = correct / total * 100
        results[lang] = acc
        total_correct_all += correct
        total_all += total
        print(f"Accuracy for {lang}: {acc:.2f}%")

    macro_avg = sum(results.values()) / len(results)
    micro_avg = (total_correct_all / total_all) * 100
    print("\n=== MMLU-ProX Summary (Stage 2 / AugmentedMindMerger) ===")
    print(f"Macro-average over {len(results)} languages: {macro_avg:.2f}%")
    print(f"Micro-average over all examples: {micro_avg:.2f}%")
    return results, macro_avg, micro_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HF id or local snapshot directory for Qwen3-VL.",
    )
    parser.add_argument(
        "--mt-path",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="HF id or local snapshot directory for the NLLB encoder.",
    )
    parser.add_argument(
        "--mapping-ckpt",
        type=str,
        default="Stage2/outputs/augmentation/mapping/pytorch_model.bin",
        help="Path to the Stage 2 mapping checkpoint (pytorch_model.bin).",
    )
    parser.add_argument("--langs", nargs="*", default=["sw", "wo", "yo"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-gen-len", type=int, default=256)
    args = parser.parse_args()

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
        max_seq_len=args.max_seq_len,
        max_gen_len=args.max_gen_len,
        max_test_examples=max_test,
        max_val_examples=max_val,
    )


if __name__ == "__main__":
    main()
