# coding=utf-8
"""
Evaluate MindMerger (NLLB encoder + mapping + Qwen3-VL) on MMLU-ProX for selected languages.

Prompt format and decoding hyperparameters match Baseline/mmlu_prox.py for comparability.
The multilingual MCQ text is encoded with the NLLB tokenizer (src_lang set per language);
the LLM conditions on mapped encoder states + boundary (same path as Stage 1 inference).
"""
from __future__ import annotations

import argparse
import random
import string

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from modeling_mindmerger import MindMerger
from tools.input_features import mt_input_features

# Match Baseline/mmlu_prox.py
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 40
PRESENCE_PENALTY = 2.0
N_SHOT = 5
FEWSHOT_SEED = 1234
MAX_NEW_TOKENS_FOR_MCQ = 6

# MMLU-ProX config name -> Stage1 mt_input_features language key (must match run_training.py NLLB map)
MMLU_TO_SOURCE_LANGUAGE = {
    "sw": "Swahili",
    "wo": "Wolof",
    "yo": "Yoruba",
}

LANGS_MAP_NLLB = {
    "Swahili": "swh_Latn",
    "Yoruba": "yor_Latn",
    "Wolof": "wol_Latn",
}


def extract_options(sample):
    option_items = []
    for k, v in sample.items():
        if k.startswith("option_") and v is not None:
            idx = int(k.split("_")[1])
            option_items.append((idx, v))
    option_items.sort(key=lambda x: x[0])
    texts = [v for _, v in option_items]
    letters = list(string.ascii_uppercase[: len(texts)])
    return letters, texts


def format_options_block(letters, texts) -> str:
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


@torch.inference_mode()
def pick_choice(
    model: MindMerger,
    tokenizer_mt: AutoTokenizer,
    tokenizer_llm,
    prompt: str,
    source_language: str,
    langs_map: dict,
    max_seq_len: int,
    choices: list[str],
    amp_dtype: torch.dtype,
) -> str:
    input_ids_m2m, attention_mask_m2m = mt_input_features(
        [prompt], tokenizer_mt, max_seq_len, [source_language], langs_map
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
        out_text = model.generate_from_mt(
            input_ids_m2m,
            attention_mask_m2m,
            tokenizer_llm,
            generation_kwargs=gen_kw,
            presence_penalty=PRESENCE_PENALTY,
        )
    for ch in out_text:
        if ch in choices:
            return ch
    return choices[0]


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

    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path, local_files_only=local_files_only)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"

    tokenizer_mt = AutoTokenizer.from_pretrained(mt_path, local_files_only=local_files_only)

    model = MindMerger(
        mt_path,
        llm_path,
        max_gen_len,
        tokenizer_llm.bos_token_id,
        tokenizer_llm.pad_token_id,
        local_files_only=local_files_only,
    )
    ckpt = torch.load(mapping_ckpt, map_location="cpu")
    model.mapping.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("Loaded mapping from:", mapping_ckpt)

    model.model_mt.to(device)
    model.mapping.to(device)
    model.eval()

    results = {}
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
                LANGS_MAP_NLLB,
                max_seq_len,
                choice_letters,
                amp_dtype,
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
    print("\n=== MMLU-ProX Summary (MindMerger) ===")
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
        help="HF id or local snapshot directory for NLLB encoder.",
    )
    parser.add_argument(
        "--mapping-ckpt",
        type=str,
        default="./outputs/M2Align/translation/mapping/pytorch_model.bin",
        help="Path to trained mapping checkpoint (pytorch_model.bin).",
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
