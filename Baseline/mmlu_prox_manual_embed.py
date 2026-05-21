"""Baseline MMLU-ProX evaluation using manually constructed embeddings.

Identical to mmlu_prox.py except that pick_choice manually extracts token
embeddings via model.get_input_embeddings() and passes inputs_embeds= to
model.generate rather than passing input_ids= directly.

This mirrors what Stage 1 and Stage 2 do when they call
    self.llm_embedding_layer(token_ids)
and allows verifying that the two paths are equivalent.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import string
from datetime import datetime

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

LANGS = [
    "en", "zh", "ja", "ko", "fr", "de", "es", "pt", "ar", "th", "hi", "bn", "sw", "af", "cs",
    "hu", "id", "it", "mr", "ne", "ru", "sr", "te", "uk", "ur", "vi", "wo", "yo", "zu",
]

TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 40
PRESENCE_PENALTY = 2.0
N_SHOT = 5
FEWSHOT_SEED = 1234
MAX_NEW_TOKENS_FOR_MCQ = 6


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"baseline_manual_embed_eval_{timestamp}.log")

    logger = logging.getLogger("baseline_manual_embed_eval")
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
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_id: str,
    local_files_only: bool,
) -> tuple[Qwen3VLForConditionalGeneration, AutoTokenizer, torch.device]:
    assert torch.cuda.is_available(), "CUDA not available – did you request a GPU?"
    device = torch.device("cuda")
    cap_major = torch.cuda.get_device_capability(0)[0]
    dtype = torch.bfloat16 if cap_major >= 8 else torch.float16

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model.eval()
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Presence penalty logits processor
# ---------------------------------------------------------------------------

class PresencePenaltyGeneratedOnly(LogitsProcessor):
    """Subtract a presence penalty from logits of tokens seen in the generated suffix.

    When inputs_embeds is used, input_ids inside generate starts empty and
    accumulates only newly generated tokens, so prompt_len should be 0.
    """

    def __init__(self, penalty: float, prompt_len: int) -> None:
        self.penalty = float(penalty)
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0:
            return scores
        if input_ids.size(1) <= self.prompt_len:
            return scores
        gen_part = input_ids[:, self.prompt_len:]
        for b in range(input_ids.size(0)):
            seen = torch.unique(gen_part[b])
            scores[b, seen] -= self.penalty
        return scores


# ---------------------------------------------------------------------------
# Prompt construction helpers (identical to mmlu_prox.py)
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
    return "\n".join(f"{L}. {t}" for L, t in zip(letters, texts))


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


def build_fewshot_prompt(
    lang: str,
    demo_samples: list[dict],
    test_sample: dict,
) -> tuple[str, list[str]]:
    blocks = []
    for s in demo_samples:
        letters, texts = extract_options(s)
        blocks.append(qwen_eval_block(s["question"], format_options_block(letters, texts), s["answer"]))

    letters, texts = extract_options(test_sample)
    blocks.append(qwen_eval_block(test_sample["question"], format_options_block(letters, texts)))
    return "\n\n".join(blocks), letters


# ---------------------------------------------------------------------------
# Inference — manual embedding construction
# ---------------------------------------------------------------------------

@torch.inference_mode()
def pick_choice(
    prompt: str,
    choices: list[str],
    *,
    model: Qwen3VLForConditionalGeneration,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> str:
    """Generate a single MCQ answer using manually extracted token embeddings.

    Applies the same chat template as mmlu_prox.py, then manually calls
    model.get_input_embeddings() to convert token IDs to embeddings before
    passing inputs_embeds= to model.generate.  This replicates what Stage 1
    and Stage 2 do with self.llm_embedding_layer(token_ids).

    When inputs_embeds is used, HF generate returns only the newly generated
    token IDs (the prompt is not reconstructed from embeddings), so no
    prompt-length slicing is needed when decoding.  The presence penalty
    prompt_len is 0 for the same reason: input_ids inside the logits
    processor accumulates only generated tokens.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # Manual embedding construction — equivalent to Stage 1/2's llm_embedding_layer call
    inputs_embeds = model.get_input_embeddings()(enc["input_ids"])

    # prompt_len=0: with inputs_embeds, input_ids in the logits processor
    # starts empty and only ever contains generated tokens
    logits_processor = LogitsProcessorList(
        [PresencePenaltyGeneratedOnly(PRESENCE_PENALTY, prompt_len=0)]
    )
    generated_ids = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=enc["attention_mask"],
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_new_tokens=MAX_NEW_TOKENS_FOR_MCQ,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    out_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    for ch in out_text:
        if ch in choices:
            return ch
    return choices[0]


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def evaluate(
    *,
    model_id: str,
    langs: list[str],
    local_files_only: bool,
    max_test_examples: int | None,
    max_val_examples: int | None,
    logger: logging.Logger,
) -> tuple[dict[str, float], float, float]:
    model, tokenizer, device = load_model(model_id, local_files_only=local_files_only)
    rng = random.Random(FEWSHOT_SEED)
    results: dict[str, float] = {}
    total_correct_all = 0
    total_all = 0

    for lang in langs:
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
        logger.info("Evaluating language: %s (%d examples) with %d-shot", lang, total, N_SHOT)

        for sample in tqdm(test_ds):
            prompt, choices = build_fewshot_prompt(lang, demo_samples, sample)
            pred = pick_choice(prompt, choices, model=model, tokenizer=tokenizer, device=device)
            if pred == sample["answer"]:
                correct += 1

        acc = correct / total * 100
        results[lang] = acc
        total_correct_all += correct
        total_all += total
        logger.info("Accuracy for %s: %.2f%%", lang, acc)

    macro_avg = sum(results.values()) / len(results)
    micro_avg = total_correct_all / total_all * 100
    logger.info("=== MMLU-ProX Summary (manual embeddings) ===")
    logger.info("Macro-average over %d languages: %.2f%%", len(results), macro_avg)
    logger.info("Micro-average over all examples: %.2f%%", micro_avg)
    return results, macro_avg, micro_avg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline MMLU-ProX evaluation with manually constructed embeddings."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--langs", nargs="*", default=None,
                        help="Override language list (e.g. --langs sw wo yo).")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a tiny subset for a fast end-to-end check.")
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    args = parser.parse_args()

    _logger = setup_logging(os.path.join(os.path.dirname(__file__), "logs"))

    langs = args.langs if args.langs is not None else LANGS
    max_test = args.max_test_examples
    max_val = args.max_val_examples
    if args.smoke:
        langs = ["en"]
        max_test = 5 if max_test is None else max_test
        max_val = 20 if max_val is None else max_val

    results, macro_avg, micro_avg = evaluate(
        model_id=args.model_id,
        langs=langs,
        local_files_only=args.local_files_only,
        max_test_examples=max_test,
        max_val_examples=max_val,
        logger=_logger,
    )
    _logger.info("=== Per-language Accuracies ===")
    for lang, acc in results.items():
        _logger.info("%s: %.2f%%", lang, acc)
