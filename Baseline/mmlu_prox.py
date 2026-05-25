"""Baseline MMLU-ProX evaluation using Qwen3-VL directly (no mapping layer)."""
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
    """Create a logger that writes to stdout and a timestamped file in *log_dir*.

    Args:
        log_dir: Directory in which to create the log file (created if absent).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"baseline_eval_{timestamp}.log")

    logger = logging.getLogger("baseline_eval")
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
    """Load Qwen3-VL and its tokenizer onto the available CUDA device.

    Selects BF16 on Ampere+ GPUs, FP16 otherwise.

    Args:
        model_id: Hugging Face model ID or local snapshot directory.
        local_files_only: If ``True``, do not attempt Hub downloads.

    Returns:
        A 3-tuple ``(model, tokenizer, device)`` with the model in eval mode.

    Raises:
        AssertionError: If no CUDA device is available.
    """
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

    HF ``generate`` does not have a built-in presence penalty; this processor
    applies one only to tokens that appeared *after* the prompt (i.e. tokens
    generated so far), not to tokens in the original conditioning prompt.

    Args:
        penalty: Magnitude of the penalty to subtract from repeated logits.
        prompt_len: Number of tokens in the conditioning prefix; positions
            at or before this index are considered part of the prompt.
    """

    def __init__(self, penalty: float, prompt_len: int) -> None:
        self.penalty = float(penalty)
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply the presence penalty and return updated logits.

        Args:
            input_ids: Token ids of shape ``[batch, seq]``.
            scores: Logit tensor of shape ``[batch, vocab_size]``.

        Returns:
            Updated *scores* with the penalty applied to repeated tokens.
        """
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
# Prompt construction helpers
# ---------------------------------------------------------------------------

def extract_options(sample: dict) -> tuple[list[str], list[str]]:
    """Extract answer options from an MMLU-ProX sample dict.

    Reads keys of the form ``"option_N"`` (N integer), sorts by N, and
    returns parallel lists of option letters and option texts.

    Args:
        sample: An MMLU-ProX sample dict.

    Returns:
        A 2-tuple ``(letters, texts)`` of equal-length lists.

    Examples:
        >>> extract_options({"option_1": "Yes", "option_2": "No"})
        (['A', 'B'], ['Yes', 'No'])
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

    Examples:
        >>> format_options_block(["A", "B"], ["Yes", "No"])
        'A. Yes\\nB. No'
    """
    return "\n".join(f"{L}. {t}" for L, t in zip(letters, texts))


def qwen_eval_block(question: str, options_block: str, answer_letter: str | None = None) -> str:
    """Build a single MCQ block in Qwen3-VL evaluation style.

    Constructs the ``"Respond with only the letter …"`` template.  Appends
    *answer_letter* for few-shot demos; leaves the block open for the test
    question.

    Args:
        question: The question text.
        options_block: Pre-formatted option lines from
            :func:`format_options_block`.
        answer_letter: Correct answer letter for a demo, or ``None`` for
            the test question.

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


def build_fewshot_prompt(
    lang: str,
    demo_samples: list[dict],
    test_sample: dict,
) -> tuple[str, list[str]]:
    """Construct an N-shot MCQ prompt for one test example.

    Args:
        lang: MMLU-ProX language code (unused in prompt body, kept for
            caller convenience).
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


# ---------------------------------------------------------------------------
# Inference
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
    """Generate a single MCQ answer letter with the baseline Qwen3-VL model.

    Applies the chat template, generates up to ``MAX_NEW_TOKENS_FOR_MCQ``
    tokens with the presence penalty, and returns the first valid choice
    letter found in the output.  Falls back to the first choice if none is
    found.

    Args:
        prompt: The full N-shot MCQ prompt string.
        choices: Valid answer letters for this question.
        model: Qwen3-VL model in eval mode.
        tokenizer: Corresponding tokenizer.
        device: Device on which *model* lives.

    Returns:
        The predicted answer letter (single uppercase character).
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
    prompt_len = enc["input_ids"].shape[1]

    logits_processor = LogitsProcessorList(
        [PresencePenaltyGeneratedOnly(PRESENCE_PENALTY, prompt_len)]
    )
    generated_ids = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS_FOR_MCQ,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_only = generated_ids[:, prompt_len:]
    out_text = tokenizer.batch_decode(
        gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
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
    """Run the baseline MMLU-ProX evaluation across all *langs*.

    Loads Qwen3-VL once, then evaluates each language with a fixed 5-shot
    prompt sampled from the validation split.

    Args:
        model_id: HF id or local path for Qwen3-VL.
        langs: MMLU-ProX language codes to evaluate.
        local_files_only: If ``True``, do not attempt Hub downloads.
        max_test_examples: Cap on test examples per language (``None`` = all).
        max_val_examples: Cap on validation examples per language.
        logger: Logger for progress and result messages.

    Returns:
        A 3-tuple ``(results, macro_avg, micro_avg)`` where *results* maps
        language code → accuracy and the averages are percentages.
    """
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
    logger.info("=== MMLU-ProX Summary ===")
    logger.info("Macro-average over %d languages: %.2f%%", len(results), macro_avg)
    logger.info("Micro-average over all examples: %.2f%%", micro_avg)
    return results, macro_avg, micro_avg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline MMLU-ProX evaluation with Qwen3-VL.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--langs", nargs="*", default=None,
                        help="Override language list (e.g. --langs en fr).")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Do not download model/dataset artifacts.")
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
