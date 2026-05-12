# coding=utf-8
"""Stage 1 evaluation utilities and MMLU-ProX evaluation script.

Training-time helpers
---------------------
``evaluate_ppl``, ``evaluate_math``, ``evaluate_classification`` are imported
by ``train.py`` during the Stage 1 training loop.

MMLU-ProX CLI evaluation
------------------------
Run directly to evaluate a trained Stage 1 MindMerger checkpoint on MMLU-ProX.
Prompt format and decoding hyperparameters match ``Baseline/mmlu_prox.py`` for
comparability.  The multilingual MCQ text is encoded with the NLLB tokenizer
(``src_lang`` set per language); the LLM conditions on mapped encoder states
plus the boundary token (same prefix as Stage 1 training).
"""
from __future__ import annotations

import argparse
import inspect
import logging
import math
import os
import random
import string
from datetime import datetime

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model import MindMerger
from tools.input_features import llm_input_features, mt_input_features
from tools.utils import extract_last_num

# Match Baseline/mmlu_prox.py decoding hyperparameters.
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 40
PRESENCE_PENALTY = 2.0
N_SHOT = 5
FEWSHOT_SEED = 1234
MAX_NEW_TOKENS_FOR_MCQ = 6

# MMLU-ProX config name → Stage1 mt_input_features language key.
MMLU_TO_SOURCE_LANGUAGE = {
    "sw": "Swahili",
    "wo": "Wolof",
    "yo": "Yoruba",
    "fr": "French",
}

LANGUAGE_NAME_TO_MMLU_CODE = {
    "swahili": "sw",
    "wolof": "wo",
    "yoruba": "yo",
    "french": "fr",
}

LANGS_MAP_NLLB = {
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

    logger = logging.getLogger("stage1_eval")
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
# Training-time evaluation utilities (used by train.py)
# ---------------------------------------------------------------------------

def evaluate_math(
    model,
    test_set,
    tokenizer_llm,
    tokenizer_mt,
    max_seq_len: int,
    max_gen_len: int,
    use_prompt: bool,
    langs_map: dict[str, str],
) -> tuple[float, list[dict]]:
    """Evaluate a MindMerger model on a math generation task.

    Generates answers for every batch in *test_set* and compares the last
    number in the generation to the last number in the ground-truth target.

    Args:
        model: A trained :class:`MindMerger` instance.
        test_set: DataLoader yielding batches with keys ``"source"``,
            ``"prompt"``, ``"target"``, and ``"source_language"``.
        tokenizer_llm: LLM tokenizer used for decoding outputs.
        tokenizer_mt: MT tokenizer (NLLB/M2M) used for encoding inputs.
        max_seq_len: Maximum sequence length for the MT tokenizer.
        max_gen_len: Maximum number of tokens the LLM may generate.
        use_prompt: If ``True``, also encode the prompt with the LLM and
            pass it as a conditioning prefix.
        langs_map: Mapping from human-readable language name to the MT
            language tag.

    Returns:
        A 2-tuple ``(accuracy, results_list)`` where *accuracy* is the
        percentage of examples whose extracted numeric answer matches the
        ground truth, and *results_list* is a list of per-example dicts
        with keys ``"question"``, ``"answer"``, ``"prompt"``,
        ``"prediction"``, and ``"output"``.
    """
    model.eval()
    results_list = []
    hit = 0
    step_trange = tqdm(test_set)
    for batch in step_trange:
        sources = batch["source"]
        prompts = batch["prompt"]
        targets = batch["target"]
        source_languages = batch["source_language"]

        input_ids_m2m, attention_mask_m2m = mt_input_features(
            sources, tokenizer_mt, max_seq_len, source_languages, langs_map
        )
        input_ids_prompt, mask_prompt = None, None
        if use_prompt:
            input_ids_prompt, mask_prompt = llm_input_features(
                prompts, tokenizer_llm, max_gen_len, add_bos_token=False, add_eos_token=False
            )

        generate_ids = model(
            input_ids_m2m, attention_mask_m2m,
            input_ids_prompt=input_ids_prompt, mask_prompt=mask_prompt,
        )
        results = tokenizer_llm.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for result, prompt, source, target in zip(results, prompts, sources, targets):
            answer = extract_last_num(target)
            pred = extract_last_num(result)
            results_list.append({
                "question": source,
                "answer": answer,
                "prompt": prompt,
                "prediction": str(pred),
                "output": result,
            })
            if float(answer) == float(pred):
                hit += 1

        acc = round(hit / len(results_list) * 100, 2)
        step_trange.set_postfix_str(f"Acc:{acc}")

    return round(hit / len(results_list) * 100, 2), results_list


def evaluate_classification(
    model,
    test_set,
    tokenizer_llm,
    tokenizer_mt,
    max_seq_len: int,
    max_gen_len: int,
    use_prompt: bool,
    langs_map: dict[str, str],
) -> tuple[float, list[dict]]:
    """Evaluate a MindMerger model on a classification task (exact-match).

    Generates text for every batch and compares the stripped output to the
    target label via exact string match.

    Args:
        model: A trained :class:`MindMerger` instance.
        test_set: DataLoader yielding batches with keys ``"prompt"``,
            ``"target"``, and ``"source_language"``.
        tokenizer_llm: LLM tokenizer used for decoding outputs.
        tokenizer_mt: MT tokenizer used for encoding inputs.
        max_seq_len: Maximum sequence length for the MT tokenizer.
        max_gen_len: Maximum number of tokens the LLM may generate.
        use_prompt: If ``True``, also encode the prompt with the LLM.
        langs_map: Mapping from human-readable language name to MT tag.

    Returns:
        A 2-tuple ``(accuracy, results_list)`` where *accuracy* is the
        percentage of exact-match hits, and *results_list* is a list of
        per-example dicts with keys ``"prompt"``, ``"prediction"``, and
        ``"answer"``.
    """
    model.eval()
    results_list = []
    hit = 0
    step_trange = tqdm(test_set)
    for batch in step_trange:
        prompts = batch["prompt"]
        targets = batch["target"]
        source_languages = batch["source_language"]

        input_ids_m2m, attention_mask_m2m = mt_input_features(
            prompts, tokenizer_mt, max_seq_len, source_languages, langs_map
        )
        input_ids_prompt, mask_prompt = None, None
        if use_prompt:
            input_ids_prompt, mask_prompt = llm_input_features(
                prompts, tokenizer_llm, max_gen_len, add_bos_token=False, add_eos_token=False
            )

        generate_ids = model(
            input_ids_m2m, attention_mask_m2m,
            input_ids_prompt=input_ids_prompt, mask_prompt=mask_prompt,
        )
        results = tokenizer_llm.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for result, prompt, target in zip(results, prompts, targets):
            result = result.strip()
            results_list.append({"prompt": prompt, "prediction": result, "answer": target})
            if target == result:
                hit += 1

        acc = round(hit / len(results_list) * 100, 2)
        step_trange.set_postfix_str(f"Acc:{acc}")

    return round(hit / len(results_list) * 100, 2), results_list


def evaluate_ppl(
    model,
    test_set,
    tokenizer_llm,
    tokenizer_mt,
    max_seq_len: int,
    max_gen_len: int,
    langs_map: dict[str, str],
    use_prompt: bool,
) -> float:
    """Compute perplexity of a MindMerger model on a held-out set.

    Runs the model in evaluation mode, accumulates the mean cross-entropy
    loss, and returns ``exp(mean_loss)`` as the perplexity.  Restores the
    model to training mode and clears the GPU cache before returning.

    Args:
        model: A trained :class:`MindMerger` instance.
        test_set: DataLoader yielding batches with keys ``"source"``,
            ``"prompt"``, ``"target"``, and ``"source_language"``.
        tokenizer_llm: LLM tokenizer used for label tokenisation.
        tokenizer_mt: MT tokenizer used for source tokenisation.
        max_seq_len: Maximum sequence length for the MT tokenizer.
        max_gen_len: Maximum sequence length for LLM labels.
        langs_map: Mapping from human-readable language name to MT tag.
        use_prompt: If ``True``, also encode the prompt with the LLM.

    Returns:
        Perplexity (a float ≥ 1.0).
    """
    model.eval()
    loss_all = 0.0
    step_i = 0
    step_trange = tqdm(test_set)
    for batch in step_trange:
        step_i += 1
        sources = batch["source"]
        prompts = batch["prompt"]
        targets = batch["target"]
        source_languages = batch["source_language"]

        input_ids_m2m, attention_mask_m2m = mt_input_features(
            sources, tokenizer_mt, max_seq_len, source_languages, langs_map
        )
        labels, mask_label = llm_input_features(
            targets, tokenizer_llm, max_gen_len, add_bos_token=False, add_eos_token=True
        )

        input_ids_prompt, mask_prompt = None, None
        if use_prompt:
            input_ids_prompt, mask_prompt = llm_input_features(
                prompts, tokenizer_llm, max_gen_len, add_bos_token=False, add_eos_token=False
            )

        loss = model(
            input_ids_m2m, attention_mask_m2m,
            input_ids_prompt=input_ids_prompt, mask_prompt=mask_prompt,
            labels=labels, mask_label=mask_label,
        )
        loss_all += loss.mean().item()
        step_trange.set_postfix_str(f"loss:{round(loss_all / step_i, 4)}")

    perplexity = math.exp(loss_all / step_i)
    model.train()
    torch.cuda.empty_cache()
    return perplexity


# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------

def extract_options(sample: dict) -> tuple[list[str], list[str]]:
    """Extract answer options from an MMLU-ProX sample dict.

    Reads all keys of the form ``"option_N"`` (where N is an integer),
    sorts them by N, and returns parallel lists of option letters and texts.

    Args:
        sample: An MMLU-ProX sample dict with keys like ``"option_1"``,
            ``"option_2"``, etc.

    Returns:
        A 2-tuple ``(letters, texts)`` where *letters* is a list of
        uppercase ASCII letters (``["A", "B", ...]``) and *texts* is the
        corresponding list of option strings.
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
    """Format option letters and texts into a multi-line string.

    Args:
        letters: List of option letters, e.g. ``["A", "B", "C", "D"]``.
        texts: Corresponding option texts.

    Returns:
        A string with one ``"L. text"`` line per option.

    Examples:
        >>> format_options_block(["A", "B"], ["Yes", "No"])
        'A. Yes\\nB. No'
    """
    return "\n".join(f"{L}. {t}" for L, t in zip(letters, texts))


def qwen_eval_block(question: str, options_block: str, answer_letter: str | None = None) -> str:
    """Build a single MCQ evaluation block in Qwen3-VL style.

    Constructs the ``"Respond with only the letter …"`` prompt template used
    throughout this project.  Appends *answer_letter* for few-shot demos;
    leaves the prompt open-ended for the test question.

    Args:
        question: The question text.
        options_block: Pre-formatted option lines (from
            :func:`format_options_block`).
        answer_letter: The correct letter to append (for demos), or
            ``None`` to leave the block open for the model to complete.

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
        demo_samples: List of few-shot demonstration sample dicts.
        test_sample: The test question to be answered.

    Returns:
        A 2-tuple ``(prompt, letters)`` where *prompt* is the full
        few-shot string and *letters* is the list of valid answer letters
        for *test_sample*.
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

    Stage 1 has no LLM-side prompt: the model conditions only on the NLLB
    encoder's output.  Feeding NLLB the full 5-shot prompt (mostly English
    scaffolding + English demos) mismatches its ``src_lang`` setting and
    right-truncates the actual test question under ``max_seq_len=256``.
    This function returns only the test question and its options.

    Args:
        test_sample: An MMLU-ProX test sample dict.

    Returns:
        A string containing the question and answer options separated by
        a newline.
    """
    letters, texts = extract_options(test_sample)
    return f"{test_sample['question']}\n{format_options_block(letters, texts)}"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def pick_choice(
    model: MindMerger,
    tokenizer_mt,
    tokenizer_llm,
    nllb_text: str,
    source_language: str,
    langs_map: dict[str, str],
    max_seq_len: int,
    choices: list[str],
    amp_dtype: torch.dtype,
) -> tuple[str, str]:
    """Run one MCQ inference step and return the predicted letter.

    Tokenises *nllb_text* with the MT tokenizer, builds the LLM prefix
    via the mapping layer, and decodes at most ``MAX_NEW_TOKENS_FOR_MCQ``
    new tokens.  The first valid answer letter in the output is returned;
    if none is found, defaults to the first choice.

    Args:
        model: A loaded :class:`MindMerger` instance in eval mode.
        tokenizer_mt: NLLB tokenizer.
        tokenizer_llm: Qwen3-VL tokenizer.
        nllb_text: The target-language question + options string.
        source_language: Human-readable language name (key in *langs_map*).
        langs_map: Mapping from language name to NLLB language tag.
        max_seq_len: Maximum sequence length for the NLLB tokenizer.
        choices: Valid answer letters for this question (e.g. ``["A","B","C","D"]``).
        amp_dtype: Dtype for ``torch.autocast`` (bf16 or fp16).

    Returns:
        A 2-tuple ``(predicted_letter, raw_output_string)``.
    """
    input_ids_m2m, attention_mask_m2m = mt_input_features(
        [nllb_text], tokenizer_mt, max_seq_len, [source_language], langs_map
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
    max_seq_len: int,
    max_gen_len: int,
    max_test_examples: int | None,
    max_val_examples: int | None,
    print_sample_count: int = 20,
    logger: logging.Logger,
) -> tuple[dict[str, float], float, float]:
    """Evaluate a Stage 1 MindMerger checkpoint on MMLU-ProX.

    Loads the model and tokenizers, iterates over *langs*, and computes
    per-language and aggregate accuracy.

    Args:
        llm_path: HF id or local path for Qwen3-VL.
        mt_path: HF id or local path for the NLLB encoder.
        mapping_ckpt: Path to the trained mapping checkpoint
            (``pytorch_model.bin``).
        langs: MMLU-ProX language codes derived from CLI full-name input.
        local_files_only: If ``True``, do not attempt Hub downloads.
        max_seq_len: Maximum NLLB sequence length.
        max_gen_len: Maximum new tokens the LLM may generate.
        max_test_examples: Cap on test examples per language (``None`` = all).
        max_val_examples: Cap on validation examples per language.
        print_sample_count: Number of (prompt, output, answer) triples to
            print per language for debugging (0 disables).
        logger: Logger for progress and result messages.

    Returns:
        A 3-tuple ``(results, macro_avg, micro_avg)`` where *results* maps
        language code → accuracy, and the averages are percentages.
    """
    device = torch.device("cuda")
    cap_major = torch.cuda.get_device_capability(0)[0]
    amp_dtype = torch.bfloat16 if cap_major >= 8 else torch.float16

    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path, local_files_only=local_files_only)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"

    tokenizer_mt = AutoTokenizer.from_pretrained(mt_path, local_files_only=local_files_only)

    mm_sig = inspect.signature(MindMerger.__init__)
    mm_kwargs = {}
    if "local_files_only" in mm_sig.parameters:
        mm_kwargs["local_files_only"] = local_files_only

    model = MindMerger(
        mt_path, llm_path, max_gen_len,
        tokenizer_llm.bos_token_id, tokenizer_llm.pad_token_id,
        **mm_kwargs,
    )
    ckpt = torch.load(mapping_ckpt, map_location="cpu")
    model.mapping.load_state_dict(ckpt["model_state_dict"], strict=False)
    logger.info("Loaded mapping from: %s", mapping_ckpt)

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
        for sample in tqdm(test_ds):
            _, choice_letters = build_fewshot_prompt(demo_samples, sample)
            nllb_text = build_nllb_test_text(sample)
            pred, raw_out = pick_choice(
                model, tokenizer_mt, tokenizer_llm,
                nllb_text, source_language, LANGS_MAP_NLLB,
                max_seq_len, choice_letters, amp_dtype,
            )
            ok = pred == sample["answer"]
            if ok:
                correct += 1
            if print_sample_count > 0 and len(debug_rows) < print_sample_count:
                debug_rows.append((_, raw_out, pred, sample["answer"], ok))

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
    logger.info("=== MMLU-ProX Summary (MindMerger) ===")
    logger.info("Macro-average over %d languages: %.2f%%", len(results), macro_avg)
    logger.info("Micro-average over all examples: %.2f%%", micro_avg)
    return results, macro_avg, micro_avg


def normalize_lang_args(langs_csv: str) -> list[str]:
    """Normalize comma-separated full language names to MMLU-ProX codes.

    The CLI input must be a comma-separated list of full language names, e.g.
    ``Swahili,Yoruba,Wolof``.
    """
    normalized: list[str] = []
    for lang in langs_csv.split(","):
        key = lang.strip()
        if not key:
            continue
        lower = key.lower()
        if lower in LANGUAGE_NAME_TO_MMLU_CODE:
            normalized.append(LANGUAGE_NAME_TO_MMLU_CODE[lower])
            continue
        raise ValueError(
            f"Unsupported language {lang!r}. Use full names only, comma-separated. "
            f"Allowed names: {sorted(LANGUAGE_NAME_TO_MMLU_CODE)}."
        )
    if not normalized:
        raise ValueError("No languages provided. Pass --langs as comma-separated full names.")
    return normalized


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run Stage 1 evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 MindMerger on MMLU-ProX.")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--mt-path", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument(
        "--mapping-ckpt", type=str,
        default="./outputs/M2Align/translation/mapping/pytorch_model.bin",
    )
    parser.add_argument(
        "--langs", type=str, default="Swahili,Wolof,Yoruba",
        help="Comma-separated full language names (e.g. Swahili,Yoruba,Wolof).",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-gen-len", type=int, default=256)
    parser.add_argument(
        "--print-sample-count", type=int, default=20,
        help="Print this many (input, raw output, target letter) triples per language; 0 disables.",
    )
    args = parser.parse_args()

    logger = setup_logging(os.path.join(os.path.dirname(__file__), "logs"))

    max_test = args.max_test_examples
    max_val = args.max_val_examples
    if args.smoke:
        args.langs = "Swahili"
        max_test = 5 if max_test is None else max_test
        max_val = 20 if max_val is None else max_val
    args.langs = normalize_lang_args(args.langs)

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
        print_sample_count=args.print_sample_count,
        logger=logger,
    )


if __name__ == "__main__":
    main()
