import string
import random
import torch

import argparse
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------
# Config
# ---------------------------
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

LANGS = [
    "en", "zh", "ja", "ko", "fr", "de", "es", "pt", "ar", "th", "hi", "bn", "sw", "af", "cs",
    "hu", "id", "it", "mr", "ne", "ru", "sr", "te", "uk", "ur", "vi", "wo", "yo", "zu"
]

# Paper decoding hyperparams for small instruct models (your target)
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 40
PRESENCE_PENALTY = 2.0

# MMLU-ProX: use 5-shot (validation split is meant for prompt construction)
N_SHOT = 5
FEWSHOT_SEED = 1234

# For MCQ, don't let it ramble: only generate a few tokens (still compatible with paper settings)
MAX_NEW_TOKENS_FOR_MCQ = 6

# If you want deterministic sampling for reproduction runs, you can set:
# torch.manual_seed(FEWSHOT_SEED)
# random.seed(FEWSHOT_SEED)

# ---------------------------
# Load model
# ---------------------------
def load_model(model_id: str, local_files_only: bool):
    assert torch.cuda.is_available(), "CUDA not available - did you request a GPU?"
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
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    tokenizer = processor.tokenizer
    model.eval()
    return model, processor, tokenizer, device

# ---------------------------
# Presence penalty (HF compatible)
# ---------------------------
class PresencePenaltyGeneratedOnly(LogitsProcessor):
    """
    Subtract penalty from logits of tokens that have appeared in the GENERATED portion so far.
    (HF generate() doesn't have presence_penalty built-in in your version.)
    """
    def __init__(self, penalty: float, prompt_len: int):
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

# ---------------------------
# Helpers: options + prompt formatting
# ---------------------------
def extract_options(sample):
    option_items = []
    for k, v in sample.items():
        if k.startswith("option_") and v is not None:
            idx = int(k.split("_")[1])
            option_items.append((idx, v))
    option_items.sort(key=lambda x: x[0])

    texts = [v for _, v in option_items]
    letters = list(string.ascii_uppercase[:len(texts)])  # supports A..J (and beyond if needed)
    return letters, texts

def format_options_block(letters, texts) -> str:
    # Match typical “{options}” formatting used in eval prompts
    # Example:
    # A. ...
    # B. ...
    return "\n".join([f"{L}. {t}" for L, t in zip(letters, texts)])

def qwen_eval_block(question: str, options_block: str, answer_letter: str | None = None) -> str:
    """
    This matches the Qwen3-VL technical report style for multiple choice:
    Respond with only the letter ... Question ... Possible answer choices ... The best answer is:
    If answer_letter is provided, we append it (for few-shot demos).
    """
    base = (
        "Respond with only the letter of the correct option.\n"
        f"Question: {question} Possible answer choices:\n"
        f"{options_block}\n"
        "The best answer is:"
    )
    if answer_letter is not None:
        return base + f" {answer_letter}\n"
    return base  # for the final query, we leave it open

def build_fewshot_prompt(lang: str, demo_samples: list, test_sample: dict) -> str:
    # Build N-shot demonstrations
    blocks = []
    for s in demo_samples:
        letters, texts = extract_options(s)
        options_block = format_options_block(letters, texts)
        blocks.append(qwen_eval_block(s["question"], options_block, s["answer"]))

    # Final test question (no answer appended)
    letters, texts = extract_options(test_sample)
    options_block = format_options_block(letters, texts)
    blocks.append(qwen_eval_block(test_sample["question"], options_block, answer_letter=None))

    # Separate blocks with a blank line (common practice, helps parsing)
    return "\n\n".join(blocks), letters  # return letters for allowed choices

@torch.inference_mode()
def pick_choice(prompt: str, choices: list[str], *, model, processor, tokenizer, device) -> str:
    # Put the whole prompt inside a single user message, as you already do
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    enc = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]

    logits_processor = LogitsProcessorList([PresencePenaltyGeneratedOnly(PRESENCE_PENALTY, prompt_len)])

    generated_ids = model.generate(
        **enc,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_new_tokens=MAX_NEW_TOKENS_FOR_MCQ,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_only = generated_ids[:, enc["input_ids"].shape[1]:]
    out_text = processor.batch_decode(
        gen_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # Strong extraction: first character that is a valid choice
    for ch in out_text:
        if ch in choices:
            return ch

    return choices[0]

# ---------------------------
# Evaluation
# ---------------------------
def evaluate(
    *,
    model_id: str,
    langs: list[str],
    local_files_only: bool,
    max_test_examples: int | None,
    max_val_examples: int | None,
):
    results = {}
    total_correct_all = 0
    total_all = 0

    rng = random.Random(FEWSHOT_SEED)

    model, processor, tokenizer, device = load_model(model_id, local_files_only=local_files_only)

    for lang in langs:
        # test split (scored)
        test_ds = load_dataset(
            "li-lab/MMLU-ProX",
            lang,
            split="test",
            download_mode="reuse_dataset_if_exists",
        )
        if max_test_examples is not None:
            test_ds = test_ds.select(range(min(max_test_examples, len(test_ds))))

        # validation split (70 items) for few-shot prompt construction
        val_ds = load_dataset(
            "li-lab/MMLU-ProX",
            lang,
            split="validation",
            download_mode="reuse_dataset_if_exists",
        )
        if max_val_examples is not None:
            val_ds = val_ds.select(range(min(max_val_examples, len(val_ds))))

        # Pick a fixed 5-shot set per language (reproducible)
        # (Many papers do “sample k demos from validation”.)
        idxs = list(range(len(val_ds)))
        rng.shuffle(idxs)
        demo_samples = [val_ds[i] for i in idxs[:N_SHOT]]

        correct = 0
        total = len(test_ds)

        print(f"\nEvaluating language: {lang} ({total} examples) with {N_SHOT}-shot")

        for sample in tqdm(test_ds):
            prompt, choices = build_fewshot_prompt(lang, demo_samples, sample)
            pred = pick_choice(
                prompt,
                choices,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                device=device,
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

    print("\n=== MMLU-ProX Summary ===")
    print(f"Macro-average over {len(results)} languages: {macro_avg:.2f}%")
    print(f"Micro-average over all examples: {micro_avg:.2f}%")

    return results, macro_avg, micro_avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--langs", nargs="*", default=None, help="Override language list (e.g., --langs en fr)")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model/dataset artifacts")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny subset for a fast end-to-end check")
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    args = parser.parse_args()

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
    )
    print("\n=== Per-language Accuracies ===")
    for lang, acc in results.items():
        print(f"{lang}: {acc:.2f}%")
