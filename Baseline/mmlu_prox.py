# import torch
# from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
# from datasets import load_dataset

# # Access the MMLU-ProX dataset:
# # LANGS = ["en","zh","ja","ko","fr","de","es","pt","ar","th","hi","bn","sw","af","cs",
# #          "hu","id","it","mr","ne","ru", "sr", "te", "uk", "ur", "vi", "wo", "yo", "zu"]
# # test = {}
# # for language in LANGS:
# #     test[language] = load_dataset("li-lab/MMLU-ProX", language, split="test")

# MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct" 
# assert torch.cuda.is_available(), "CUDA not available - did you request a GPU?" 
# device = torch.device("cuda") 
# cap_major = torch.cuda.get_device_capability(0)[0] 
# dtype = torch.bfloat16 if cap_major >= 8 else torch.float16 

# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     MODEL_ID,
#     dtype=dtype,
#     device_map="auto",
#     low_cpu_mem_usage=True,
#     local_files_only=True,
# )
# processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
# model.eval() 


# @torch.inference_mode() 
# def generate_text_response(user_text: str, max_new_tokens: int = 8) -> str: 
#     messages = [ { "role": "user", "content": [{"type": "text", "text": user_text}], } ] 
#     inputs = processor.apply_chat_template( messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", ) 
#     inputs = {k: v.to(device) for k, v in inputs.items()} 

#     generated_ids = model.generate( **inputs, do_sample=False, temperature=0.0, top_p=1.0, max_new_tokens=max_new_tokens, ) 
#     gen_only = generated_ids[:, inputs["input_ids"].shape[1]:] 
#     return processor.batch_decode( gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False, )[0].strip() 

# # ---- Smoke test ---- 
# print( generate_text_response( "Answer with just one letter: A, B, C, or D. What is 2+2? " "A)3 B)4 C)5 D)6" ) )

# import string
# import torch
# from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
# from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
# from datasets import load_dataset
# from tqdm import tqdm

# # ---------------------------
# # Config
# # ---------------------------
# MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# LANGS = [
#     "en", "zh", "ja", "ko", "fr", "de", "es", "pt", "ar", "th", "hi", "bn", "sw", "af", "cs",
#     "hu", "id", "it", "mr", "ne", "ru", "sr", "te", "uk", "ur", "vi", "wo", "yo", "zu"
# ]

# # Paper says for small instruct models: temperature=1.0, top_p=1.0, top_k=40, presence_penalty=2.0
# TEMPERATURE = 1.0
# TOP_P = 1.0
# TOP_K = 40
# PRESENCE_PENALTY = 2.0

# # IMPORTANT: The paper sets max output length to 32768 tokens, but for MMLU multiple-choice,
# # letting the model ramble destroys accuracy when you parse a single-letter answer.
# # We keep the paper's global cap as a constant, but we *actually generate* only a few tokens.
# MAX_OUTPUT_LEN_PAPER = 32768
# MAX_NEW_TOKENS_FOR_MCQ = 4  # critical for accuracy

# # ---------------------------
# # Load model (offline)
# # ---------------------------
# assert torch.cuda.is_available(), "CUDA not available - did you request a GPU?"
# device = torch.device("cuda")
# cap_major = torch.cuda.get_device_capability(0)[0]
# dtype = torch.bfloat16 if cap_major >= 8 else torch.float16

# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     MODEL_ID,
#     dtype=dtype,
#     device_map="auto",
#     low_cpu_mem_usage=True,
#     local_files_only=True,
# )
# processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
# tokenizer = processor.tokenizer
# model.eval()

# # ---------------------------
# # Presence penalty (HF way)
# # ---------------------------
# class PresencePenaltyGeneratedOnly(LogitsProcessor):
#     def __init__(self, penalty: float, prompt_len: int):
#         self.penalty = float(penalty)
#         self.prompt_len = int(prompt_len)

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         if self.penalty == 0.0:
#             return scores
#         # Only consider tokens generated after the prompt
#         if input_ids.size(1) <= self.prompt_len:
#             return scores
#         gen_part = input_ids[:, self.prompt_len:]
#         for b in range(input_ids.size(0)):
#             seen = torch.unique(gen_part[b])
#             scores[b, seen] -= self.penalty
#         return scores


# # ---------------------------
# # Helpers
# # ---------------------------
# def extract_options(sample):
#     """
#     Returns:
#       letters: ["A", "B", ...]
#       texts  : ["option text A", "option text B", ...]
#     """
#     option_items = []
#     for k, v in sample.items():
#         if k.startswith("option_") and v is not None:
#             idx = int(k.split("_")[1])
#             option_items.append((idx, v))
#     option_items.sort(key=lambda x: x[0])

#     texts = [v for _, v in option_items]
#     letters = list(string.ascii_uppercase[:len(texts)])
#     return letters, texts


# def format_prompt(question: str, letters: list[str], texts: list[str]) -> str:
#     allowed = ", ".join(letters)
#     lines = [
#         "Choose the correct answer.",
#         f"Reply with exactly ONE capital letter from: {allowed}.",
#         "No explanation.",
#         "",
#         f"Question: {question}",
#         "Options:",
#     ]
#     for L, txt in zip(letters, texts):
#         lines.append(f"{L}. {txt}")
#     lines.append("")
#     lines.append("Answer:")
#     return "\n".join(lines)



# @torch.inference_mode()
# def pick_choice_by_sampling(prompt: str, choices: list[str]) -> str:
#     """
#     Sampling-based decoding with paper hyperparams, but with a short max_new_tokens for MCQ,
#     and robust extraction of the first non-whitespace character.
#     """
#     messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
#     enc = processor.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_tensors="pt",
#         return_dict=True,
#     )
#     enc = {k: v.to(device) for k, v in enc.items()}
#     prompt_len = enc["input_ids"].shape[1]
#     logits_processor = LogitsProcessorList([PresencePenaltyGeneratedOnly(PRESENCE_PENALTY, prompt_len)])

#     generated_ids = model.generate(
#         **enc,
#         do_sample=True,
#         temperature=TEMPERATURE,
#         top_p=TOP_P,
#         top_k=TOP_K,
#         max_new_tokens=MAX_NEW_TOKENS_FOR_MCQ,  # critical for MCQ parsing
#         logits_processor=logits_processor,
#         pad_token_id=tokenizer.eos_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#     )

#     gen_only = generated_ids[:, enc["input_ids"].shape[1]:]
#     out_text = processor.batch_decode(
#         gen_only,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False,
#     )[0]

#     out_text = out_text.strip()
#     first = next((c for c in out_text if not c.isspace()), "")
#     if first in choices:
#         return first

#     # Fallback: scan for any valid choice letter in the short output
#     for ch in out_text:
#         if ch in choices:
#             return ch

#     return choices[0]


# # ---------------------------
# # Evaluation
# # ---------------------------
# def evaluate():
#     results = {}
#     total_correct_all = 0
#     total_all = 0

#     for lang in LANGS:
#         dataset = load_dataset(
#             "li-lab/MMLU-ProX",
#             lang,
#             split="test",
#             download_mode="reuse_dataset_if_exists",
#         )
#         correct = 0
#         total = len(dataset)

#         print(f"\nEvaluating language: {lang} ({total} examples)")
#         for sample in tqdm(dataset):
#             letters, texts = extract_options(sample)
#             answer = sample["answer"]

#             prompt = format_prompt(sample["question"], letters, texts)
#             pred = pick_choice_by_sampling(prompt, letters)

#             if pred == answer:
#                 correct += 1

#         acc = correct / total * 100
#         results[lang] = acc
#         total_correct_all += correct
#         total_all += total
#         print(f"Accuracy for {lang}: {acc:.2f}%")

#     macro_avg = sum(results.values()) / len(results)
#     micro_avg = (total_correct_all / total_all) * 100

#     print("\n=== MMLU-ProX Summary ===")
#     print(f"Macro-average over {len(results)} languages: {macro_avg:.2f}%")
#     print(f"Micro-average over all examples: {micro_avg:.2f}%")

#     return results, macro_avg, micro_avg


# if __name__ == "__main__":
#     results, macro_avg, micro_avg = evaluate()
#     print("\n=== Per-language Accuracies ===")
#     for lang, acc in results.items():
#         print(f"{lang}: {acc:.2f}%")


import string
import random
import torch

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
assert torch.cuda.is_available(), "CUDA not available - did you request a GPU?"
device = torch.device("cuda")
cap_major = torch.cuda.get_device_capability(0)[0]
dtype = torch.bfloat16 if cap_major >= 8 else torch.float16

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True,
    local_files_only=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
tokenizer = processor.tokenizer
model.eval()

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
def pick_choice(prompt: str, choices: list[str]) -> str:
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
def evaluate():
    results = {}
    total_correct_all = 0
    total_all = 0

    rng = random.Random(FEWSHOT_SEED)

    for lang in LANGS:
        # test split (scored)
        test_ds = load_dataset(
            "li-lab/MMLU-ProX",
            lang,
            split="test",
            download_mode="reuse_dataset_if_exists",
        )

        # validation split (70 items) for few-shot prompt construction
        val_ds = load_dataset(
            "li-lab/MMLU-ProX",
            lang,
            split="validation",
            download_mode="reuse_dataset_if_exists",
        )

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
            pred = pick_choice(prompt, choices)

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
    results, macro_avg, micro_avg = evaluate()
    print("\n=== Per-language Accuracies ===")
    for lang, acc in results.items():
        print(f"{lang}: {acc:.2f}%")
