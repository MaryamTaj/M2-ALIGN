"""Evaluate one approach on one benchmark, in every language the benchmark covers.

The three approaches differ only in what the backbone reads:
- direct: the frozen backbone reads the native question.
- translate-then-test: NLLB-3.3B first translates the question into
  English, and the frozen backbone reads the English. 
- mapping: the mapped native question goes in front of the chat, and the
  chat holds the native question too, the same as in training.

Every eval file has the same row format, for example:

    {"id": "201307251", "lang": "bn", "question": "এটা কি মেঘাচ্ছন্ন?",
     "image": "Stage3/data/gqa/images/n161313.jpg", "answer": "no"}

The job writes {results}/{lang}.jsonl, with one record per question, and
{results}/summary.json, with the accuracy for each language.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

from src.backbones import HF_CACHE, LLMInputs, load_backbone
from src.encoders import NLLB_TAGS
from src.loaders import (
    DATA_DIR, VQA_SYSTEM, load_image, math_messages, read_jsonl, vqa_messages, write_jsonl,
)
from src.model import MappedVLM

EVAL_DIR = os.path.join(DATA_DIR, "eval")
TRANSLATED_DIR = os.path.join(DATA_DIR, "eval_translated")
TRANSLATOR = "facebook/nllb-200-3.3B"
TRANSLATE_BATCH_SIZE = 16
TRANSLATE_BEAMS = 4

APPROACHES = ["direct", "translate-then-test", "mapping"]
XGQA_LANGS = ["bn", "de", "id", "ko", "pt", "ru", "zh"]
CVQA_LANGS = ["bn", "ga", "id", "jv", "ko", "mn", "pt", "ru", "si", "zh"]
WORLDCUISINES_LANGS = ["bn", "id", "jv", "ko", "ru", "si", "zh"]
MATH_LANGS = ["bn", "de", "ru", "zh", "en"]

MAX_QUESTION_LEN = 256 
VQA_MAX_NEW_TOKENS = 16
MATH_MAX_NEW_TOKENS = 512
OPTION_LETTERS = "ABCDEFGH"


# Prompts 
def chat(system: str, prompt: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def cvqa_open_ended_prompt(question: str) -> str:
    """CVQA's own open-ended protocol: the question alone, with no options and no length instruction."""
    return f"Question: {question}"


def cvqa_mcq_prompt(question: str, choices: list[str]) -> str:
    options = "\n".join(f"{OPTION_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"Question: {question}\n{options}\nAnswer with the option's letter from the given choices directly."


# Marking
def normalize_answer(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text.lower().strip())
    return re.sub(r"\s+", " ", text).strip()


def extract_math_answer(text: str) -> str | None:
    """The number after "The answer is", else after "####", else the last number in the text."""
    for pattern in (r"(?:the\s+)?answer\s+is[:\s]+(-?[\d,]+(?:\.\d+)?)", r"####\s*(-?[\d,]+(?:\.\d+)?)"):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).replace(",", "")
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None


def math_correct(pred: str, target: str) -> bool:
    answer = extract_math_answer(pred)
    if answer is None:
        return False
    try:
        return float(answer) == float(target.replace(",", ""))
    except ValueError:
        return answer.strip() == target.strip()


# Benchmarks
class Benchmark:
    """One benchmark: where its files are, which languages it covers, and how to score one row."""

    data: str  # the folder under data/eval/
    langs: list[str]

    def score(self, predictor: Predictor, row: dict) -> dict:
        """Return the fields to record for this row. Always includes "correct"."""
        raise NotImplementedError


class XGQA(Benchmark):
    data, langs = "xgqa", XGQA_LANGS

    def score(self, predictor, row):
        pred = predictor.answer(row, vqa_messages(row["question"]), VQA_MAX_NEW_TOKENS)
        return {"pred": pred, "correct": normalize_answer(pred) == normalize_answer(row["answer"])}


class CVQAOpen(Benchmark):
    """No options shown. The prediction is the choice the model finds most likely, per token."""

    data, langs = "cvqa", CVQA_LANGS

    def score(self, predictor, row):
        messages = chat(VQA_SYSTEM, cvqa_open_ended_prompt(row["question"]))
        scores = predictor.choice_scores(row, messages, row["choices"])
        best = max(range(len(scores)), key=scores.__getitem__)
        return {
            "pred": row["choices"][best], "choice_scores": [round(s, 4) for s in scores],
            "correct": best == row["answer_index"],
        }


class CVQAClosed(Benchmark):
    """Lettered options in the prompt. The prediction is the letter with the highest next-token logit."""

    data, langs = "cvqa", CVQA_LANGS

    def score(self, predictor, row):
        messages = chat(VQA_SYSTEM, cvqa_mcq_prompt(row["question"], row["choices"]))
        logits, letter_mass = predictor.letter_scores(row, messages, len(row["choices"]))
        best = max(range(len(logits)), key=logits.__getitem__)
        return {
            "pred": row["choices"][best], "pred_letter": OPTION_LETTERS[best],
            "letter_mass": round(letter_mass, 4), "correct": best == row["answer_index"],
        }


class WorldCuisines(Benchmark):
    """Correct if any accepted answer appears in the output, ignoring case, as in the WorldCuisines paper."""

    langs = WORLDCUISINES_LANGS

    def __init__(self, task: str) -> None:
        self.data = f"worldcuisines_{task}"

    def score(self, predictor, row):
        pred = predictor.answer(row, vqa_messages(row["question"]), VQA_MAX_NEW_TOKENS)
        return {"pred": pred, "correct": any(a.lower() in pred.lower() for a in row["answers"] if a)}


class Math(Benchmark):
    langs = MATH_LANGS

    def __init__(self, data: str) -> None:
        self.data = data

    def score(self, predictor, row):
        pred = predictor.answer(row, math_messages(row["question"]), MATH_MAX_NEW_TOKENS)
        return {"pred": pred, "extracted": extract_math_answer(pred), "correct": math_correct(pred, row["answer"])}


BENCHMARKS = {
    "xgqa": XGQA(),
    "cvqa": CVQAOpen(),
    "cvqa_mcq": CVQAClosed(),
    "worldcuisines_task1": WorldCuisines("task1"),
    "worldcuisines_task2": WorldCuisines("task2"),
    "mgsm": Math("mgsm"),
    "msvamp": Math("msvamp"),
}


# Predictors
class Predictor:
    """The frozen backbone on its own. Used by direct and translate-then-test.

    Each method handles one row at a time. The row's image, if it has
    one, goes inside the chat.
    """

    def __init__(self, backbone) -> None:
        self.backbone = backbone

    def inputs(self, row: dict, messages: list[dict]) -> LLMInputs:
        return self.backbone.merge_vision(self.render(row, messages))

    def render(self, row: dict, messages: list[dict]) -> dict:
        image = load_image(row["image"]) if "image" in row else None
        chat_batch = self.backbone.batch_chats([self.backbone.render_chat(messages, image)])
        return {key: t.to(self.backbone.device) for key, t in chat_batch.items()}

    @torch.inference_mode()
    def answer(self, row: dict, messages: list[dict], max_new_tokens: int) -> str:
        return self.backbone.generate(self.inputs(row, messages), max_new_tokens)[0]

    @torch.inference_mode()
    def choice_scores(self, row: dict, messages: list[dict], choices: list[str]) -> list[float]:
        """The mean log-probability of each choice's tokens, written right after the prompt."""
        prompt = self.inputs(row, messages)
        n_prompt = prompt.embeds.size(1)
        scores = []
        for choice in choices:
            ids = self.backbone.tokenizer(choice, add_special_tokens=False, return_tensors="pt")["input_ids"]
            if ids.size(1) == 0:
                scores.append(float("-inf"))
                continue
            ids = ids.to(self.backbone.device)
            logits = self.backbone.logits(LLMInputs.cat([prompt, self.backbone.embed_tokens(ids)]))
            # The logit at position i predicts token i + 1, so the choice's
            # tokens are predicted from position n_prompt - 1 up to the second-to-last.
            log_probs = logits[0, n_prompt - 1:-1].float().log_softmax(dim=-1)
            scores.append(log_probs.gather(1, ids[0, :, None]).mean().item())
        return scores

    @torch.inference_mode()
    def letter_scores(self, row: dict, messages: list[dict], n_choices: int) -> tuple[list[float], float]:
        """The next-token logit of each option letter, and the total probability on those letters.

        A low total means the model did not answer with a letter at all.
        """
        next_logits = self.backbone.logits(self.inputs(row, messages))[0, -1].float()
        ids = torch.tensor(option_letter_ids(self.backbone.tokenizer, n_choices), device=next_logits.device)
        return next_logits[ids].tolist(), next_logits.softmax(dim=-1)[ids].sum().item()


class MappedPredictor(Predictor):
    """The mapping approach: the mapped question goes in front of the chat."""

    def __init__(self, model: MappedVLM) -> None:
        super().__init__(model.backbone)
        self.model = model

    def inputs(self, row: dict, messages: list[dict]) -> LLMInputs:
        source = self.model.encoder.tokenize([row["question"]], [row["lang"]], MAX_QUESTION_LEN)
        return self.model.build_inputs({"source": source, "chat": self.render(row, messages)})


def option_letter_ids(tokenizer, n_choices: int) -> list[int]:
    """The token id of each option letter. Each letter must be one token, because the scorer reads one logit per letter."""
    ids = []
    for letter in OPTION_LETTERS[:n_choices]:
        tokens = tokenizer(letter, add_special_tokens=False)["input_ids"]
        if len(tokens) != 1:
            raise ValueError(f"Option letter {letter!r} is {len(tokens)} tokens, expected 1")
        ids.append(tokens[0])
    return ids


def load_predictor(args) -> Predictor:
    if args.approach == "mapping":
        model = MappedVLM(args.backbone, args.encoder).to("cuda").eval()
        model.mapping.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["mapping"])
        return MappedPredictor(model)
    return Predictor(load_backbone(args.backbone).to("cuda"))


# Translate-then-test
@functools.lru_cache(maxsize=1)
def load_translator() -> tuple[NllbTokenizer, AutoModelForSeq2SeqLM]:
    tokenizer = NllbTokenizer.from_pretrained(TRANSLATOR, cache_dir=HF_CACHE)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR, cache_dir=HF_CACHE, dtype=torch.float16)
    return tokenizer, model.to("cuda").eval()


@torch.inference_mode()
def translate(texts: list[str], src_tag: str) -> dict[str, str]:
    """Translate texts into English. Returns {source text: English}.

    Each distinct text is translated once (xGQA repeats many questions).
    Texts are sorted by length so each batch pads to a similar size.
    """
    tokenizer, model = load_translator()
    tokenizer.src_lang = src_tag
    english_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    unique = sorted(set(texts), key=len)
    out = {}
    for start in range(0, len(unique), TRANSLATE_BATCH_SIZE):
        chunk = unique[start:start + TRANSLATE_BATCH_SIZE]
        enc = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=256, padding=True).to("cuda")
        ids = model.generate(**enc, forced_bos_token_id=english_id, max_new_tokens=256, num_beams=TRANSLATE_BEAMS)
        out.update(zip(chunk, tokenizer.batch_decode(ids, skip_special_tokens=True)))
    return out


def translate_rows(rows: list[dict], lang: str, path: str) -> list[dict]:
    """Replace each question with its English translation, keep the original, and cache the result at path."""
    english = translate([r["question"] for r in rows], NLLB_TAGS[lang])
    translated = [{**r, "question": english[r["question"]], "question_native": r["question"]} for r in rows]
    # Write to a temporary file first, so a job that stops halfway never leaves a partial cache.
    write_jsonl(path + ".tmp", translated)
    os.replace(path + ".tmp", path)
    return translated


# Evaluation loop
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one approach on one benchmark.")
    parser.add_argument("--approach", required=True, choices=APPROACHES)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--encoder", default="", help="mapping only.")
    parser.add_argument("--checkpoint", default="", help="mapping only: the trained mapping (.pt).")
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS))
    parser.add_argument("--results", required=True, help="Folder for the per-language records and summary.json.")
    parser.add_argument("--max-examples", type=int, default=0, help="Smoke test: score only this many rows per language.")
    args = parser.parse_args()
    if args.approach == "mapping" and not (args.encoder and args.checkpoint):
        parser.error("--approach mapping needs --encoder and --checkpoint")
    evaluate(args)


def load_rows(approach: str, benchmark: Benchmark, lang: str) -> list[dict]:
    """The rows to score. For translate-then-test, the questions are in English."""
    rows = read_jsonl(os.path.join(EVAL_DIR, benchmark.data, f"{lang}.jsonl"))
    if approach != "translate-then-test" or lang == "en":
        return rows
    cache = os.path.join(TRANSLATED_DIR, benchmark.data, f"{lang}.jsonl")
    return read_jsonl(cache) if os.path.exists(cache) else translate_rows(rows, lang, cache)


def evaluate(args) -> None:
    benchmark = BENCHMARKS[args.benchmark]
    # Load every language's rows first. Any translating happens here, and
    # the translator is freed before the backbone loads.
    rows_by_lang = {
        lang: load_rows(args.approach, benchmark, lang)[:args.max_examples or None] for lang in benchmark.langs
    }
    load_translator.cache_clear()
    torch.cuda.empty_cache()

    predictor = load_predictor(args)
    summary = {}
    for lang, rows in rows_by_lang.items():
        records = [{**row, **benchmark.score(predictor, row)} for row in tqdm(rows, desc=f"{args.benchmark}/{lang}")]
        for r in records[:3]:
            print(f"  {lang} | question {r['question'][:80]!r} | pred {r['pred'][:80]!r} | correct {r['correct']}")
        # Write each language as soon as it finishes, so a timed-out job keeps what it has done.
        write_jsonl(os.path.join(args.results, f"{lang}.jsonl"), records)
        accuracy = 100 * sum(r["correct"] for r in records) / len(records)
        summary[lang] = {"accuracy": round(accuracy, 2), "n": len(records)}
        print(f"{args.benchmark} {lang}: {accuracy:.2f}% (n={len(records)})", flush=True)

    with open(os.path.join(args.results, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
