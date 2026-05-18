"""Build translated Stage 2 training data for M2-ALIGN.

Downloads MMLU-aux (auxiliary_train), MetaMathQA (train), and MultiNLI (train),
samples --n-samples examples from each, translates to each target language using
NLLB-200-3.3B, and writes one JSONL file per (dataset, language) pair.

Output files written to --output-dir:
  mmlu_aux_<lang>.jsonl   -- MCQ, answer = letter (A/B/C/D)
  metamath_<lang>.jsonl   -- math CoT, answer = translated CoT response
  multinli_<lang>.jsonl   -- NLI, answer = English label (Entailment/Neutral/Contradiction)

MMLU queries are translated by translating the question and each answer option
separately, then recombining with letter labels intact.  This avoids NLLB
corrupting the structural A/B/C/D markers.

For MetaMathQA, both the question and the full chain-of-thought response are
translated.  
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import string

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer


NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "sw": "swh_Latn",
    "yo": "yor_Latn",
    "wo": "wol_Latn",
}

LANG_NAME = {
    "en": "English",
    "sw": "Swahili",
    "yo": "Yoruba",
    "wo": "Wolof",
}

NLI_LABEL_MAP = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

LETTERS = list(string.ascii_uppercase)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows):,} rows → {path}")


# ---------------------------------------------------------------------------
# Batched translation
# ---------------------------------------------------------------------------

def translate_texts(
    texts: list[str],
    target_lang_code: str,
    tokenizer: NllbTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    max_source_length: int,
    max_target_length: int,
    num_beams: int,
    batch_size: int,
) -> list[str]:
    """Translate a list of English strings to *target_lang_code* in batches."""
    tokenizer.src_lang = NLLB_LANG_MAP["en"]
    forced_bos = tokenizer.convert_tokens_to_ids(NLLB_LANG_MAP[target_lang_code])
    results: list[str] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=max_source_length,
            padding=True,
        ).to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_target_length,
                num_beams=num_beams,
            )
        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        results.extend(t.strip() for t in decoded)
    return results


# ---------------------------------------------------------------------------
# MMLU aux
# ---------------------------------------------------------------------------

def load_mmlu_examples(n: int, seed: int) -> list[dict]:
    print("Loading MMLU auxiliary_train ...")
    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    indices = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    rows = []
    for i in indices:
        ex = ds[i]
        choices = [c.strip() for c in ex["choices"]]
        answer_idx = int(ex["answer"])
        rows.append({
            "question": ex["question"].strip(),
            "choices": choices,
            "answer_letter": LETTERS[answer_idx],
        })
    print(f"  Sampled {len(rows):,} MMLU examples.")
    return rows


def build_mmlu_translated(
    rows: list[dict],
    lang_code: str,
    tokenizer: NllbTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict]:
    n_choices = max(len(r["choices"]) for r in rows)

    # Translate questions in one batch.
    print(f"  Translating MMLU questions ({len(rows):,}) ...")
    translated_questions = translate_texts(
        [r["question"] for r in rows],
        lang_code, tokenizer, model, device,
        args.max_source_length, args.max_target_length_short, args.num_beams, args.batch_size,
    )

    # Translate each option position across all examples in one batch.
    translated_choices_by_position: list[list[str]] = []
    for pos in range(n_choices):
        texts = [r["choices"][pos] if pos < len(r["choices"]) else "" for r in rows]
        non_empty_mask = [t != "" for t in texts]
        non_empty_texts = [t for t, keep in zip(texts, non_empty_mask) if keep]
        if not non_empty_texts:
            translated_choices_by_position.append([""] * len(rows))
            continue
        print(f"  Translating MMLU option {LETTERS[pos]} ({len(non_empty_texts):,} texts) ...")
        translated_non_empty = translate_texts(
            non_empty_texts,
            lang_code, tokenizer, model, device,
            args.max_source_length, args.max_target_length_short, args.num_beams, args.batch_size,
        )
        full = []
        ne_iter = iter(translated_non_empty)
        for keep in non_empty_mask:
            full.append(next(ne_iter) if keep else "")
        translated_choices_by_position.append(full)

    out = []
    for idx, row in enumerate(rows):
        translated_q = translated_questions[idx]
        n = len(row["choices"])
        translated_opts = [translated_choices_by_position[pos][idx] for pos in range(n)]
        options_block = "\n".join(f"{LETTERS[pos]}. {opt}" for pos, opt in enumerate(translated_opts))
        query = f"{translated_q}\n{options_block}"
        out.append({
            "id": stable_id(f"mmlu_{row['question']}_{lang_code}"),
            "query": query,
            "answer": row["answer_letter"],
            "source_language": LANG_NAME[lang_code],
            "target_language": "English",
            "source_dataset": "mmlu_aux",
        })
    return out


# ---------------------------------------------------------------------------
# MetaMathQA
# ---------------------------------------------------------------------------

def load_metamath_examples(n: int, seed: int) -> list[dict]:
    print("Loading MetaMathQA train ...")
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    indices = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    rows = [{"question": ds[i]["query"].strip(), "response": ds[i]["response"].strip()} for i in indices]
    print(f"  Sampled {len(rows):,} MetaMathQA examples.")
    return rows


def build_metamath_translated(
    rows: list[dict],
    lang_code: str,
    tokenizer: NllbTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict]:
    print(f"  Translating MetaMathQA questions ({len(rows):,}) ...")
    translated_questions = translate_texts(
        [r["question"] for r in rows],
        lang_code, tokenizer, model, device,
        args.max_source_length, args.max_target_length_short, args.num_beams, args.batch_size,
    )
    print(f"  Translating MetaMathQA responses ({len(rows):,}) ...")
    translated_responses = translate_texts(
        [r["response"] for r in rows],
        lang_code, tokenizer, model, device,
        args.max_source_length_long, args.max_target_length_long, args.num_beams, args.batch_size,
    )
    return [
        {
            "id": stable_id(f"metamath_{row['question']}_{lang_code}"),
            "query": translated_questions[i],
            "answer": translated_responses[i],
            "source_language": LANG_NAME[lang_code],
            "target_language": "English",
            "source_dataset": "metamath_qa",
        }
        for i, row in enumerate(rows)
    ]


# ---------------------------------------------------------------------------
# MultiNLI
# ---------------------------------------------------------------------------

def load_multinli_examples(n: int, seed: int) -> list[dict]:
    print("Loading MultiNLI train ...")
    ds = load_dataset("nyu-mll/multi_nli", split="train")
    valid_indices = [i for i in range(len(ds)) if ds[i]["label"] in NLI_LABEL_MAP]
    indices = random.Random(seed).sample(valid_indices, min(n, len(valid_indices)))
    rows = [
        {
            "premise": ds[i]["premise"].strip(),
            "hypothesis": ds[i]["hypothesis"].strip(),
            "label": NLI_LABEL_MAP[ds[i]["label"]],
        }
        for i in indices
    ]
    print(f"  Sampled {len(rows):,} MultiNLI examples.")
    return rows


def build_multinli_translated(
    rows: list[dict],
    lang_code: str,
    tokenizer: NllbTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict]:
    # Concatenate premise + hypothesis as MERLIN does, translate the combined string.
    combined = [f"{r['premise']}\n{r['hypothesis']}" for r in rows]
    print(f"  Translating MultiNLI premise+hypothesis ({len(combined):,}) ...")
    translated = translate_texts(
        combined,
        lang_code, tokenizer, model, device,
        args.max_source_length, args.max_target_length_short, args.num_beams, args.batch_size,
    )
    return [
        {
            "id": stable_id(f"multinli_{rows[i]['premise']}_{lang_code}"),
            "query": translated[i],
            "answer": rows[i]["label"],  # Keep label in English (standard for NLI eval)
            "source_language": LANG_NAME[lang_code],
            "target_language": "English",
            "source_dataset": "multi_nli",
        }
        for i in range(len(rows))
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    target_langs = [x.strip() for x in args.target_languages.split(",") if x.strip()]
    for lang in target_langs:
        if lang not in NLLB_LANG_MAP:
            raise ValueError(f"Unsupported language code {lang!r}. Supported: {sorted(NLLB_LANG_MAP)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading NLLB model from: {args.nllb_model}")
    tokenizer = NllbTokenizer.from_pretrained(args.nllb_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.nllb_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    print("Model loaded.\n")

    mmlu_rows = load_mmlu_examples(args.n_samples, args.seed)
    metamath_rows = load_metamath_examples(args.n_samples, args.seed)
    multinli_rows = load_multinli_examples(args.n_samples, args.seed)

    for lang in target_langs:
        print(f"\n{'='*60}")
        print(f"Translating to {LANG_NAME[lang]} ({lang})")
        print(f"{'='*60}")

        mmlu_out = build_mmlu_translated(mmlu_rows, lang, tokenizer, model, device, args)
        write_jsonl(os.path.join(args.output_dir, f"mmlu_aux_{lang}.jsonl"), mmlu_out)

        metamath_out = build_metamath_translated(metamath_rows, lang, tokenizer, model, device, args)
        write_jsonl(os.path.join(args.output_dir, f"metamath_{lang}.jsonl"), metamath_out)

        multinli_out = build_multinli_translated(multinli_rows, lang, tokenizer, model, device, args)
        write_jsonl(os.path.join(args.output_dir, f"multinli_{lang}.jsonl"), multinli_out)

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build translated Stage 2 training data.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write output JSONL files.")
    parser.add_argument("--nllb-model", type=str, default="facebook/nllb-200-3.3B",
                        help="HF id or local path of the NLLB-200-3.3B model.")
    parser.add_argument("--target-languages", type=str, default="sw,yo,wo",
                        help="Comma-separated target language codes (sw, yo, wo).")
    parser.add_argument("--n-samples", type=int, default=3000,
                        help="Number of examples to sample from each dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Translation batch size.")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=256,
                        help="Max source tokens for questions/options/NLI.")
    parser.add_argument("--max-source-length-long", type=int, default=512,
                        help="Max source tokens for MetaMathQA CoT responses.")
    parser.add_argument("--max-target-length-short", type=int, default=256,
                        help="Max generated tokens for questions/options/NLI.")
    parser.add_argument("--max-target-length-long", type=int, default=512,
                        help="Max generated tokens for MetaMathQA CoT responses.")
    main(parser.parse_args())
