"""Translate English task-specialization queries to target languages using NLLB.

Reads an English JSONL file produced by ``prepare_task_specialization_data.py``,
translates the ``"query"`` field of each row to each requested target language
with the NLLB-200 model, and writes the results to a new JSONL file.  Answer
strings are kept in English unchanged.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer


NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "sw": "swh_Latn",
    "yo": "yor_Latn",
    "wo": "wol_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
}

LANG_NAME = {
    "en": "English",
    "sw": "Swahili",
    "yo": "Yoruba",
    "wo": "Wolof",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ar": "Arabic",
    "hi": "Hindi",
    "bn": "Bengali",
}


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file and return its rows as a list of dicts.

    Tolerates a leading BOM and stray non-JSON prefixes before ``{``.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        A list of parsed JSON objects.

    Raises:
        ValueError: If any line cannot be parsed as JSON.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip().lstrip("﻿")
            if not line:
                continue
            if not line.startswith("{"):
                brace = line.find("{")
                if brace > 0:
                    print(
                        f"warning: {path}:{lineno} starts with non-JSON prefix "
                        f"{line[:brace]!r}; stripping before parse."
                    )
                    line = line[brace:]
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON at {path}:{lineno}: {exc.msg} "
                    f"(line preview: {line[:120]!r})"
                ) from exc
    return rows


def translate_text(
    text: str,
    target_lang_code: str,
    tokenizer: NllbTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    max_source_length: int,
    max_target_length: int,
    num_beams: int,
) -> str:
    """Translate *text* from English to *target_lang_code* using NLLB.

    Args:
        text: English source text to translate.
        target_lang_code: Two-letter language code (key in :data:`NLLB_LANG_MAP`).
        tokenizer: An instantiated :class:`NllbTokenizer`.
        model: An instantiated NLLB seq2seq model in eval mode.
        device: Device on which *model* lives.
        max_source_length: Maximum source token length (longer sequences
            are right-truncated).
        max_target_length: Maximum number of tokens to generate.
        num_beams: Number of beams for beam search.

    Returns:
        The translated string (decoded without special tokens).
    """
    tokenizer.src_lang = NLLB_LANG_MAP["en"]
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_source_length,
    ).to(device)
    forced_bos = tokenizer.convert_tokens_to_ids(NLLB_LANG_MAP[target_lang_code])
    gen_ids = model.generate(
        **encoded,
        forced_bos_token_id=forced_bos,
        max_new_tokens=max_target_length,
        num_beams=num_beams,
    )
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def main(args) -> None:
    """Translate queries and write the output JSONL file.

    Loads the NLLB model once, iterates over *args.max_input_rows* rows
    from *args.input_path*, translates each query to every language in
    *args.target_languages*, and writes one output row per
    (source-row, target-language) pair to *args.output_path*.

    Args:
        args: Parsed argument namespace.

    Raises:
        ValueError: If any code in *args.target_languages* is not in
            :data:`NLLB_LANG_MAP`.
    """
    target_langs = [x.strip() for x in args.target_languages.split(",") if x.strip()]
    for lang in target_langs:
        if lang not in NLLB_LANG_MAP:
            raise ValueError(f"Unsupported target language code: {lang!r}")

    rows = read_jsonl(args.input_path)
    print(f"Input rows: {len(rows)}")
    rows = rows[: args.max_input_rows]
    print(f"Rows selected for translation: {len(rows)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = NllbTokenizer.from_pretrained(args.nllb_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.nllb_model).to(device)
    model.eval()

    out_rows = []
    for row in tqdm(rows, desc="translating"):
        en_query = row["query"]
        for lang in target_langs:
            translated = translate_text(
                text=en_query,
                target_lang_code=lang,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length,
                num_beams=args.num_beams,
            )
            out_rows.append({
                "id": f'{row["id"]}_{lang}',
                "query": translated,
                "answer": row["answer"],
                "source_language": LANG_NAME[lang],
                "target_language": "English",
                "source_dataset": row.get("source_dataset", "task_specialization_en"),
                "source_query_en": en_query,
                "target_lang_code": lang,
            })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Output rows: {len(out_rows)}")
    print(f"Saved translated data: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate English task-specialization queries to target languages."
    )
    parser.add_argument("--input-path", type=str, required=True,
                        help="Path to task_specialization_en.jsonl.")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Destination JSONL file for translated rows.")
    parser.add_argument("--target-languages", type=str, default="sw,yo,wo",
                        help="Comma-separated target language codes (e.g. 'sw,yo,wo').")
    parser.add_argument("--max-input-rows", type=int, default=3000,
                        help="Maximum number of English rows to translate.")
    parser.add_argument("--nllb-model", type=str, default="facebook/nllb-200-3.3B")
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    main(parser.parse_args())
