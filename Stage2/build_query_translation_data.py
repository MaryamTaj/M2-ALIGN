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
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(args):
    target_langs = [x.strip() for x in args.target_languages.split(",") if x.strip()]
    for lang in target_langs:
        if lang not in NLLB_LANG_MAP:
            raise ValueError(f"Unsupported target language code: {lang}")

    rows = read_jsonl(args.input_path)
    print(f"Input rows: {len(rows)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = NllbTokenizer.from_pretrained(args.nllb_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.nllb_model).to(device)
    model.eval()

    out_rows = []
    for row in tqdm(rows, desc="translating"):
        en_query = row["query"]
        for lang in target_langs:
            tokenizer.src_lang = NLLB_LANG_MAP["en"]
            encoded = tokenizer(
                en_query,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_source_length,
            ).to(device)

            forced_bos = tokenizer.convert_tokens_to_ids(NLLB_LANG_MAP[lang])
            gen_ids = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos,
                max_new_tokens=args.max_target_length,
                num_beams=args.num_beams,
            )
            translated_query = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            out_rows.append(
                {
                    "id": f'{row["id"]}_{lang}',
                    "query": translated_query,
                    "answer": row["answer"],
                    "source_language": LANG_NAME[lang],
                    "target_language": "English",
                    "source_dataset": row.get("source_dataset", "task_specialization_en"),
                    "source_query_en": en_query,
                    "target_lang_code": lang,
                }
            )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Output rows: {len(out_rows)}")
    print(f"Saved translated data: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--target-languages", type=str, default="sw,yo,wo")
    parser.add_argument("--nllb-model", type=str, default="facebook/nllb-200-3.3B")
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    args = parser.parse_args()
    main(args)
