import argparse
import json
import os
from datasets import load_dataset


LANG_TO_NLLB = {
    "English": "eng_Latn",
    'Swahili': 'swh_Latn', 
    'Yoruba': 'yor_Latn',
    'Wolof': 'wol_Latn'
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/nllb")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--samples_per_language", type=int, default=3000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    langs = ["Swahili","Yoruba","Wolof"]

    for source_lang in langs:
        source_code = LANG_TO_NLLB[source_lang]
        target_code = LANG_TO_NLLB["English"]
        config_name = f"{source_code}-{target_code}"
        ds = load_dataset("allenai/nllb", config_name, split=args.split)
        out_path = os.path.join(args.output_dir, f"{source_lang}_to_English.jsonl")
        kept = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for sample in ds:
                translation = sample.get("translation", {})
                source_text = translation.get(source_code, "").strip()
                target_text = translation.get(target_code, "").strip()
                if not source_text or not target_text:
                    continue
                f.write(json.dumps({"source": source_text, "target": target_text}, ensure_ascii=False) + "\n")
                kept += 1
                if kept >= args.samples_per_language:
                    break
        print(f"{source_lang}: wrote {kept} pairs to {out_path}")


if __name__ == "__main__":
    main()
