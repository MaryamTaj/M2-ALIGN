import argparse
import json
import os
from datasets import load_dataset


LANG_TO_NLLB = {
    "English": "eng_Latn",
    "Swahili": "swh_Latn",
    "Yoruba": "yor_Latn",
    "Wolof": "wol_Latn",
    "French": "fra_Latn",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/nllb")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--samples_per_language", type=int, default=3000)
    parser.add_argument(
        "--languages",
        type=str,
        default="French",
        help="Comma-separated source language names matching keys in LANG_TO_NLLB (e.g. French or Swahili,Yoruba,Wolof).",
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default on: stream rows from the Hub and stop after samples_per_language (avoids multi‑GiB full downloads). "
        "Use --no-streaming only if you want a full local cache (needs HF_DATASETS_CACHE on a large filesystem).",
    )
    args = parser.parse_args()

    cache_hint = os.environ.get("HF_DATASETS_CACHE") or os.environ.get("HF_HOME") or "~/.cache/huggingface"
    print(
        f"load_nllb_corpus: streaming={args.streaming}; "
        f"HF cache env effective root ~ {cache_hint} "
        "(set HF_HOME/HF_DATASETS_CACHE to $SCRATCH/... on clusters if not streaming)."
    )

    os.makedirs(args.output_dir, exist_ok=True)
    langs = [x.strip() for x in args.languages.split(",") if x.strip()]
    for name in langs:
        if name not in LANG_TO_NLLB:
            raise ValueError(f"Unknown language {name!r}; allowed: {sorted(LANG_TO_NLLB)}")

    for source_lang in langs:
        source_code = LANG_TO_NLLB[source_lang]
        target_code = LANG_TO_NLLB["English"]
        config_name = f"{target_code}-{source_code}"
        ds = load_dataset(
            "allenai/nllb",
            config_name,
            split=args.split,
            streaming=args.streaming,
        )
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
