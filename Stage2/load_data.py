"""Load Stage 2 training data from HuggingFace Hub, mirroring MindMerger.

Tasks and sources
-----------------
mgsm / msvamp  → 30,000 samples from Mathoctopus/Mathoctopus_cross_train
                  (Chen et al., 2023: Swahili query, English CoT response)
xnli           → 2,490 samples from xnli sw validation split
                  (Swahili premise+hypothesis, English label)
xcsqa          → 8,888 samples from INK-USC/xcsr X-CSQA-en train split,
                  translated to Swahili with NLLB-200-3.3B

All tasks output JSONL with fields:
    query            – Swahili input text
    answer           – expected response (English CoT / NLI label / answer letter)
    source_language  – "Swahili" (for NLLB tokenization in Stage2/train.py)
    source_dataset   – dataset identifier string
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

TASK_DEFAULTS: dict[str, int] = {
    "mgsm":   30_000,
    "msvamp": 30_000,
    "xnli":    2_940,
    "xcsqa":   8_888,
}

_XNLI_INT_TO_STR = {0: "entailment", 1: "neutral", 2: "contradiction"}
_NLLB_EN  = "eng_Latn"
_NLLB_SW  = "swh_Latn"


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"load_data_{timestamp}.log")
    logger = logging.getLogger("stage2_load_data")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def write_jsonl(path: str, rows: list[dict], logger: logging.Logger) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows → %s", len(rows), path)


# ---------------------------------------------------------------------------
# NLLB translation (used only for xcsqa)
# ---------------------------------------------------------------------------

def load_nllb(nllb_model: str, logger: logging.Logger):
    """Load NLLB-200-3.3B tokenizer and model."""
    logger.info("Loading NLLB tokenizer and model from: %s", nllb_model)
    tokenizer = NllbTokenizer.from_pretrained(nllb_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model, torch_dtype=dtype).to(device)
    model.eval()
    logger.info("NLLB loaded on %s.", device)
    return tokenizer, model, device


def translate_batch(
    texts: list[str],
    tokenizer: NllbTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 256,
    num_beams: int = 4,
) -> list[str]:
    """Translate a list of English strings to Swahili with NLLB."""
    tokenizer.src_lang = _NLLB_EN
    forced_bos = tokenizer.convert_tokens_to_ids(_NLLB_SW)
    results: list[str] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_length,
                num_beams=num_beams,
            )
        results.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
    return results


# ---------------------------------------------------------------------------
# Math — Chen et al. (2023) via Mathoctopus/Mathoctopus_cross_train
# ---------------------------------------------------------------------------

def load_math(n_samples: int, seed: int, logger: logging.Logger, **_) -> list[dict]:
    """Stream Mathoctopus_cross_train and collect n_samples Swahili rows.

    cross_train = Swahili query, English CoT response — the correct pairing
    for MindMerger: NLLB encodes the Swahili question, the frozen LLM
    generates the English answer.
    """
    logger.info("Streaming Mathoctopus/Mathoctopus_cross_train for Swahili ...")
    ds = load_dataset("Mathoctopus/Mathoctopus_cross_train", split="train", streaming=True)

    sw_rows: list[dict] = []
    for sample in ds:
        if sample.get("lang") != "Swahili":
            continue
        query    = sample.get("query",    "").strip()
        response = sample.get("response", "").strip()
        if not query or not response:
            continue
        sw_rows.append({
            "id": stable_id(f"math_{query}"),
            "query": query,
            "answer": response,
            "source_language": "Swahili",
            "source_dataset": "mathoctopus_cross_train",
        })
        if len(sw_rows) >= n_samples * 2:
            break

    logger.info("Collected %d Swahili math rows before sampling.", len(sw_rows))
    n = min(n_samples, len(sw_rows))
    if n < n_samples:
        logger.warning("Requested %d but only %d available; using all.", n_samples, n)
    return random.Random(seed).sample(sw_rows, n)


# ---------------------------------------------------------------------------
# XNLI — xnli Swahili validation split
# ---------------------------------------------------------------------------

def load_xnli(n_samples: int, seed: int, logger: logging.Logger, **_) -> list[dict]:
    """Load XNLI Swahili validation rows."""
    logger.info("Loading xnli sw validation ...")
    ds = load_dataset("xnli", "sw", split="validation")
    logger.info("XNLI sw validation: %d rows", len(ds))

    rows: list[dict] = []
    for r in ds:
        label_str = _XNLI_INT_TO_STR.get(int(r["label"]), "entailment")
        rows.append({
            "id": stable_id(f"xnli_{r['premise']}_{r['hypothesis']}"),
            "query": f"{r['premise']}\n{r['hypothesis']}",
            "answer": label_str,
            "source_language": "Swahili",
            "source_dataset": "xnli_sw_validation",
        })

    n = min(n_samples, len(rows))
    if n < n_samples:
        logger.warning("Requested %d but only %d available; using all.", n_samples, n)
    return random.Random(seed).sample(rows, n)


# ---------------------------------------------------------------------------
# X-CSQA — English train translated to Swahili with NLLB-200-3.3B
# ---------------------------------------------------------------------------

def load_xcsqa(
    n_samples: int,
    seed: int,
    logger: logging.Logger,
    nllb_model: str = "facebook/nllb-200-3.3B",
    batch_size: int = 16,
    num_beams: int = 4,
    **_,
) -> list[dict]:
    """Load X-CSQA English train and translate questions+choices to Swahili.

    INK-USC/xcsr provides no Swahili training split. We load the 8,888-example
    English train split, translate each question stem and each answer choice
    individually with NLLB-200-3.3B, then reassemble the formatted query.
    Answer keys (A/B/C/D/E) are structural labels and are not translated.
    """
    logger.info("Loading INK-USC/xcsr X-CSQA-en train ...")
    ds = load_dataset("INK-USC/xcsr", "X-CSQA-en", split="train")
    logger.info("X-CSQA-en train: %d rows", len(ds))

    n = min(n_samples, len(ds))
    if n < n_samples:
        logger.warning("Requested %d but only %d available; using all.", n_samples, n)
    indices = random.Random(seed).sample(range(len(ds)), n)
    sampled = [ds[i] for i in indices]

    # Collect all texts that need translation in one pass to maximise batching.
    stems   = [r["question"]["stem"] for r in sampled]
    # choices is {"label": [...], "text": [...]} in HuggingFace format.
    n_opts  = max(len(r["question"]["choices"]["label"]) for r in sampled)
    choices_per_pos: list[list[str]] = [
        [r["question"]["choices"]["text"][pos] if pos < len(r["question"]["choices"]["text"]) else ""
         for r in sampled]
        for pos in range(n_opts)
    ]

    tokenizer, model, device = load_nllb(nllb_model, logger)

    logger.info("Translating %d stems to Swahili ...", len(stems))
    sw_stems = translate_batch(stems, tokenizer, model, device, batch_size, num_beams=num_beams)

    sw_choices_per_pos: list[list[str]] = []
    for pos, texts in enumerate(choices_per_pos):
        non_empty = [(i, t) for i, t in enumerate(texts) if t]
        logger.info("Translating option %d (%d texts) ...", pos, len(non_empty))
        translated = translate_batch(
            [t for _, t in non_empty], tokenizer, model, device, batch_size, num_beams=num_beams
        )
        full: list[str] = [""] * len(texts)
        for (i, _), sw in zip(non_empty, translated):
            full[i] = sw
        sw_choices_per_pos.append(full)

    rows: list[dict] = []
    for idx, r in enumerate(sampled):
        labels = r["question"]["choices"]["label"]
        sw_opts = [sw_choices_per_pos[pos][idx] for pos in range(len(labels))]
        options_block = "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(labels, sw_opts))
        query = f"{sw_stems[idx]}\n{options_block}"
        rows.append({
            "id": r["id"],
            "query": query,
            "answer": r["answerKey"],
            "source_language": "Swahili",
            "source_dataset": "xcsqa_en_train_translated",
        })

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

LOADERS = {
    "mgsm":   load_math,
    "msvamp": load_math,
    "xnli":   load_xnli,
    "xcsqa":  load_xcsqa,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Stage 2 training data from HuggingFace (MindMerger protocol)."
    )
    parser.add_argument("--task", required=True, choices=list(LOADERS),
                        help="Task: mgsm, msvamp, xnli, or xcsqa.")
    parser.add_argument("--output_dir", type=str, default="./data/stage2",
                        help="Directory to write the output JSONL file.")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples (defaults to MindMerger paper values per task).")
    parser.add_argument("--seed", type=int, default=42)
    # xcsqa translation args (ignored for other tasks)
    parser.add_argument("--nllb_model", type=str, default="facebook/nllb-200-3.3B",
                        help="HuggingFace ID or local path for NLLB-200-3.3B (xcsqa only).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Translation batch size (xcsqa only).")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Beam width for NLLB translation (xcsqa only).")
    args = parser.parse_args()

    logger = setup_logging(os.path.join(os.path.dirname(__file__), "logs"))

    n_samples = args.n_samples if args.n_samples is not None else TASK_DEFAULTS[args.task]
    logger.info("Task=%s  n_samples=%d", args.task, n_samples)

    rows = LOADERS[args.task](
        n_samples=n_samples,
        seed=args.seed,
        logger=logger,
        nllb_model=args.nllb_model,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.task}.jsonl")
    write_jsonl(out_path, rows, logger)
    logger.info("Done.")


if __name__ == "__main__":
    main()
