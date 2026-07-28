"""Stage 3b evaluation on real/native multilingual VQA benchmarks.

Loads the trained AugmentedVisualMindMerger (NLLB encoder -> mapping ->
frozen Qwen3-VL + LLM-side query prefix) and evaluates on whichever of
xGQA / WorldCuisines / CVQA applies per language (see the Stage 3 plan's
language survey). No NLLB-translated data is ever used here -- only the
JSONL files produced by load_evaluation_data.py (real/native benchmarks).

Run this against the Stage 2 checkpoint (zero-shot, no VQA training yet --
the primary hypothesis's "before" reference point) and again against the
final Stage 3b checkpoint (the "after" result).

Supported benchmarks
--------------------
Policy: wherever a language has coverage in more than one benchmark below,
run all of them -- they're independent evaluations, not interchangeable.

xgqa           -- open-ended, English short answers. Active languages:
                  bn/de/ru/zh/pt/id/ko (all 8 of xGQA's own languages that
                  this project has a checkpoint for).
worldcuisines  -- open-ended, English short answers (Indonesian, Javanese --
                  deferred)
cvqa           -- multiple-choice dataset, but scored **open-ended via
                  answer-choice log-likelihood** (the CVQA paper's own
                  "open-ended" evaluation protocol: prompt with the question
                  only, no visible options; pick whichever choice string the
                  model assigns the highest average per-token
                  log-probability to; compare that index against the gold
                  label). Active languages: mn/si/ga (no xGQA coverage for
                  any of these three), plus bn/ru/zh/pt/id/ko (also run
                  through xGQA above, per the policy note). Javanese also
                  covered but deferred. No German coverage -- CVQA has no
                  German subset at all.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, NllbTokenizer
from tqdm import tqdm

from model import AugmentedVisualMindMerger

# Data/outputs/logs live on $SCRATCH, not in the git checkout.
SCRATCH_ROOT = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN", "Stage3")

NLLB_CODES: dict[str, str] = {
    "bn": "ben_Beng",
    "id": "ind_Latn",
    "jv": "jav_Latn",
    "ru": "rus_Cyrl",
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "pt": "por_Latn",
    "ko": "kor_Hang",
    "mn": "khk_Cyrl",  # NLLB-200 only ships Halh Mongolian, Cyrillic script
    "si": "sin_Sinh",
    "ga": "gle_Latn",
}

_VQA_SYSTEM = "You are a helpful assistant that answers questions about images."


def build_open_ended_prompt(question: str) -> str:
    """Must match train.py's `_build_user_prompt` exactly."""
    return f"Question: {question}\nAnswer with a single word or short phrase, in English."


def build_cvqa_open_ended_prompt(question: str) -> str:
    """CVQA's own "open-ended" protocol: the question only, no options shown
    and no "single word" instruction (CVQA answers are full phrases, e.g.
    city names, not necessarily one word)."""
    return f"Question: {question}"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(benchmark: str) -> logging.Logger:
    """Create a logger that writes to stdout.

    The job script's SLURM ``--output`` file is the single log file for a
    run, so this only needs to format stdout consistently -- it must not
    also write its own file, or every run ends up with two logs.
    """
    logger = logging.getLogger(f"stage3b_eval_{benchmark}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    llm_path: str,
    mt_path: str,
    mapping_ckpt: str,
    local_files_only: bool,
    max_gen_len: int,
    logger: logging.Logger,
) -> tuple[AugmentedVisualMindMerger, AutoProcessor, NllbTokenizer, AutoTokenizer, torch.device, torch.dtype]:
    assert torch.cuda.is_available(), "CUDA not available - request a GPU node."
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path, use_fast=True, local_files_only=local_files_only)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"
    tokenizer_llm.truncation_side = "left"

    tokenizer_mt = NllbTokenizer.from_pretrained(mt_path, local_files_only=local_files_only)
    processor = AutoProcessor.from_pretrained(llm_path, local_files_only=local_files_only)

    model = AugmentedVisualMindMerger(
        mt_path=mt_path, llm_path=llm_path, max_gen_len=max_gen_len,
        llm_bos_token_id=tokenizer_llm.bos_token_id, llm_pad_token_id=tokenizer_llm.pad_token_id,
        local_files_only=local_files_only,
    )

    ckpt = torch.load(mapping_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.mapping.load_state_dict(state_dict, strict=False)
    logger.info("Loaded mapping from: %s", mapping_ckpt)
    if missing:
        logger.warning("  missing keys (%d): %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("  unexpected keys (%d): %s", len(unexpected), unexpected[:5])

    model.to(device)
    model.eval()
    return model, processor, tokenizer_mt, tokenizer_llm, device, amp_dtype


# ---------------------------------------------------------------------------
# Image resolution
# ---------------------------------------------------------------------------

def load_image_by_id(images_dir: str, image_id: str) -> Image.Image | None:
    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(images_dir, f"{image_id}{ext}")
        if os.path.exists(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                return None
    return None


def load_image_by_url(url: str, cache_dir: str) -> Image.Image | None:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, hashlib.sha1(url.encode()).hexdigest() + ".jpg")
    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            pass
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "M2-ALIGN/1.0"})
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(cache_path, format="JPEG", quality=85)
        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answer(
    image: Image.Image,
    question: str,
    prompt: str,
    system_message: str,
    nllb_code: str,
    model: AugmentedVisualMindMerger,
    processor: AutoProcessor,
    tokenizer_mt: NllbTokenizer,
    tokenizer_llm: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_new_tokens: int,
    visual_pixels: int,
    max_mt_seq_len: int,
    max_llm_seq_len: int,
) -> str:
    image_inputs = processor(
        images=[image], text=[""], return_tensors="pt", min_pixels=visual_pixels, max_pixels=visual_pixels,
    )
    pixel_values = image_inputs["pixel_values"].to(device)
    image_grid_thw = image_inputs["image_grid_thw"].to(device)

    tokenizer_mt.src_lang = nllb_code
    enc_mt = tokenizer_mt(question, truncation=True, max_length=max_mt_seq_len, return_tensors="pt")
    input_ids_mt = enc_mt["input_ids"].to(device)
    mask_mt = enc_mt["attention_mask"].to(device)

    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    formatted = tokenizer_llm.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc_llm = tokenizer_llm(formatted, truncation=True, max_length=max_llm_seq_len, return_tensors="pt")
    input_ids_query_llm = enc_llm["input_ids"].to(device)
    mask_query_llm = enc_llm["attention_mask"].to(device)

    gen_kw = dict(do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=tokenizer_llm.eos_token_id)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        out_texts = model.generate(
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            input_ids_mt=input_ids_mt, attention_mask_mt=mask_mt,
            input_ids_query_llm=input_ids_query_llm, mask_query_llm=mask_query_llm,
            tokenizer_llm=tokenizer_llm, generation_kwargs=gen_kw,
        )
    return (out_texts[0] if out_texts else "").strip()


@torch.inference_mode()
def score_choice_loglikelihood(
    image: Image.Image,
    question: str,
    choice_text: str,
    nllb_code: str,
    model: AugmentedVisualMindMerger,
    processor: AutoProcessor,
    tokenizer_mt: NllbTokenizer,
    tokenizer_llm: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    visual_pixels: int,
    max_mt_seq_len: int,
    max_llm_seq_len: int,
) -> float:
    """Length-normalized log-likelihood of *choice_text* as a continuation of
    the CVQA open-ended prompt (question only, no options shown).

    Reuses `AugmentedVisualMindMerger._build_prefix_raw` -- the same prefix
    construction `forward`/`generate` use -- so the choice-scoring inputs are
    identical to what training/generation would have seen up to this point;
    only the continuation differs (a candidate answer instead of a
    generated/label sequence). One forward pass per candidate choice (CVQA
    has ~4), not a single batched pass across choices, since choices differ
    in token length and batching them would need padding-aware log-prob
    gathering for a marginal speed gain over ~4 sequential passes/question.
    """
    image_inputs = processor(
        images=[image], text=[""], return_tensors="pt", min_pixels=visual_pixels, max_pixels=visual_pixels,
    )
    pixel_values = image_inputs["pixel_values"].to(device)
    image_grid_thw = image_inputs["image_grid_thw"].to(device)

    tokenizer_mt.src_lang = nllb_code
    enc_mt = tokenizer_mt(question, truncation=True, max_length=max_mt_seq_len, return_tensors="pt")
    input_ids_mt = enc_mt["input_ids"].to(device)
    mask_mt = enc_mt["attention_mask"].to(device)

    prompt = build_cvqa_open_ended_prompt(question)
    messages = [{"role": "system", "content": _VQA_SYSTEM}, {"role": "user", "content": prompt}]
    formatted = tokenizer_llm.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc_llm = tokenizer_llm(formatted, truncation=True, max_length=max_llm_seq_len, return_tensors="pt")
    input_ids_query_llm = enc_llm["input_ids"].to(device)
    mask_query_llm = enc_llm["attention_mask"].to(device)

    choice_ids = tokenizer_llm(choice_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    n_choice_tok = choice_ids.shape[1]
    if n_choice_tok == 0:
        return float("-inf")

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        prefix_embeds, prefix_mask = model._build_prefix_raw(
            pixel_values, image_grid_thw, input_ids_mt, mask_mt,
            input_ids_query_llm, mask_query_llm,
        )
        choice_embeds = model.llm_embedding_layer(choice_ids).to(model.llm_dtype)
        full_embeds = torch.cat([prefix_embeds, choice_embeds], dim=1)
        full_mask = torch.cat([prefix_mask, torch.ones_like(choice_ids)], dim=1)
        logits = model.model_llm(inputs_embeds=full_embeds, attention_mask=full_mask).logits

    # Position i's logits predict token i+1, so the logit that predicts
    # choice token k sits at (prefix_len - 1 + k).
    prefix_len = prefix_embeds.shape[1]
    choice_logits = logits[0, prefix_len - 1 : prefix_len - 1 + n_choice_tok, :].float()
    log_probs = torch.log_softmax(choice_logits, dim=-1)
    token_logprobs = log_probs.gather(1, choice_ids[0].unsqueeze(-1)).squeeze(-1)
    return (token_logprobs.sum() / n_choice_tok).item()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def open_ended_correct(pred_text: str, target: str) -> bool:
    return normalize_answer(pred_text) == normalize_answer(target)


# ---------------------------------------------------------------------------
# Evaluation loops
# ---------------------------------------------------------------------------

def evaluate_open_ended(
    rows: list[dict],
    lang: str,
    image_resolver,
    model, processor, tokenizer_mt, tokenizer_llm, device, amp_dtype,
    max_examples: int | None, visual_pixels: int, max_mt_seq_len: int, max_llm_seq_len: int,
    logger: logging.Logger,
) -> float:
    nllb_code = NLLB_CODES[lang]
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    for idx, row in enumerate(tqdm(rows, desc=f"open-ended/{lang}")):
        image = image_resolver(row)
        if image is None:
            continue
        prompt = build_open_ended_prompt(row["query"])
        pred = generate_answer(
            image, row["query"], prompt, _VQA_SYSTEM, nllb_code,
            model, processor, tokenizer_mt, tokenizer_llm, device, amp_dtype,
            max_new_tokens=16, visual_pixels=visual_pixels,
            max_mt_seq_len=max_mt_seq_len, max_llm_seq_len=max_llm_seq_len,
        )
        ok = open_ended_correct(pred, row["answer"])
        correct += int(ok)
        if idx < 5:
            logger.info("idx=%d question=%r pred=%r target=%r ok=%s", idx, row["query"][:80], pred, row["answer"], ok)
    acc = correct / len(rows) * 100 if rows else 0.0
    logger.info("lang=%s accuracy=%.2f%% (n=%d)", lang, acc, len(rows))
    return acc


def evaluate_cvqa_open_ended(
    rows: list[dict],
    lang: str,
    image_resolver,
    model, processor, tokenizer_mt, tokenizer_llm, device, amp_dtype,
    max_examples: int | None, visual_pixels: int, max_mt_seq_len: int, max_llm_seq_len: int,
    logger: logging.Logger,
) -> float:
    """CVQA's own open-ended protocol: no options shown in the prompt; score
    each candidate answer choice by log-likelihood and take the argmax."""
    nllb_code = NLLB_CODES[lang]
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    for idx, row in enumerate(tqdm(rows, desc=f"cvqa-open-ended/{lang}")):
        image = image_resolver(row)
        if image is None:
            continue
        choices = row["choices"]
        scores = [
            score_choice_loglikelihood(
                image, row["query"], choice, nllb_code,
                model, processor, tokenizer_mt, tokenizer_llm, device, amp_dtype,
                visual_pixels=visual_pixels, max_mt_seq_len=max_mt_seq_len, max_llm_seq_len=max_llm_seq_len,
            )
            for choice in choices
        ]
        pred_idx = max(range(len(scores)), key=lambda i: scores[i])
        target_idx = int(row["answer_index"])
        ok = pred_idx == target_idx
        correct += int(ok)
        if idx < 5:
            logger.info(
                "idx=%d question=%r choices=%r scores=%r pred=%d target=%d ok=%s",
                idx, row["query"][:80], choices, [round(s, 3) for s in scores], pred_idx, target_idx, ok,
            )
    acc = correct / len(rows) * 100 if rows else 0.0
    logger.info("lang=%s accuracy=%.2f%% (n=%d)", lang, acc, len(rows))
    return acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3b AugmentedVisualMindMerger VQA evaluation.")
    parser.add_argument("--benchmark", required=True, choices=["xgqa", "worldcuisines", "cvqa"])
    parser.add_argument("--lang", required=True,
                         help="ISO code (bn/de/ru/zh/pt/id/ko for xgqa; pt/ko/mn/si/ga/bn/ru/zh for cvqa; id/jv for worldcuisines).")
    parser.add_argument("--eval-data", required=True, help="JSONL from load_evaluation_data.py.")
    parser.add_argument("--images-dir", default=None, help="Local GQA images dir (required for --benchmark xgqa).")
    parser.add_argument("--image-cache-dir", default=os.path.join(SCRATCH_ROOT, "data", "stage3b_eval", "image_cache"),
                        help="URL image cache dir (used for worldcuisines/cvqa).")
    parser.add_argument("--llm-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--mt-path", default="facebook/nllb-200-3.3B")
    parser.add_argument("--mapping-ckpt", required=True,
                         help="Mapping checkpoint to evaluate -- Stage 2 (zero-shot 'before') "
                              "or Stage 3b (final 'after'); see the Stage 3 verification plan.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--visual-pixels", type=int, default=256 * 256)
    parser.add_argument("--max-mt-seq-len", type=int, default=256)
    parser.add_argument("--max-llm-seq-len", type=int, default=512)
    parser.add_argument("--max-gen-len", type=int, default=16)
    parser.add_argument("--smoke", action="store_true", help="Run 5 examples only.")
    args = parser.parse_args()

    if args.benchmark == "xgqa" and not args.images_dir:
        parser.error("--images-dir is required for --benchmark xgqa")

    logger = setup_logging(args.benchmark)
    max_examples = 5 if args.smoke else args.max_examples

    rows = _read_jsonl(args.eval_data)
    logger.info("Loaded %d rows from %s", len(rows), args.eval_data)

    model, processor, tokenizer_mt, tokenizer_llm, device, amp_dtype = load_model(
        args.llm_path, args.mt_path, args.mapping_ckpt, args.local_files_only, args.max_gen_len, logger,
    )

    if args.benchmark == "xgqa":
        resolver = lambda row: load_image_by_id(args.images_dir, row["vg_image_id"])
    else:
        resolver = lambda row: load_image_by_url(row["image_url"], args.image_cache_dir)

    common_kw = dict(
        image_resolver=resolver, model=model, processor=processor,
        tokenizer_mt=tokenizer_mt, tokenizer_llm=tokenizer_llm, device=device, amp_dtype=amp_dtype,
        max_examples=max_examples, visual_pixels=args.visual_pixels,
        max_mt_seq_len=args.max_mt_seq_len, max_llm_seq_len=args.max_llm_seq_len,
        logger=logger,
    )

    if args.benchmark == "cvqa":
        evaluate_cvqa_open_ended(rows, args.lang, **common_kw)
    else:
        evaluate_open_ended(rows, args.lang, **common_kw)


if __name__ == "__main__":
    main()
