"""Stage 3b evaluation on real/native multilingual VQA and text-reasoning benchmarks.

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
worldcuisines_task1/_task2 -- open-ended. Two independent benchmark rows,
                  both active for bn/ru/zh/id/ko/jv/si (formal register
                  everywhere a casual/formal or ngoko/krama split exists;
                  see load_evaluation_data.py's WORLDCUISINES_LANG_TAGS).
                  No pt/de/mn/ga coverage.
                    worldcuisines_task1 -- task1, Dish Name Prediction.
                    worldcuisines_task2 -- task2, Dish Origin Prediction
                                            (same schema, smaller split).
                  Both scored by
                  `evaluate_worldcuisines_open_ended`/`worldcuisines_correct`,
                  not the xGQA scorer: dual-reference (native-language
                  answer + English answer for the same qa_id, see
                  `load_worldcuisines`) and case-insensitive substring
                  containment, matching the WorldCuisines authors' own
                  `evaluation/score/score.py` (`oe_mode="dual"`) rather than
                  xGQA's exact-match. Prompted with `build_worldcuisines_prompt`,
                  which -- unlike the deferred-English rationale considered
                  and rejected here -- keeps the "single word/short phrase,
                  in English" instruction: the row's own-language `answer`
                  field is frequently just the English name left
                  untranslated (100% of rows for zh_cn/si_formal_spoken,
                  95%/93% for id_formal/bn -- see `load_worldcuisines`'s
                  docstring), so `answers` is always at least the English
                  reference and the dual-reference scorer already credits a
                  native answer on the minority of rows where one exists --
                  prompting for English just maximizes the odds of matching
                  whichever reference is actually there. This departs from
                  the WorldCuisines authors' own harness, which sends no
                  system prompt and no added instruction at all (just
                  `open_ended_prompt` verbatim) -- this project keeps its
                  system message (`_VQA_SYSTEM`) and English instruction
                  regardless, for consistency with xGQA/CVQA's prompting.
cvqa           -- multiple-choice dataset, but scored **open-ended via
                  answer-choice log-likelihood** (the CVQA paper's own
                  "open-ended" evaluation protocol: prompt with the question
                  only, no visible options; pick whichever choice string the
                  model assigns the highest average per-token
                  log-probability to; compare that index against the gold
                  label). Active languages: am/ig/om/mn/si/ga (no xGQA
                  coverage for any of these six -- am/ig/om are big-headroom,
                  low-resource languages added specifically to test where a
                  mapping layer has the most room to help; see
                  [[m2rb_vs_colleague_a2_comparison]] memory), plus
                  bn/ru/zh/pt/id/ko (also run through xGQA above, per the
                  policy note). Javanese also covered but deferred. No
                  German coverage -- CVQA has no German subset at all.
cvqa_generation -- diagnostic variant of cvqa: same eval data and same
                  no-options-shown prompt (`build_cvqa_open_ended_prompt`),
                  but scored via true `generate_answer` output
                  (`evaluate_cvqa_generation`/`cvqa_generation_correct`,
                  substring match against the gold choice text) instead of
                  `score_choice_loglikelihood`'s choice-ranking. Tests
                  whether part of the cvqa gap vs. Baseline is a
                  log-likelihood/generation eval-format artifact rather
                  than a capability gap -- see
                  [[cvqa_worldcuisines_open_questions]] memory. Same
                  language coverage as cvqa.
cvqa_mcq       -- closed-ended variant of cvqa: same eval data, but the
                  four English choices are shown in the prompt as lettered
                  options (`build_cvqa_mcq_prompt`) and scored by comparing
                  the next-token logits of the option letters at the first
                  answer position (`score_mcq_letters`) -- the options-shown
                  MCQ protocol every published CVQA number uses. One forward
                  pass per question. NLLB/X_m still see the native question
                  only; the options go into the LLM-side `T` only. Also logs
                  `letter_mass` (softmax probability on the option letters)
                  to separate "the prefix broke the MCQ answer format" from
                  "the prefix changed which option wins". Same language
                  coverage as cvqa.
mgsm / msvamp  -- text-only math, run through the SAME model with no image:
                  ``pixel_values``/``image_grid_thw``/``mm_token_type_ids``
                  are ``None``, so the prefix is ``[BOS] + X_m +
                  [end_boundary] + T`` with ``T`` a plain chat template.
                  Own prompt (`build_math_prompt`, `_MATH_SYSTEM`), greedy
                  chain-of-thought (512 new tokens) and numeric scoring
                  (`extract_math_answer`/`math_correct`), identical to
                  Baseline/evaluate.py's mgsm/msvamp path, so the per-item
                  comparison against the baseline isolates what the mapped
                  prefix does to reasoning. --eval-data is
                  Stage3/data/{mgsm,msvamp}/<lang>.jsonl
                  (load_text_evaluation.py); MGSM/MSVAMP are parallel across
                  languages, so ``en`` is the human-written original. The
                  mapping checkpoint is loaded strictly (a silently-random
                  mapping would make the probe meaningless).
                  (Formerly Stage3/evaluate_text.py.)
"""
from __future__ import annotations

import argparse
import contextlib
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

from model import AugmentedVisualMindMerger, _deepstack_injection

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

# Text benchmarks: VQA languages plus the MGSM/MSVAMP-only ones.
TEXT_BENCHMARKS = ("mgsm", "msvamp")
TEXT_MAX_NEW_TOKENS = 512
TEXT_NLLB_CODES: dict[str, str] = {
    **NLLB_CODES,
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ja": "jpn_Jpan",
    "sw": "swh_Latn",
    "th": "tha_Thai",
    "te": "tel_Telu",
}
_MATH_SYSTEM = "You are a helpful assistant that solves math problems step by step."


def build_open_ended_prompt(question: str) -> str:
    """Must match train.py's `_build_user_prompt` exactly."""
    return f"Question: {question}\nAnswer with a single word or short phrase, in English."


def build_cvqa_open_ended_prompt(question: str) -> str:
    """CVQA's own "open-ended" protocol: the question only, no options shown
    and no "single word" instruction (CVQA answers are full phrases, e.g.
    city names, not necessarily one word)."""
    return f"Question: {question}"


_OPTION_LETTERS = "ABCDEFGH"


def build_cvqa_mcq_prompt(question: str, choices: list[str]) -> str:
    """Options-shown CVQA prompt (standard LLaVA-style MCQ instruction).
    Must match Baseline/evaluate.py's `build_cvqa_mcq_prompt` exactly."""
    options = "\n".join(f"{_OPTION_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"Question: {question}\n{options}\n"
        "Answer with the option's letter from the given choices directly."
    )


def option_letter_token_ids(tokenizer, n_choices: int) -> list[int]:
    """Token id of each bare option letter ("A", "B", ...) as the first
    answer token. Raises if a letter is not a single token, since the MCQ
    scorer reads exactly one next-token logit per option."""
    ids = []
    for letter in _OPTION_LETTERS[:n_choices]:
        tok = tokenizer(letter, add_special_tokens=False)["input_ids"]
        if len(tok) != 1:
            raise ValueError(f"Option letter {letter!r} tokenizes to {tok}, expected a single token")
        ids.append(tok[0])
    return ids


def build_worldcuisines_prompt(question: str) -> str:
    """Same "single word or short phrase, in English" instruction as
    `build_open_ended_prompt` -- reinstated after checking the actual gold
    data: a row's own-language `answer` field is frequently just the
    English name left untranslated by WorldCuisines' own annotators (100%
    of the time for zh_cn/si_formal_spoken, 95%/93% for id_formal/bn,
    60-76% for ko_formal/ru_formal/jv_krama -- see `load_worldcuisines`'s
    docstring in load_evaluation_data.py). Since `answers` (built by
    `load_worldcuisines`) is *always* at least the English reference and
    only sometimes also the native one, prompting for English maximizes
    the chance of matching whichever reference actually exists, without
    costing anything: the dual-reference scorer in
    `evaluate_worldcuisines_open_ended` still credits a native-language
    answer on the minority of rows where one is available, since this is
    only an instruction, not a constraint enforced on the output.
    `question` here is WorldCuisines' own `open_ended_prompt`
    (native-language question -- see `load_worldcuisines`); this project
    adds the "Question: "/English-instruction wrapping in English for
    consistency with `build_open_ended_prompt` above, same as xGQA,
    even though the WorldCuisines authors' own harness
    (`evaluation/src/base_model.py`) sends `open_ended_prompt` verbatim
    with no added instruction.
    """
    return f"Question: {question}\nAnswer with a single word or short phrase, in English."


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

def _build_chat_messages_with_image(image: Image.Image, system_message: str, prompt: str) -> list[dict]:
    """Chat messages with the image embedded in the user turn's content.

    Rendering these through `processor.apply_chat_template` (rather than
    `tokenizer_llm.apply_chat_template` on the text alone) makes the
    processor emit the correct number of `<|image_pad|>` placeholders --
    sentineled with `<|vision_start|>`/`<|vision_end|>` -- for
    `image_grid_thw`, in the system -> user(image+question) -> assistant
    order Qwen3-VL was instruction-tuned on.
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]


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
    tokenizer_mt.src_lang = nllb_code
    enc_mt = tokenizer_mt(question, truncation=True, max_length=max_mt_seq_len, return_tensors="pt")
    input_ids_mt = enc_mt["input_ids"].to(device)
    mask_mt = enc_mt["attention_mask"].to(device)

    messages = _build_chat_messages_with_image(image, system_message, prompt)
    enc_llm = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        processor_kwargs={
            "min_pixels": visual_pixels, "max_pixels": visual_pixels,
            "truncation": True, "max_length": max_llm_seq_len,
        },
    )
    pixel_values = enc_llm["pixel_values"].to(device)
    image_grid_thw = enc_llm["image_grid_thw"].to(device)
    input_ids_query_llm = enc_llm["input_ids"].to(device)
    mask_query_llm = enc_llm["attention_mask"].to(device)
    mm_token_type_ids_query_llm = enc_llm["mm_token_type_ids"].to(device)

    gen_kw = dict(do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=tokenizer_llm.eos_token_id)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        out_texts = model.generate(
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            input_ids_mt=input_ids_mt, attention_mask_mt=mask_mt,
            input_ids_query_llm=input_ids_query_llm, mask_query_llm=mask_query_llm,
            mm_token_type_ids_query_llm=mm_token_type_ids_query_llm,
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
    tokenizer_mt.src_lang = nllb_code
    enc_mt = tokenizer_mt(question, truncation=True, max_length=max_mt_seq_len, return_tensors="pt")
    input_ids_mt = enc_mt["input_ids"].to(device)
    mask_mt = enc_mt["attention_mask"].to(device)

    prompt = build_cvqa_open_ended_prompt(question)
    messages = _build_chat_messages_with_image(image, _VQA_SYSTEM, prompt)
    enc_llm = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        processor_kwargs={
            "min_pixels": visual_pixels, "max_pixels": visual_pixels,
            "truncation": True, "max_length": max_llm_seq_len,
        },
    )
    pixel_values = enc_llm["pixel_values"].to(device)
    image_grid_thw = enc_llm["image_grid_thw"].to(device)
    input_ids_query_llm = enc_llm["input_ids"].to(device)
    mask_query_llm = enc_llm["attention_mask"].to(device)
    mm_token_type_ids_query_llm = enc_llm["mm_token_type_ids"].to(device)

    choice_ids = tokenizer_llm(choice_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    n_choice_tok = choice_ids.shape[1]
    if n_choice_tok == 0:
        return float("-inf")

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        prefix_embeds, prefix_mask, mm_token_type_ids, deepstack_visual_embeds = model._build_prefix_raw(
            pixel_values, image_grid_thw, input_ids_mt, mask_mt,
            input_ids_query_llm, mask_query_llm, mm_token_type_ids_query_llm,
        )
        choice_embeds = model.llm_embedding_layer(choice_ids).to(model.llm_dtype)
        full_embeds = torch.cat([prefix_embeds, choice_embeds], dim=1)
        full_mask = torch.cat([prefix_mask, torch.ones_like(choice_ids)], dim=1)
        full_mm_token_type_ids = torch.cat([mm_token_type_ids, torch.zeros_like(choice_ids)], dim=1)
        position_ids = model._compute_position_ids(full_mm_token_type_ids, image_grid_thw, full_mask)
        visual_pos_masks = (full_mm_token_type_ids == 1) & (full_mask != 0)
        text_model = model.model_llm.model.language_model
        with _deepstack_injection(text_model, visual_pos_masks, deepstack_visual_embeds):
            logits = model.model_llm(
                inputs_embeds=full_embeds, attention_mask=full_mask, position_ids=position_ids,
            ).logits

    # Position i's logits predict token i+1, so the logit that predicts
    # choice token k sits at (prefix_len - 1 + k).
    prefix_len = prefix_embeds.shape[1]
    choice_logits = logits[0, prefix_len - 1 : prefix_len - 1 + n_choice_tok, :].float()
    log_probs = torch.log_softmax(choice_logits, dim=-1)
    token_logprobs = log_probs.gather(1, choice_ids[0].unsqueeze(-1)).squeeze(-1)
    return (token_logprobs.sum() / n_choice_tok).item()


@torch.inference_mode()
def score_mcq_letters(
    image: Image.Image,
    question: str,
    choices: list[str],
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
) -> tuple[list[float], float]:
    """Next-token logits of the option letters after the options-shown MCQ prompt.

    Same prefix construction as `score_choice_loglikelihood`
    (`_build_prefix_raw`, NLLB fed the native question only), but the
    options are rendered into `T` and a single forward pass over the prefix
    is read at its last position -- the logit that predicts the first
    answer token.

    Returns:
        ``(letter_logits, letter_mass)``: one logit per choice, and the total
        softmax probability (over the full vocabulary) on the option letters.
    """
    tokenizer_mt.src_lang = nllb_code
    enc_mt = tokenizer_mt(question, truncation=True, max_length=max_mt_seq_len, return_tensors="pt")
    input_ids_mt = enc_mt["input_ids"].to(device)
    mask_mt = enc_mt["attention_mask"].to(device)

    prompt = build_cvqa_mcq_prompt(question, choices)
    messages = _build_chat_messages_with_image(image, _VQA_SYSTEM, prompt)
    enc_llm = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        processor_kwargs={
            "min_pixels": visual_pixels, "max_pixels": visual_pixels,
            "truncation": True, "max_length": max_llm_seq_len,
        },
    )
    pixel_values = enc_llm["pixel_values"].to(device)
    image_grid_thw = enc_llm["image_grid_thw"].to(device)
    input_ids_query_llm = enc_llm["input_ids"].to(device)
    mask_query_llm = enc_llm["attention_mask"].to(device)
    mm_token_type_ids_query_llm = enc_llm["mm_token_type_ids"].to(device)

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        prefix_embeds, prefix_mask, mm_token_type_ids, deepstack_visual_embeds = model._build_prefix_raw(
            pixel_values, image_grid_thw, input_ids_mt, mask_mt,
            input_ids_query_llm, mask_query_llm, mm_token_type_ids_query_llm,
        )
        position_ids = model._compute_position_ids(mm_token_type_ids, image_grid_thw, prefix_mask)
        visual_pos_masks = (mm_token_type_ids == 1) & (prefix_mask != 0)
        text_model = model.model_llm.model.language_model
        with _deepstack_injection(text_model, visual_pos_masks, deepstack_visual_embeds):
            logits = model.model_llm(
                inputs_embeds=prefix_embeds, attention_mask=prefix_mask, position_ids=position_ids,
            ).logits

    next_logits = logits[0, -1, :].float()
    letter_ids = torch.tensor(option_letter_token_ids(tokenizer_llm, len(choices)), device=next_logits.device)
    letter_logits = next_logits[letter_ids]
    letter_mass = torch.softmax(next_logits, dim=-1)[letter_ids].sum().item()
    return letter_logits.tolist(), letter_mass


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def open_ended_correct(pred_text: str, target: str) -> bool:
    return normalize_answer(pred_text) == normalize_answer(target)


def worldcuisines_correct(pred_text: str, answers: list[str]) -> bool:
    """WorldCuisines' own scoring convention -- `score_oe` in the authors'
    `evaluation/score/score.py`, `oe_mode="dual"`: correct if *any* reference
    answer string appears anywhere in the generated text, case-insensitive.
    Deliberately looser than `open_ended_correct`'s exact match (no
    punctuation stripping either) -- matches the source benchmark's own
    published methodology rather than this project's xGQA convention, since
    WorldCuisines answers are free-text dish names, not short GQA-style
    single tokens.
    """
    pred_lower = pred_text.lower()
    return any(ans.lower() in pred_lower for ans in answers if ans)


def cvqa_generation_correct(pred_text: str, target: str) -> bool:
    """Generation-scoring counterpart to `evaluate_cvqa_open_ended`'s
    log-likelihood ranking: correct if the gold choice text appears
    anywhere in the generated text, case-insensitive -- same
    substring-containment convention as `worldcuisines_correct`, since CVQA
    choices are free-text phrases (deity/dance-style/city names), not
    single GQA-style tokens, so exact match would be overly strict."""
    return target.lower() in pred_text.lower()


# ---------------------------------------------------------------------------
# Per-example result logging
# ---------------------------------------------------------------------------

def _open_results_writer(results_path: str | None):
    """Open *results_path* for writing, creating its parent dir if needed.

    Returns a context manager yielding the open file, or
    :func:`contextlib.nullcontext` (yielding ``None``) when *results_path*
    is ``None`` -- callers can then unconditionally check ``if rf is not
    None`` without a separate code path for the "not requested" case. Used
    by Stage3/analysis/aggregate_breakdown.py, worldcuisines_task_comparison.py
    and qualitative_cases.py, all of which read this file back as JSONL.
    """
    if results_path is None:
        return contextlib.nullcontext()
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    return open(results_path, "w", encoding="utf-8")


def _write_result(rf, row: dict, **fields) -> None:
    """Append one JSON-line record to *rf* (a no-op if *rf* is ``None``).

    Starts from a copy of *row* -- so every metadata field the loader
    attaches (question types, category, prompt_type, ...) is carried
    through automatically without evaluate.py needing to know their names
    -- then overlays *fields* (pred/correct/etc.).
    """
    if rf is None:
        return
    record = dict(row)
    record.update(fields)
    rf.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    results_path: str | None = None,
) -> float:
    nllb_code = NLLB_CODES[lang]
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    with _open_results_writer(results_path) as rf:
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
            _write_result(rf, row, pred=pred, correct=bool(ok))
            if idx < 5:
                logger.info("idx=%d question=%r pred=%r target=%r ok=%s", idx, row["query"][:80], pred, row["answer"], ok)
    acc = correct / len(rows) * 100 if rows else 0.0
    logger.info("lang=%s accuracy=%.2f%% (n=%d)", lang, acc, len(rows))
    return acc


def evaluate_worldcuisines_open_ended(
    rows: list[dict],
    lang: str,
    image_resolver,
    model, processor, tokenizer_mt, tokenizer_llm, device, amp_dtype,
    max_examples: int | None, visual_pixels: int, max_mt_seq_len: int, max_llm_seq_len: int,
    logger: logging.Logger,
    results_path: str | None = None,
) -> float:
    """Same generation path as `evaluate_open_ended` (xGQA) -- system message,
    `_VQA_SYSTEM` -- but `build_worldcuisines_prompt` wrapping (no forced
    "in English" instruction, see that function's docstring) and scored
    with `worldcuisines_correct` against `row["answers"]` (dual-reference
    list from `load_worldcuisines`) instead of exact match against a single
    `row["answer"]`. See this module's docstring for why the scoring
    convention differs from xGQA.
    """
    nllb_code = NLLB_CODES[lang]
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    with _open_results_writer(results_path) as rf:
        for idx, row in enumerate(tqdm(rows, desc=f"worldcuisines/{lang}")):
            image = image_resolver(row)
            if image is None:
                continue
            prompt = build_worldcuisines_prompt(row["query"])
            pred = generate_answer(
                image, row["query"], prompt, _VQA_SYSTEM, nllb_code,
                model, processor, tokenizer_mt, tokenizer_llm, device, amp_dtype,
                max_new_tokens=16, visual_pixels=visual_pixels,
                max_mt_seq_len=max_mt_seq_len, max_llm_seq_len=max_llm_seq_len,
            )
            ok = worldcuisines_correct(pred, row["answers"])
            correct += int(ok)
            _write_result(rf, row, pred=pred, correct=bool(ok))
            if idx < 5:
                logger.info(
                    "idx=%d question=%r pred=%r targets=%r ok=%s",
                    idx, row["query"][:80], pred, row["answers"], ok,
                )
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
    results_path: str | None = None,
) -> float:
    """CVQA's own open-ended protocol: no options shown in the prompt; score
    each candidate answer choice by log-likelihood and take the argmax."""
    nllb_code = NLLB_CODES[lang]
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    with _open_results_writer(results_path) as rf:
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
            _write_result(
                rf, row, pred=choices[pred_idx], pred_index=pred_idx,
                choice_scores=[round(s, 4) for s in scores], correct=bool(ok),
            )
            if idx < 5:
                logger.info(
                    "idx=%d question=%r choices=%r scores=%r pred=%d target=%d ok=%s",
                    idx, row["query"][:80], choices, [round(s, 3) for s in scores], pred_idx, target_idx, ok,
                )
    acc = correct / len(rows) * 100 if rows else 0.0
    logger.info("lang=%s accuracy=%.2f%% (n=%d)", lang, acc, len(rows))
    return acc


def evaluate_cvqa_generation(
    rows: list[dict],
    lang: str,
    image_resolver,
    model, processor, tokenizer_mt, tokenizer_llm, device, amp_dtype,
    max_examples: int | None, visual_pixels: int, max_mt_seq_len: int, max_llm_seq_len: int,
    logger: logging.Logger,
    results_path: str | None = None,
) -> float:
    """Diagnostic counterpart to `evaluate_cvqa_open_ended`: same eval data
    and same no-options-shown prompt (`build_cvqa_open_ended_prompt`), but
    scored via true `generate_answer` output instead of
    `score_choice_loglikelihood`'s per-choice ranking -- one generation call
    per question (not one forward pass per choice). Scored against the gold
    choice text (`choices[answer_index]`) with `cvqa_generation_correct`."""
    nllb_code = NLLB_CODES[lang]
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    with _open_results_writer(results_path) as rf:
        for idx, row in enumerate(tqdm(rows, desc=f"cvqa-generation/{lang}")):
            image = image_resolver(row)
            if image is None:
                continue
            choices = row["choices"]
            target_idx = int(row["answer_index"])
            target = choices[target_idx]
            prompt = build_cvqa_open_ended_prompt(row["query"])
            pred = generate_answer(
                image, row["query"], prompt, _VQA_SYSTEM, nllb_code,
                model, processor, tokenizer_mt, tokenizer_llm, device, amp_dtype,
                max_new_tokens=16, visual_pixels=visual_pixels,
                max_mt_seq_len=max_mt_seq_len, max_llm_seq_len=max_llm_seq_len,
            )
            ok = cvqa_generation_correct(pred, target)
            correct += int(ok)
            _write_result(rf, row, pred=pred, target=target, correct=bool(ok))
            if idx < 5:
                logger.info(
                    "idx=%d question=%r pred=%r target=%r ok=%s",
                    idx, row["query"][:80], pred, target, ok,
                )
    acc = correct / len(rows) * 100 if rows else 0.0
    logger.info("lang=%s accuracy=%.2f%% (n=%d)", lang, acc, len(rows))
    return acc


def log_mcq_summary(
    pred_letters: list[str], gold_letters: list[str], letter_masses: list[float], logger: logging.Logger,
) -> None:
    """Predicted-vs-gold letter distribution and mean letter mass for a
    `cvqa_mcq` run. Shared shape with Baseline/evaluate.py's copy."""
    if not pred_letters:
        return
    letters = sorted(set(gold_letters) | set(pred_letters))
    n = len(pred_letters)
    pred_dist = {l: round(100 * pred_letters.count(l) / n, 1) for l in letters}
    gold_dist = {l: round(100 * gold_letters.count(l) / n, 1) for l in letters}
    logger.info("mcq predicted letter %%: %s | gold letter %%: %s", pred_dist, gold_dist)
    logger.info(
        "mcq letter_mass mean=%.4f min=%.4f (share < 0.5: %.1f%%)",
        sum(letter_masses) / n, min(letter_masses), 100 * sum(m < 0.5 for m in letter_masses) / n,
    )


def evaluate_cvqa_mcq(
    rows: list[dict],
    lang: str,
    image_resolver,
    model, processor, tokenizer_mt, tokenizer_llm, device, amp_dtype,
    max_examples: int | None, visual_pixels: int, max_mt_seq_len: int, max_llm_seq_len: int,
    logger: logging.Logger,
    results_path: str | None = None,
) -> float:
    """Options-shown CVQA: lettered choices in the prompt, prediction = the
    option letter with the highest next-token logit (`score_mcq_letters`)."""
    nllb_code = NLLB_CODES[lang]
    if max_examples is not None:
        rows = rows[:max_examples]
    correct = 0
    pred_letters, gold_letters, letter_masses = [], [], []
    with _open_results_writer(results_path) as rf:
        for idx, row in enumerate(tqdm(rows, desc=f"cvqa-mcq/{lang}")):
            image = image_resolver(row)
            if image is None:
                continue
            choices = row["choices"]
            letter_logits, letter_mass = score_mcq_letters(
                image, row["query"], choices, nllb_code,
                model, processor, tokenizer_mt, tokenizer_llm, device, amp_dtype,
                visual_pixels=visual_pixels, max_mt_seq_len=max_mt_seq_len, max_llm_seq_len=max_llm_seq_len,
            )
            pred_idx = max(range(len(letter_logits)), key=lambda i: letter_logits[i])
            target_idx = int(row["answer_index"])
            ok = pred_idx == target_idx
            correct += int(ok)
            pred_letters.append(_OPTION_LETTERS[pred_idx])
            gold_letters.append(_OPTION_LETTERS[target_idx])
            letter_masses.append(letter_mass)
            _write_result(
                rf, row, pred=choices[pred_idx], pred_index=pred_idx, pred_letter=_OPTION_LETTERS[pred_idx],
                letter_logits=[round(s, 4) for s in letter_logits], letter_mass=round(letter_mass, 4),
                correct=bool(ok),
            )
            if idx < 5:
                logger.info(
                    "idx=%d question=%r choices=%r letter_logits=%r letter_mass=%.3f pred=%d target=%d ok=%s",
                    idx, row["query"][:80], choices, [round(s, 3) for s in letter_logits], letter_mass,
                    pred_idx, target_idx, ok,
                )
    log_mcq_summary(pred_letters, gold_letters, letter_masses, logger)
    acc = correct / len(rows) * 100 if rows else 0.0
    logger.info("lang=%s accuracy=%.2f%% (n=%d)", lang, acc, len(rows))
    return acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Text-only math: mgsm / msvamp (prompt + scoring identical to
# Baseline/evaluate.py's mgsm/msvamp path)
# ---------------------------------------------------------------------------

def build_math_prompt(question: str) -> str:
    return f"{question}\n\nLet's think step by step."


def extract_math_answer(text: str) -> str | None:
    m = re.search(r"(?:the\s+)?answer\s+is[:\s]+(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def math_correct(pred_text: str, target: str) -> bool:
    pred = extract_math_answer(pred_text)
    if pred is None:
        return False
    try:
        return float(pred) == float(target.replace(",", ""))
    except ValueError:
        return pred.strip() == target.strip()


def generate_text(
    question: str,
    nllb_code: str,
    model: AugmentedVisualMindMerger,
    tokenizer_mt: NllbTokenizer,
    tokenizer_llm: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_new_tokens: int,
    max_mt_seq_len: int,
) -> str:
    """Generate for one text-only question: mapped NLLB prefix + plain chat template, no image."""
    tokenizer_mt.src_lang = nllb_code
    enc_mt = tokenizer_mt(question, truncation=True, max_length=max_mt_seq_len, return_tensors="pt")

    messages = [
        {"role": "system", "content": _MATH_SYSTEM},
        {"role": "user", "content": build_math_prompt(question)},
    ]
    enc_llm = tokenizer_llm.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )

    gen_kw = dict(do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=tokenizer_llm.eos_token_id)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        out_texts = model.generate(
            pixel_values=None, image_grid_thw=None,
            input_ids_mt=enc_mt["input_ids"].to(device), attention_mask_mt=enc_mt["attention_mask"].to(device),
            input_ids_query_llm=enc_llm["input_ids"].to(device), mask_query_llm=enc_llm["attention_mask"].to(device),
            mm_token_type_ids_query_llm=None,
            tokenizer_llm=tokenizer_llm, generation_kwargs=gen_kw,
        )
    return (out_texts[0] if out_texts else "").strip()


def evaluate_math(
    rows: list[dict],
    lang: str,
    task: str,
    model: AugmentedVisualMindMerger,
    tokenizer_mt: NllbTokenizer,
    tokenizer_llm: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_examples: int | None,
    max_mt_seq_len: int,
    logger: logging.Logger,
    results_path: str | None = None,
) -> float:
    if max_examples is not None:
        rows = rows[:max_examples]
    nllb_code = TEXT_NLLB_CODES[lang]
    logger.info("task=%s lang=%s nllb=%s n=%d", task, lang, nllb_code, len(rows))
    correct = 0
    with _open_results_writer(results_path) as rf:
        for idx, row in enumerate(tqdm(rows, desc=f"{task}/{lang}")):
            pred = generate_text(
                row["question"], nllb_code, model, tokenizer_mt, tokenizer_llm,
                device, amp_dtype, TEXT_MAX_NEW_TOKENS, max_mt_seq_len,
            )
            ok = math_correct(pred, row["answer"])
            correct += int(ok)
            _write_result(rf, {"idx": idx, "question": row["question"], "answer": row["answer"]},
                          pred=pred, extracted=extract_math_answer(pred), correct=ok)
            if idx < 5:
                logger.info("idx=%d question=%r pred=%r extracted=%r target=%s ok=%s",
                            idx, row["question"][:80], pred[:300], extract_math_answer(pred), row["answer"], ok)
    acc = correct / len(rows) * 100 if rows else 0.0
    logger.info("lang=%s accuracy=%.2f%% (n=%d)", lang, acc, len(rows))
    return acc


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3b AugmentedVisualMindMerger VQA + text evaluation.")
    parser.add_argument("--benchmark", required=True,
                         choices=["xgqa", "worldcuisines_task1", "worldcuisines_task2", "cvqa", "cvqa_generation",
                                  "cvqa_mcq", *TEXT_BENCHMARKS])
    parser.add_argument("--lang", required=True,
                         help="ISO code (bn/de/ru/zh/pt/id/ko for xgqa; am/ig/om/pt/ko/mn/si/ga/bn/ru/zh for "
                              "cvqa/cvqa_generation/cvqa_mcq; bn/ru/zh/id/ko/jv/si for worldcuisines_task1/worldcuisines_task2).")
    parser.add_argument("--eval-data", required=True,
                        help="JSONL from load_evaluation_data.py (VQA) or load_text_evaluation.py (mgsm/msvamp).")
    parser.add_argument("--images-dir", default=None, help="Local GQA images dir (required for --benchmark xgqa).")
    parser.add_argument("--image-cache-dir", default=None,
                        help="Image cache dir. Defaults to <Stage3 data>/cvqa/images (cvqa) or "
                             "<Stage3 data>/worldcuisines/images (worldcuisines_task1/_task2), matching "
                             "load_evaluation_data.py's --cvqa-image-cache-dir / "
                             "--worldcuisines-image-cache-dir. Both are pre-populated on the workstation "
                             "node before this job runs -- this GPU job has no network access (Narval's "
                             "compute nodes have none), so it only ever reads the local cache.")
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
    parser.add_argument("--results-jsonl", default=None,
                         help="If given, write one JSON line per example (full row plus pred/correct, "
                              "and pred_index/choice_scores for cvqa) to this path -- input for "
                              "Stage3/analysis/aggregate_breakdown.py, worldcuisines_task_comparison.py "
                              "and qualitative_cases.py. Parent dir is created if missing.")
    args = parser.parse_args()

    if args.benchmark in TEXT_BENCHMARKS and args.lang not in TEXT_NLLB_CODES:
        parser.error(f"--lang {args.lang} has no NLLB code for {args.benchmark}; known: {sorted(TEXT_NLLB_CODES)}")
    if args.benchmark == "xgqa" and not args.images_dir:
        parser.error("--images-dir is required for --benchmark xgqa")
    if args.image_cache_dir is None:
        if args.benchmark in ("cvqa", "cvqa_generation", "cvqa_mcq"):
            args.image_cache_dir = os.path.join(SCRATCH_ROOT, "data", "cvqa", "images")
        elif args.benchmark in ("worldcuisines_task1", "worldcuisines_task2"):
            args.image_cache_dir = os.path.join(SCRATCH_ROOT, "data", "worldcuisines", "images")

    logger = setup_logging(args.benchmark)
    max_examples = 5 if args.smoke else args.max_examples

    rows = _read_jsonl(args.eval_data)
    logger.info("Loaded %d rows from %s", len(rows), args.eval_data)

    if args.benchmark in TEXT_BENCHMARKS:
        model, _processor, tokenizer_mt, tokenizer_llm, device, amp_dtype = load_model(
            args.llm_path, args.mt_path, args.mapping_ckpt, args.local_files_only, TEXT_MAX_NEW_TOKENS, logger,
        )
        # load_model only warns on key mismatches; fail loudly instead.
        ckpt = torch.load(args.mapping_ckpt, map_location="cpu")
        model.mapping.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
        evaluate_math(
            rows, args.lang, args.benchmark, model, tokenizer_mt, tokenizer_llm, device, amp_dtype,
            max_examples, args.max_mt_seq_len, logger, results_path=args.results_jsonl,
        )
        return

    model, processor, tokenizer_mt, tokenizer_llm, device, amp_dtype = load_model(
        args.llm_path, args.mt_path, args.mapping_ckpt, args.local_files_only, args.max_gen_len, logger,
    )

    if args.benchmark == "xgqa":
        resolver = lambda row: load_image_by_id(args.images_dir, row["vg_image_id"])
    elif args.benchmark in ("cvqa", "cvqa_generation", "cvqa_mcq"):
        resolver = lambda row: load_image_by_id(args.image_cache_dir, row["id"])
    else:
        resolver = lambda row: load_image_by_url(row["image_url"], args.image_cache_dir)

    common_kw = dict(
        image_resolver=resolver, model=model, processor=processor,
        tokenizer_mt=tokenizer_mt, tokenizer_llm=tokenizer_llm, device=device, amp_dtype=amp_dtype,
        max_examples=max_examples, visual_pixels=args.visual_pixels,
        max_mt_seq_len=args.max_mt_seq_len, max_llm_seq_len=args.max_llm_seq_len,
        logger=logger, results_path=args.results_jsonl,
    )

    if args.benchmark == "cvqa":
        evaluate_cvqa_open_ended(rows, args.lang, **common_kw)
    elif args.benchmark == "cvqa_generation":
        evaluate_cvqa_generation(rows, args.lang, **common_kw)
    elif args.benchmark == "cvqa_mcq":
        evaluate_cvqa_mcq(rows, args.lang, **common_kw)
    elif args.benchmark in ("worldcuisines_task1", "worldcuisines_task2"):
        evaluate_worldcuisines_open_ended(rows, args.lang, **common_kw)
    else:
        evaluate_open_ended(rows, args.lang, **common_kw)


if __name__ == "__main__":
    main()
