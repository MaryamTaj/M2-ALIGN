# coding=utf-8
"""
Evaluate the Stage 2 augmented MindMerger checkpoint on MMLU-ProX.

Unlike `Stage1/run_evaluating.py`, this script feeds the LLM exactly the
prefix it was trained on at Stage 2:

    [BOS] + X_m + [end_boundary] + T

where
    X_m = mapping(NLLB_encoder(prompt_in_source_language))
    T   = LLM_token_embedding(prompt)        # same prompt, LLM-tokenised

This matches the augmentation stage of the MindMerger paper (NeurIPS 2024)
and avoids the train/eval mismatch that occurs when Stage 1's eval path
(which omits T) is used with a Stage 2 checkpoint.

Decoding hyperparameters are kept consistent with Baseline/mmlu_prox.py and
Stage1/run_evaluating.py for direct comparability.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import string
import time
from typing import Iterable

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, NllbTokenizer

from modeling_augmentation import AugmentedMindMerger


# #region agent log (debug session 0a4016)
_DBG_SESSION_ID = "0a4016"
_DBG_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DBG_REPO_ROOT = os.path.normpath(os.path.join(_DBG_SCRIPT_DIR, ".."))
_DBG_LOG_PATH = os.path.join(_DBG_REPO_ROOT, ".cursor", f"debug-{_DBG_SESSION_ID}.log")


def _dbg_log(location: str, message: str, data: dict | None = None,
             hypothesisId: str | None = None, runId: str = "stage2-eval") -> None:
    """Append an NDJSON debug entry and mirror it to stdout for SLURM capture."""
    ts = int(time.time() * 1000)
    payload = {
        "sessionId": _DBG_SESSION_ID,
        "id": f"log_{ts}_{location}",
        "timestamp": ts,
        "runId": runId,
        "hypothesisId": hypothesisId,
        "location": location,
        "message": message,
        "data": data or {},
    }
    line = json.dumps(payload, ensure_ascii=False, default=str)
    try:
        os.makedirs(os.path.dirname(_DBG_LOG_PATH), exist_ok=True)
        with open(_DBG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(f"[DBG-RUNTIME] {line}", flush=True)
# #endregion


TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 40
PRESENCE_PENALTY = 2.0
N_SHOT = 5
FEWSHOT_SEED = 1234
MAX_NEW_TOKENS_FOR_MCQ = 6


# MMLU-ProX config name -> human-readable language key used by the NLLB lang map.
MMLU_TO_SOURCE_LANGUAGE = {
    "sw": "Swahili",
    "wo": "Wolof",
    "yo": "Yoruba",
    "fr": "French",
}

NLLB_LANG_MAP = {
    "English": "eng_Latn",
    "Swahili": "swh_Latn",
    "Yoruba": "yor_Latn",
    "Wolof": "wol_Latn",
    "French": "fra_Latn",
}


# ---------------------------------------------------------------------------
# Prompt construction (kept in sync with Stage1/run_evaluating.py).
# ---------------------------------------------------------------------------

def extract_options(sample: dict) -> tuple[list[str], list[str]]:
    option_items = []
    for k, v in sample.items():
        if k.startswith("option_") and v is not None:
            idx = int(k.split("_")[1])
            option_items.append((idx, v))
    option_items.sort(key=lambda x: x[0])
    texts = [v for _, v in option_items]
    letters = list(string.ascii_uppercase[: len(texts)])
    return letters, texts


def format_options_block(letters: list[str], texts: list[str]) -> str:
    return "\n".join([f"{L}. {t}" for L, t in zip(letters, texts)])


def qwen_eval_block(question: str, options_block: str, answer_letter: str | None = None) -> str:
    base = (
        "Respond with only the letter of the correct option.\n"
        f"Question: {question} Possible answer choices:\n"
        f"{options_block}\n"
        "The best answer is:"
    )
    if answer_letter is not None:
        return base + f" {answer_letter}\n"
    return base


def build_fewshot_prompt(demo_samples: list, test_sample: dict) -> tuple[str, list[str]]:
    blocks = []
    for s in demo_samples:
        letters, texts = extract_options(s)
        options_block = format_options_block(letters, texts)
        blocks.append(qwen_eval_block(s["question"], options_block, s["answer"]))

    letters, texts = extract_options(test_sample)
    options_block = format_options_block(letters, texts)
    blocks.append(qwen_eval_block(test_sample["question"], options_block, answer_letter=None))
    return "\n\n".join(blocks), letters


def build_nllb_test_text(test_sample: dict) -> str:
    # X_m is supposed to be a *target-language* sentence embedding. We must NOT
    # feed it the full 5-shot prompt: that prompt is mostly English instruction
    # scaffolding and English demonstrations, which (a) mismatches NLLB's
    # ``src_lang`` setting and (b) routinely overflows NLLB's max_seq_len so
    # the actual test question gets right-truncated away. Encode only the test
    # question and its options - both of which are in the target language.
    letters, texts = extract_options(test_sample)
    options_block = format_options_block(letters, texts)
    return f"{test_sample['question']}\n{options_block}"


# ---------------------------------------------------------------------------
# Tokenisation helpers (kept in sync with Stage2/run_augmentation.py so that
# eval-time token streams match what the model saw during training).
# ---------------------------------------------------------------------------

def mt_input_features(
    texts: Iterable[str],
    langs: Iterable[str],
    tokenizer_mt: NllbTokenizer,
    max_seq_len: int,
    device: torch.device,
):
    ids, masks = [], []
    for text, lang in zip(texts, langs):
        tokenizer_mt.src_lang = NLLB_LANG_MAP[lang]
        enc = tokenizer_mt(text, truncation=True, max_length=max_seq_len, padding=False)
        ids.append(enc["input_ids"])
        masks.append(enc["attention_mask"])

    max_len = max(len(x) for x in ids)
    pad_id = tokenizer_mt.pad_token_id
    for i in range(len(ids)):
        while len(ids[i]) < max_len:
            ids[i].append(pad_id)
            masks[i].append(0)

    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def llm_input_features(
    texts: Iterable[str],
    tokenizer_llm: AutoTokenizer,
    max_seq_len: int,
    add_bos: bool,
    add_eos: bool,
    device: torch.device,
):
    # Match the training-time helper exactly so that `T` is identical to the
    # token stream the mapping was trained against.
    if hasattr(tokenizer_llm, "add_bos_token"):
        tokenizer_llm.add_bos_token = add_bos
    if hasattr(tokenizer_llm, "add_eos_token"):
        tokenizer_llm.add_eos_token = add_eos
    enc = tokenizer_llm(
        list(texts),
        truncation=True,
        max_length=max_seq_len,
        padding=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ---------------------------------------------------------------------------
# Per-example MCQ scoring.
# ---------------------------------------------------------------------------

_PICK_CHOICE_LOGGED_GEN_KW = False


@torch.inference_mode()
def pick_choice(
    model: AugmentedMindMerger,
    tokenizer_mt: NllbTokenizer,
    tokenizer_llm: AutoTokenizer,
    prompt: str,
    nllb_text: str,
    source_language: str,
    max_mt_seq_len: int,
    max_llm_seq_len: int,
    choices: list[str],
    amp_dtype: torch.dtype,
    device: torch.device,
) -> tuple[str, str]:
    # NLLB encodes only the test question + options (in the target language).
    # The LLM gets the full 5-shot prompt with a much larger cap and left-side
    # truncation (set on the tokenizer in ``evaluate``), so if the prompt still
    # overflows we drop demonstrations from the front rather than the test
    # question at the end.
    input_ids_mt, mask_mt = mt_input_features(
        [nllb_text], [source_language], tokenizer_mt, max_mt_seq_len, device
    )
    input_ids_query_llm, mask_query_llm = llm_input_features(
        [prompt], tokenizer_llm, max_llm_seq_len, add_bos=False, add_eos=False, device=device
    )

    gen_kw = dict(
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_new_tokens=MAX_NEW_TOKENS_FOR_MCQ,
        eos_token_id=tokenizer_llm.eos_token_id,
    )

    # #region agent log (H4: capture exact decoding config + prefix lengths once)
    global _PICK_CHOICE_LOGGED_GEN_KW
    if not _PICK_CHOICE_LOGGED_GEN_KW:
        _PICK_CHOICE_LOGGED_GEN_KW = True
        _dbg_log(
            "run_evaluating.py:pick_choice_first_call",
            "First eval call: decoding kwargs + prefix-length composition for sanity check.",
            data={
                "gen_kw": {**gen_kw, "presence_penalty": PRESENCE_PENALTY},
                "amp_dtype": str(amp_dtype),
                "input_ids_mt_shape": list(input_ids_mt.shape),
                "input_ids_query_llm_shape": list(input_ids_query_llm.shape),
                "max_mt_seq_len": max_mt_seq_len,
                "max_llm_seq_len": max_llm_seq_len,
                "llm_truncation_side": getattr(tokenizer_llm, "truncation_side", None),
                "max_new_tokens_for_mcq": MAX_NEW_TOKENS_FOR_MCQ,
                "n_choices": len(choices),
                "source_language": source_language,
                "nllb_text_head": (nllb_text or "")[:200],
            },
            hypothesisId="H4",
        )
    # #endregion

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        out_texts = model.generate(
            input_ids_mt=input_ids_mt,
            attention_mask_mt=mask_mt,
            input_ids_query_llm=input_ids_query_llm,
            mask_query_llm=mask_query_llm,
            tokenizer_llm=tokenizer_llm,
            generation_kwargs=gen_kw,
            presence_penalty=PRESENCE_PENALTY,
        )
    out_text = (out_texts[0] if out_texts else "").strip()
    for ch in out_text:
        if ch in choices:
            return ch, out_text
    return choices[0], out_text


# ---------------------------------------------------------------------------
# Evaluation driver.
# ---------------------------------------------------------------------------

def evaluate(
    *,
    llm_path: str,
    mt_path: str,
    mapping_ckpt: str,
    langs: list[str],
    local_files_only: bool,
    max_mt_seq_len: int,
    max_llm_seq_len: int,
    max_gen_len: int,
    max_test_examples: int | None,
    max_val_examples: int | None,
    print_sample_count: int = 20,
):
    device = torch.device("cuda")
    cap_major = torch.cuda.get_device_capability(0)[0]
    amp_dtype = torch.bfloat16 if cap_major >= 8 else torch.float16

    tokenizer_llm = AutoTokenizer.from_pretrained(
        llm_path, use_fast=False, local_files_only=local_files_only
    )
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"
    # Left-truncate the LLM prompt so that, if a 5-shot MMLU-ProX prompt still
    # exceeds ``max_llm_seq_len``, we drop demonstrations from the front rather
    # than slicing off the test question at the end.
    tokenizer_llm.truncation_side = "left"

    tokenizer_mt = NllbTokenizer.from_pretrained(mt_path, local_files_only=local_files_only)

    # #region agent log (H5: BOS embedding may actually be EOS for Qwen3-VL)
    bos_id = tokenizer_llm.bos_token_id
    eos_id = tokenizer_llm.eos_token_id
    pad_id = tokenizer_llm.pad_token_id
    def _safe_decode(tok_id):
        if tok_id is None:
            return None
        try:
            return tokenizer_llm.decode([tok_id], skip_special_tokens=False)
        except Exception as exc:
            return f"<decode-fail:{exc}>"
    _dbg_log(
        "run_evaluating.py:tokenizer_load",
        "LLM tokenizer special-token IDs (Qwen3-VL has no native BOS; check fallback chain).",
        data={
            "bos_token_id": bos_id,
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
            "bos_token_repr": _safe_decode(bos_id),
            "eos_token_repr": _safe_decode(eos_id),
            "pad_token_repr": _safe_decode(pad_id),
            "bos_eq_eos": bos_id == eos_id,
            "bos_eq_pad": bos_id == pad_id,
            "padding_side": tokenizer_llm.padding_side,
            "vocab_size": getattr(tokenizer_llm, "vocab_size", None),
        },
        hypothesisId="H5",
    )
    # #endregion

    # #region agent log (H1: peek at task-specialization training data format)
    try:
        peek_path_en = os.path.join(_DBG_REPO_ROOT, "Stage2", "data", "task_specialization_en.jsonl")
        peek_rows_en = []
        if os.path.isfile(peek_path_en):
            with open(peek_path_en, "r", encoding="utf-8") as f:
                for i, raw in enumerate(f):
                    if i >= 5:
                        break
                    raw = raw.strip().lstrip("\ufeff")
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                        peek_rows_en.append({
                            "query_head": (row.get("query") or "")[:200],
                            "answer": (row.get("answer") or "")[:200],
                            "answer_len_chars": len(row.get("answer") or ""),
                            "answer_looks_like_letter": (
                                len((row.get("answer") or "").strip()) == 1
                                and (row.get("answer") or "").strip().upper() in string.ascii_uppercase
                            ),
                            "source_dataset": row.get("source_dataset"),
                        })
                    except Exception:
                        peek_rows_en.append({"raw_head": raw[:200]})
        _dbg_log(
            "run_evaluating.py:training_data_peek",
            "First 5 rows of task_specialization_en.jsonl (used to train Stage 2 mapping).",
            data={"path": peek_path_en, "rows": peek_rows_en},
            hypothesisId="H1",
        )
    except Exception as exc:
        _dbg_log("run_evaluating.py:training_data_peek", f"peek failed: {exc}", hypothesisId="H1")
    # #endregion

    # Stay forward-compatible with model defs that may not yet accept
    # `local_files_only` as a keyword argument.
    sig = inspect.signature(AugmentedMindMerger.__init__)
    extra_kwargs = {}
    if "local_files_only" in sig.parameters:
        extra_kwargs["local_files_only"] = local_files_only

    model = AugmentedMindMerger(
        mt_path=mt_path,
        llm_path=llm_path,
        max_gen_len=max_gen_len,
        llm_bos_token_id=tokenizer_llm.bos_token_id,
        llm_pad_token_id=tokenizer_llm.pad_token_id,
        **extra_kwargs,
    )

    ckpt = torch.load(mapping_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.mapping.load_state_dict(state_dict, strict=False)
    print(f"Loaded Stage 2 mapping from: {mapping_ckpt}")
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    # #region agent log (H6: silent ckpt key drift would leave mapping at random init)
    try:
        boundary_norm = float(model.mapping.end_boundary.detach().float().norm().item())
        first_w_norm = float(model.mapping.mlp.linear1.weight.detach().float().norm().item())
    except Exception as exc:
        boundary_norm = None
        first_w_norm = f"err:{exc}"
    _dbg_log(
        "run_evaluating.py:mapping_load",
        "Stage 2 mapping checkpoint load summary; weight norms gauge whether init-from-checkpoint actually applied.",
        data={
            "mapping_ckpt": mapping_ckpt,
            "ckpt_meta_step": ckpt.get("step") if isinstance(ckpt, dict) else None,
            "ckpt_meta_loss": ckpt.get("loss") if isinstance(ckpt, dict) else None,
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "missing_sample": list(missing)[:5],
            "unexpected_sample": list(unexpected)[:5],
            "end_boundary_norm": boundary_norm,
            "linear1_weight_norm": first_w_norm,
        },
        hypothesisId="H6",
    )
    # #endregion

    model.model_mt.to(device)
    model.model_llm.to(device)
    model.mapping.to(device)
    model.eval()

    results: dict[str, float] = {}
    total_correct_all = 0
    total_all = 0
    rng = random.Random(FEWSHOT_SEED)

    for lang in langs:
        if lang not in MMLU_TO_SOURCE_LANGUAGE:
            raise ValueError(f"Unsupported language {lang!r}; add to MMLU_TO_SOURCE_LANGUAGE.")

        source_language = MMLU_TO_SOURCE_LANGUAGE[lang]

        test_ds = load_dataset(
            "li-lab/MMLU-ProX",
            lang,
            split="test",
            download_mode="reuse_dataset_if_exists",
        )
        if max_test_examples is not None:
            test_ds = test_ds.select(range(min(max_test_examples, len(test_ds))))

        val_ds = load_dataset(
            "li-lab/MMLU-ProX",
            lang,
            split="validation",
            download_mode="reuse_dataset_if_exists",
        )
        if max_val_examples is not None:
            val_ds = val_ds.select(range(min(max_val_examples, len(val_ds))))

        idxs = list(range(len(val_ds)))
        rng.shuffle(idxs)
        demo_samples = [val_ds[i] for i in idxs[:N_SHOT]]

        correct = 0
        total = len(test_ds)
        print(f"\nEvaluating language: {lang} ({source_language}, {total} examples) with {N_SHOT}-shot")

        debug_rows: list[tuple[str, str, str, str, bool]] = []
        # #region agent log (H1/H4: capture first-N raw model outputs to NDJSON per language)
        ndjson_capture_n = 8
        ndjson_seen = 0
        # #endregion
        for sample in tqdm(test_ds):
            prompt, choice_letters = build_fewshot_prompt(demo_samples, sample)
            nllb_text = build_nllb_test_text(sample)
            pred, raw_out = pick_choice(
                model,
                tokenizer_mt,
                tokenizer_llm,
                prompt,
                nllb_text,
                source_language,
                max_mt_seq_len,
                max_llm_seq_len,
                choice_letters,
                amp_dtype,
                device,
            )
            ok = pred == sample["answer"]
            if ok:
                correct += 1
            if print_sample_count > 0 and len(debug_rows) < print_sample_count:
                debug_rows.append((prompt, raw_out, pred, sample["answer"], ok))

            # #region agent log (H1/H4: per-sample structured output dump)
            if ndjson_seen < ndjson_capture_n:
                ndjson_seen += 1
                first_letter_of_raw = next((c for c in (raw_out or "") if c.upper() in string.ascii_uppercase), None)
                _dbg_log(
                    "run_evaluating.py:eval_sample",
                    f"Eval sample for lang={lang}",
                    data={
                        "lang": lang,
                        "source_language": source_language,
                        "sample_idx_in_lang": ndjson_seen,
                        "prompt_tail": prompt[-400:],
                        "raw_out": raw_out,
                        "raw_out_repr": repr(raw_out),
                        "raw_out_len_chars": len(raw_out or ""),
                        "first_alpha_char_of_raw": first_letter_of_raw,
                        "predicted_letter": pred,
                        "target_letter": sample.get("answer"),
                        "choice_letters": choice_letters,
                        "correct": ok,
                    },
                    hypothesisId="H1+H4",
                )
            # #endregion

        if debug_rows:
            print(f"\n=== Printed examples (n={len(debug_rows)}) for MMLU-ProX lang={lang} ===")
            for i, (prompt, raw_out, pred, target, correct_i) in enumerate(debug_rows, 1):
                print(
                    f"\n--- example {i}/{len(debug_rows)} ---\n"
                    f"INPUT (few-shot MCQ prompt):\n{prompt}\n\n"
                    f"MODEL OUTPUT (raw decode):\n{raw_out!r}\n\n"
                    f"PREDICTED LETTER: {pred}\n"
                    f"TARGET LETTER: {target}\n"
                    f"CORRECT: {correct_i}\n",
                    flush=True,
                )

        acc = correct / total * 100
        results[lang] = acc
        total_correct_all += correct
        total_all += total
        print(f"Accuracy for {lang}: {acc:.2f}%")

    macro_avg = sum(results.values()) / len(results)
    micro_avg = (total_correct_all / total_all) * 100
    print("\n=== MMLU-ProX Summary (Stage 2 / AugmentedMindMerger) ===")
    print(f"Macro-average over {len(results)} languages: {macro_avg:.2f}%")
    print(f"Micro-average over all examples: {micro_avg:.2f}%")
    return results, macro_avg, micro_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HF id or local snapshot directory for Qwen3-VL.",
    )
    parser.add_argument(
        "--mt-path",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="HF id or local snapshot directory for the NLLB encoder.",
    )
    parser.add_argument(
        "--mapping-ckpt",
        type=str,
        default="Stage2/outputs/augmentation/mapping/pytorch_model.bin",
        help="Path to the Stage 2 mapping checkpoint (pytorch_model.bin).",
    )
    parser.add_argument("--langs", nargs="*", default=["sw", "wo", "yo"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help=(
            "Max sequence length for the NLLB encoder (X_m). At eval time NLLB "
            "sees only the test question + options, so 256 is plenty."
        ),
    )
    parser.add_argument(
        "--max-llm-seq-len",
        type=int,
        default=4096,
        help=(
            "Max sequence length for the Qwen3-VL prompt (T). 5-shot MMLU-ProX "
            "prompts routinely exceed 256 tokens; 4096 keeps the whole prompt "
            "intact for typical questions. truncation_side='left' is set on the "
            "tokenizer so any residual overflow drops demonstrations rather "
            "than the test question."
        ),
    )
    parser.add_argument("--max-gen-len", type=int, default=256)
    parser.add_argument(
        "--print-sample-count",
        type=int,
        default=20,
        help="Print this many (input, raw output, target letter) triples per language; 0 disables.",
    )
    args = parser.parse_args()

    max_test = args.max_test_examples
    max_val = args.max_val_examples
    if args.smoke:
        args.langs = ["sw"]
        max_test = 5 if max_test is None else max_test
        max_val = 20 if max_val is None else max_val

    evaluate(
        llm_path=args.llm_path,
        mt_path=args.mt_path,
        mapping_ckpt=args.mapping_ckpt,
        langs=args.langs,
        local_files_only=args.local_files_only,
        max_mt_seq_len=args.max_seq_len,
        max_llm_seq_len=args.max_llm_seq_len,
        max_gen_len=args.max_gen_len,
        max_test_examples=max_test,
        max_val_examples=max_val,
        print_sample_count=args.print_sample_count,
    )


if __name__ == "__main__":
    main()
