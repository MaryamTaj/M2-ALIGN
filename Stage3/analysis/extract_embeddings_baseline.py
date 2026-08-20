"""Extract cross-lingual embeddings from raw Qwen3-VL (no mapping layer).

Must match Stage3/analysis/extract_embeddings.py's --pooling llm-layers mode
conceptually (same benchmark/id-parallel setup, same per-layer mean-pooled
output schema so it feeds tsne_plot.py/layerwise_retrieval.py unmodified),
but there is no analogue of --pooling xm here: Baseline has no NLLB
encoder/mapping MLP producing a pre-LLM X_m span (see Baseline/evaluate.py's
module docstring -- the native-language question goes straight into
Qwen3-VL's own chat template), so the only representation to probe is the
frozen decoder's own hidden states, pooled over the question-token span at
each layer. This is the "before" reference curve for tsne_plot.py/
layerwise_retrieval.py -- without it, a claim like "M2RB's mapping produces
cross-lingual alignment" has no baseline to show that Qwen3-VL didn't
already have comparable alignment zero-shot.

Locating the question span
---------------------------
Unlike extract_embeddings.py (which builds the prefix manually and so knows
exactly where the mapped span sits), this script gets a single opaque
`input_ids` sequence back from `processor.apply_chat_template` with the
image and instruction-wrapped question rendered together. The question's
own token span is found by tokenizing the question text alone and searching
for it as a contiguous subsequence of the full `input_ids` -- exact match
first, then progressively trimming up to 2 tokens off either end (BPE
merges can differ right at a text boundary, e.g. "Question: <q>\\n" fusing
the first question token with the preceding colon+space). Examples where no
match is found at all (rare -- only for degenerate/empty questions after
tokenization) are skipped and logged, not silently mis-pooled.

Because DeepStack visual-feature injection only fires when `pixel_values`
flows through Qwen3VLForConditionalGeneration's own forward pass (see
model.py's `_deepstack_injection` docstring -- M2RB has to patch this back
in manually specifically because it bypasses that path), this script calls
the model directly with `output_hidden_states=True` and no extra plumbing:
DeepStack already fires natively here, exactly as it does during Baseline's
own generate_answer/score_choice_loglikelihood.

Saves the same .pt schema as extract_embeddings.py's llm-layers mode:
    {"ids": [...], "languages": [...], "embeddings": FloatTensor[N, L, D],
     "pooling": "llm-layers", "model": "baseline", "benchmark": str}

Example
-------
    python Stage3/analysis/extract_embeddings_baseline.py \\
        --benchmark xgqa --languages bn,de,ru,zh,pt,id,ko \\
        --eval-data-dir $SCRATCH/M2-ALIGN/Stage3/data/xgqa \\
        --images-dir /path/to/gqa/images \\
        --max-examples-per-lang 150 \\
        --out $SCRATCH/M2-ALIGN/Stage3/analysis_out/layers_baseline_xgqa.pt

    python Stage3/analysis/layerwise_retrieval.py \\
        --embeddings baseline=.../layers_baseline_xgqa.pt \\
        --embeddings stage3=.../layers_stage3b_xgqa.pt \\
        --plot .../xgqa_layerwise_retrieval_with_baseline.png
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_REPO_ROOT, "Baseline"))

import torch  # noqa: E402

from evaluate import (  # noqa: E402
    _VQA_SYSTEM, build_open_ended_prompt, build_worldcuisines_prompt,
    load_image_by_id, load_image_by_url, load_model, setup_logging,
)


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prompt_for(benchmark: str, question: str) -> str:
    return build_worldcuisines_prompt(question) if benchmark.startswith("worldcuisines") else build_open_ended_prompt(question)


def _find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int] | None:
    """Locate `needle` as a contiguous run in `haystack`; see this module's
    docstring for why an exact match can fail right at the boundary and how
    the trimmed retry handles it."""
    n = len(needle)
    if n == 0:
        return None
    for trim_left in range(3):
        for trim_right in range(3):
            sub = needle[trim_left: n - trim_right] if trim_right else needle[trim_left:]
            m = len(sub)
            if m == 0:
                continue
            for i in range(len(haystack) - m + 1):
                if haystack[i:i + m] == sub:
                    return i, i + m
    return None


@torch.inference_mode()
def _embed_llm_layers(image, question: str, prompt: str, model, processor, device, amp_dtype) -> torch.Tensor | None:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _VQA_SYSTEM}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    question_ids = processor.tokenizer(question, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
    span = _find_subsequence(inputs["input_ids"][0].tolist(), question_ids)
    if span is None:
        return None
    start, end = span

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        out = model(**inputs, output_hidden_states=True)
    layer_embeds = [h[0, start:end, :].float().mean(dim=0).cpu() for h in out.hidden_states]
    return torch.stack(layer_embeds)  # [num_layers+1, dim]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=["xgqa", "worldcuisines_task1", "worldcuisines_task2"],
                         help="cvqa is excluded: its row ids aren't shared across languages, so there's no "
                              "parallel content to align/retrieve.")
    parser.add_argument("--languages", required=True, help="Comma-separated ISO codes, e.g. bn,de,ru,zh,pt,id,ko.")
    parser.add_argument("--eval-data-dir", required=True,
                         help="Dir containing '{lang}.jsonl' per language, as written by "
                              "load_evaluation_data.py (e.g. .../data/xgqa or .../data/worldcuisines/task1).")
    parser.add_argument("--images-dir", default=None, help="Local GQA images dir (required for --benchmark xgqa).")
    parser.add_argument("--image-cache-dir", default=None, help="Local image cache dir (required for worldcuisines).")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-examples-per-lang", type=int, default=300,
                         help="Subsample per language (applied before intersecting ids) -- one GPU forward pass per example.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.benchmark == "xgqa" and not args.images_dir:
        parser.error("--images-dir is required for --benchmark xgqa")
    if args.benchmark.startswith("worldcuisines") and not args.image_cache_dir:
        parser.error("--image-cache-dir is required for worldcuisines benchmarks")

    logger = setup_logging(f"extract_embeddings_baseline_{args.benchmark}")
    languages = args.languages.split(",")
    rng = random.Random(args.seed)

    rows_by_lang: dict[str, dict[str, dict]] = {}
    for lang in languages:
        path = os.path.join(args.eval_data_dir, f"{lang}.jsonl")
        rows = _read_jsonl(path)
        rng.shuffle(rows)
        rows = rows[: args.max_examples_per_lang]
        rows_by_lang[lang] = {r["id"]: r for r in rows}
        logger.info("lang=%s: loaded %d rows (post-subsample) from %s", lang, len(rows), path)

    shared_ids = set.intersection(*(set(d) for d in rows_by_lang.values())) if rows_by_lang else set()
    shared_ids = sorted(shared_ids)
    logger.info("Shared ids across all %d languages: %d", len(languages), len(shared_ids))
    if not shared_ids:
        logger.error("No ids shared across all requested languages -- nothing to embed.")
        sys.exit(1)

    model, processor, device = load_model(args.model_id, local_files_only=args.local_files_only)
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    out_ids, out_langs, out_embeds = [], [], []
    n_skipped = 0
    for lang in languages:
        n_embedded = 0
        for id_ in shared_ids:
            row = rows_by_lang[lang][id_]
            if args.benchmark == "xgqa":
                image = load_image_by_id(args.images_dir, row["vg_image_id"])
            else:
                image = load_image_by_url(row["image_url"], args.image_cache_dir)
            if image is None:
                n_skipped += 1
                continue
            prompt = _prompt_for(args.benchmark, row["query"])
            emb = _embed_llm_layers(image, row["query"], prompt, model, processor, device, amp_dtype)
            if emb is None:
                n_skipped += 1
                continue
            out_ids.append(id_)
            out_langs.append(lang)
            out_embeds.append(emb)
            n_embedded += 1
        logger.info("lang=%s: embedded %d/%d shared examples", lang, n_embedded, len(shared_ids))
    if n_skipped:
        logger.warning("Skipped %d examples (unresolved image or question span)", n_skipped)

    embeddings = torch.stack(out_embeds)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "ids": out_ids, "languages": out_langs, "embeddings": embeddings,
        "pooling": "llm-layers", "model": "baseline", "benchmark": args.benchmark,
    }, args.out)
    logger.info("Saved %s: embeddings shape=%s", args.out, tuple(embeddings.shape))


if __name__ == "__main__":
    main()
