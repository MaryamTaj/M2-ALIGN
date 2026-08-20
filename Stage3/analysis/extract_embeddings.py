"""Extract cross-lingual embeddings from a trained mapping checkpoint.

Feeds this project's own parallel-by-id evaluation data (xGQA's testdev
questions share a `qid` across its 8 languages; WorldCuisines' rows share a
`qa_id`-derived `id` across its languages/tasks -- see
load_evaluation_data.py) through a Stage 1/2/3 mapping checkpoint and saves
one embedding per (id, language), restricted to ids present in *every*
requested language so downstream analysis (t-SNE, layer-wise retrieval)
sees true parallel content. Two pooling modes:

  --pooling xm (default, cheap: NLLB encoder + mapping MLP only, no image,
      no LLM forward pass needed)
      The mapped embedding X_m = mapping(encoder_mt(question)), mean-pooled
      over its (unpadded) token positions. This is literally the space the
      mapping's training objective aligns to the LLM's own text embeddings
      -- the most direct probe of "did this stage learn a language-agnostic
      representation of the question."

  --pooling llm-layers (fuller, needs the image + a GPU forward through the
      frozen LLM decoder)
      Builds the real inference-time prefix [BOS, X_m, end_boundary, T]
      (T = image + untranslated question, exactly as evaluate.py's
      generate_answer does) and runs it through the frozen Qwen3-VL decoder
      with output_hidden_states=True (DeepStack injection included, via
      model._deepstack_injection -- see model.py). Pools each layer's
      hidden state over the X_m span specifically (not the image or query
      tokens), giving one [num_layers+1, llm_dim] embedding per example:
      how the mapped question's representation evolves with depth, through
      the same early layers DeepStack injects visual features into.

Saves a single .pt file (torch.save) with:
    {"ids": [...], "languages": [...], "embeddings": FloatTensor,
     "pooling": "xm"|"llm-layers", "mapping_ckpt": str, "benchmark": str}
`embeddings` is `[N, llm_dim]` for "xm", `[N, num_layers+1, llm_dim]` for
"llm-layers", with `ids[i]`/`languages[i]` naming row i.

Example
-------
    # Cheap: X_m alignment for Stage1 vs Stage2 vs Stage3 (repeat per stage)
    python Stage3/analysis/extract_embeddings.py \\
        --benchmark xgqa --languages bn,de,ru,zh,pt,id,ko \\
        --eval-data-dir $SCRATCH/M2-ALIGN/Stage3/data/xgqa \\
        --mapping-ckpt $SCRATCH/M2-ALIGN/Stage3/outputs/bn/pytorch_model.bin \\
        --pooling xm --max-examples-per-lang 300 \\
        --out $SCRATCH/M2-ALIGN/Stage3/analysis_out/xm_stage3_xgqa.pt

    # Fuller: layer-wise, needs GPU images
    python Stage3/analysis/extract_embeddings.py \\
        --benchmark xgqa --languages bn,de,ru,zh,pt,id,ko \\
        --eval-data-dir $SCRATCH/M2-ALIGN/Stage3/data/xgqa \\
        --images-dir /path/to/gqa/images \\
        --mapping-ckpt $SCRATCH/M2-ALIGN/Stage3/outputs/bn/pytorch_model.bin \\
        --pooling llm-layers --max-examples-per-lang 150 \\
        --out $SCRATCH/M2-ALIGN/Stage3/analysis_out/layers_stage3_xgqa.pt

The output feeds tsne_plot.py and layerwise_retrieval.py directly.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from evaluate import (  # noqa: E402
    NLLB_CODES, _VQA_SYSTEM, _build_chat_messages_with_image,
    build_open_ended_prompt, build_worldcuisines_prompt,
    load_image_by_id, load_image_by_url, load_model, setup_logging,
)
from model import _deepstack_injection  # noqa: E402

# English isn't in evaluate.py's NLLB_CODES (the eval pipeline never runs
# "en" through the NLLB path -- English questions need no translation), but
# xGQA's own XGQA_LANGS includes "en" and it's a useful retrieval anchor, so
# this script adds it locally rather than touching the shared eval mapping.
_EXTRA_NLLB_CODES = {"en": "eng_Latn"}


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _nllb_code(lang: str) -> str:
    return NLLB_CODES.get(lang) or _EXTRA_NLLB_CODES[lang]


def _prompt_for(benchmark: str, question: str) -> str:
    return build_worldcuisines_prompt(question) if benchmark.startswith("worldcuisines") else build_open_ended_prompt(question)


@torch.inference_mode()
def _embed_xm(question: str, nllb_code: str, model, tokenizer_mt, device, amp_dtype, max_mt_seq_len: int) -> torch.Tensor:
    tokenizer_mt.src_lang = nllb_code
    enc = tokenizer_mt(question, truncation=True, max_length=max_mt_seq_len, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        mt_out = model.encoder_mt(input_ids=input_ids, attention_mask=mask, output_hidden_states=False)
        x_m = model.mapping(mt_out[0])
    mask_f = mask.unsqueeze(-1).float()
    pooled = (x_m.float() * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
    return pooled[0].cpu()


@torch.inference_mode()
def _embed_llm_layers(
    image, question: str, benchmark: str, nllb_code: str,
    model, processor, tokenizer_mt, device, amp_dtype,
    visual_pixels: int, max_mt_seq_len: int, max_llm_seq_len: int,
) -> torch.Tensor:
    tokenizer_mt.src_lang = nllb_code
    enc_mt = tokenizer_mt(question, truncation=True, max_length=max_mt_seq_len, return_tensors="pt")
    input_ids_mt = enc_mt["input_ids"].to(device)
    mask_mt = enc_mt["attention_mask"].to(device)
    mt_seq_len = input_ids_mt.shape[1]

    prompt = _prompt_for(benchmark, question)
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
        llm_embeds, llm_mask, mm_token_type_ids, deepstack = model._build_prefix_raw(
            pixel_values, image_grid_thw, input_ids_mt, mask_mt,
            input_ids_query_llm, mask_query_llm, mm_token_type_ids_query_llm,
        )
        position_ids = model._compute_position_ids(mm_token_type_ids, image_grid_thw, llm_mask)
        visual_pos_masks = (mm_token_type_ids == 1) & (llm_mask != 0)
        text_model = model.model_llm.model.language_model
        with _deepstack_injection(text_model, visual_pos_masks, deepstack):
            out = text_model(
                inputs_embeds=llm_embeds, attention_mask=llm_mask,
                position_ids=position_ids, output_hidden_states=True,
            )
    # Layout is [BOS(1), X_m(mt_seq_len), end_boundary(1), T(...)] -- batch
    # size 1 so there's no padding to account for (mask_mt/llm_mask are all
    # ones for a single unpadded example).
    xm_start, xm_end = 1, 1 + mt_seq_len
    layer_embeds = [h[0, xm_start:xm_end, :].float().mean(dim=0).cpu() for h in out.hidden_states]
    return torch.stack(layer_embeds)  # [num_layers+1, llm_dim]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=["xgqa", "worldcuisines_task1", "worldcuisines_task2"],
                         help="cvqa is excluded: its row ids aren't shared across languages, so there's no "
                              "parallel content to align/retrieve.")
    parser.add_argument("--languages", required=True, help="Comma-separated ISO codes, e.g. bn,de,ru,zh,pt,id,ko.")
    parser.add_argument("--eval-data-dir", required=True,
                         help="Dir containing '{lang}.jsonl' per language, as written by "
                              "load_evaluation_data.py (e.g. .../data/xgqa or .../data/worldcuisines/task1).")
    parser.add_argument("--images-dir", default=None, help="Local GQA images dir (required for --pooling llm-layers with xgqa).")
    parser.add_argument("--image-cache-dir", default=None,
                         help="Local image cache dir (required for --pooling llm-layers with worldcuisines).")
    parser.add_argument("--mapping-ckpt", required=True, help="Stage1/Stage2/Stage3 mapping checkpoint to probe.")
    parser.add_argument("--llm-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--mt-path", default="facebook/nllb-200-3.3B")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--pooling", choices=["xm", "llm-layers"], default="xm")
    parser.add_argument("--max-examples-per-lang", type=int, default=300,
                         help="Subsample per language (applied before intersecting ids) -- t-SNE/retrieval "
                              "don't need the full eval set, and llm-layers mode is one GPU forward pass per example.")
    parser.add_argument("--visual-pixels", type=int, default=256 * 256)
    parser.add_argument("--max-mt-seq-len", type=int, default=256)
    parser.add_argument("--max-llm-seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.pooling == "llm-layers":
        if args.benchmark == "xgqa" and not args.images_dir:
            parser.error("--images-dir is required for --pooling llm-layers with --benchmark xgqa")
        if args.benchmark.startswith("worldcuisines") and not args.image_cache_dir:
            parser.error("--image-cache-dir is required for --pooling llm-layers with worldcuisines")

    logger = setup_logging(f"extract_embeddings_{args.benchmark}")
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

    model, processor, tokenizer_mt, tokenizer_llm, device, amp_dtype = load_model(
        args.llm_path, args.mt_path, args.mapping_ckpt, args.local_files_only, max_gen_len=1, logger=logger,
    )

    out_ids, out_langs, out_embeds = [], [], []
    for lang in languages:
        nllb_code = _nllb_code(lang)
        for id_ in shared_ids:
            row = rows_by_lang[lang][id_]
            if args.pooling == "xm":
                emb = _embed_xm(row["query"], nllb_code, model, tokenizer_mt, device, amp_dtype, args.max_mt_seq_len)
            else:
                if args.benchmark == "xgqa":
                    image = load_image_by_id(args.images_dir, row["vg_image_id"])
                else:
                    image = load_image_by_url(row["image_url"], args.image_cache_dir)
                if image is None:
                    continue
                emb = _embed_llm_layers(
                    image, row["query"], args.benchmark, nllb_code,
                    model, processor, tokenizer_mt, device, amp_dtype,
                    args.visual_pixels, args.max_mt_seq_len, args.max_llm_seq_len,
                )
            out_ids.append(id_)
            out_langs.append(lang)
            out_embeds.append(emb)
        logger.info("lang=%s: embedded %d/%d shared examples", lang, sum(1 for l in out_langs if l == lang), len(shared_ids))

    embeddings = torch.stack(out_embeds)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "ids": out_ids, "languages": out_langs, "embeddings": embeddings,
        "pooling": args.pooling, "mapping_ckpt": args.mapping_ckpt, "benchmark": args.benchmark,
    }, args.out)
    logger.info("Saved %s: embeddings shape=%s", args.out, tuple(embeddings.shape))


if __name__ == "__main__":
    main()
