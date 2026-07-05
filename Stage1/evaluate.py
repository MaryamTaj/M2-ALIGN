# coding=utf-8
"""Perplexity evaluation for Stage 1 MindMerger training."""
from __future__ import annotations

import math

import torch
from tqdm import tqdm

from tools.input_features import llm_input_features, mt_input_features


def evaluate_ppl(
    model,
    test_set,
    tokenizer_llm,
    tokenizer_mt,
    max_seq_len: int,
    max_gen_len: int,
    langs_map: dict[str, str],
    use_prompt: bool,
) -> float:
    """Compute perplexity of a MindMerger model on a held-out set.

    Runs the model in evaluation mode, accumulates the mean cross-entropy
    loss, and returns ``exp(mean_loss)`` as the perplexity.  Restores the
    model to training mode and clears the GPU cache before returning.

    Args:
        model: A trained :class:`MindMerger` instance.
        test_set: DataLoader yielding batches with keys ``"source"``,
            ``"prompt"``, ``"target"``, and ``"source_language"``.
        tokenizer_llm: LLM tokenizer used for label tokenisation.
        tokenizer_mt: MT tokenizer used for source tokenisation.
        max_seq_len: Maximum sequence length for the MT tokenizer.
        max_gen_len: Maximum sequence length for LLM labels.
        langs_map: Mapping from human-readable language name to MT tag.
        use_prompt: If ``True``, also encode the prompt with the LLM.

    Returns:
        Perplexity (a float >= 1.0).
    """
    model.eval()
    loss_all = 0.0
    step_i = 0
    step_trange = tqdm(test_set)
    for batch in step_trange:
        step_i += 1
        sources = batch["source"]
        prompts = batch["prompt"]
        targets = batch["target"]
        source_languages = batch["source_language"]

        input_ids_m2m, attention_mask_m2m = mt_input_features(
            sources, tokenizer_mt, max_seq_len, source_languages, langs_map
        )
        labels, mask_label = llm_input_features(
            targets, tokenizer_llm, max_gen_len, add_bos_token=False, add_eos_token=True
        )

        input_ids_prompt, mask_prompt = None, None
        if use_prompt:
            input_ids_prompt, mask_prompt = llm_input_features(
                prompts, tokenizer_llm, max_gen_len, add_bos_token=False, add_eos_token=False
            )

        loss = model(
            input_ids_m2m, attention_mask_m2m,
            input_ids_prompt=input_ids_prompt, mask_prompt=mask_prompt,
            labels=labels, mask_label=mask_label,
        )
        loss_all += loss.mean().item()
        step_trange.set_postfix_str(f"loss:{round(loss_all / step_i, 4)}")

    perplexity = math.exp(loss_all / step_i)
    model.train()
    torch.cuda.empty_cache()
    return perplexity
