import math

import torch
from tqdm import tqdm

from tools.input_features import llm_input_features, mt_input_features
from tools.utils import extract_last_num


def evaluate_math(
    model,
    test_set,
    tokenizer_llm,
    tokenizer_mt,
    max_seq_len: int,
    max_gen_len: int,
    use_prompt: bool,
    langs_map: dict[str, str],
) -> tuple[float, list[dict]]:
    """Evaluate a MindMerger model on a math generation task.

    Generates answers for every batch in *test_set* and compares the last
    number in the generation to the last number in the ground-truth target.

    Args:
        model: A trained :class:`MindMerger` instance.
        test_set: DataLoader yielding batches with keys ``"source"``,
            ``"prompt"``, ``"target"``, and ``"source_language"``.
        tokenizer_llm: LLM tokenizer used for decoding outputs.
        tokenizer_mt: MT tokenizer (NLLB/M2M) used for encoding inputs.
        max_seq_len: Maximum sequence length for the MT tokenizer.
        max_gen_len: Maximum number of tokens the LLM may generate.
        use_prompt: If ``True``, also encode the prompt with the LLM and
            pass it as a conditioning prefix.
        langs_map: Mapping from human-readable language name to the MT
            language tag.

    Returns:
        A 2-tuple ``(accuracy, results_list)`` where *accuracy* is the
        percentage of examples whose extracted numeric answer matches the
        ground truth, and *results_list* is a list of per-example dicts
        with keys ``"question"``, ``"answer"``, ``"prompt"``,
        ``"prediction"``, and ``"output"``.
    """
    model.eval()
    results_list = []
    hit = 0
    step_trange = tqdm(test_set)
    for batch in step_trange:
        sources = batch["source"]
        prompts = batch["prompt"]
        targets = batch["target"]
        source_languages = batch["source_language"]

        input_ids_m2m, attention_mask_m2m = mt_input_features(
            sources, tokenizer_mt, max_seq_len, source_languages, langs_map
        )
        input_ids_prompt, mask_prompt = None, None
        if use_prompt:
            input_ids_prompt, mask_prompt = llm_input_features(
                prompts, tokenizer_llm, max_gen_len, add_bos_token=False, add_eos_token=False
            )

        generate_ids = model(
            input_ids_m2m, attention_mask_m2m,
            input_ids_prompt=input_ids_prompt, mask_prompt=mask_prompt,
        )
        results = tokenizer_llm.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for result, prompt, source, target in zip(results, prompts, sources, targets):
            answer = extract_last_num(target)
            pred = extract_last_num(result)
            results_list.append({
                "question": source,
                "answer": answer,
                "prompt": prompt,
                "prediction": str(pred),
                "output": result,
            })
            if float(answer) == float(pred):
                hit += 1

        acc = round(hit / len(results_list) * 100, 2)
        step_trange.set_postfix_str(f"Acc:{acc}")

    return round(hit / len(results_list) * 100, 2), results_list


def evaluate_classification(
    model,
    test_set,
    tokenizer_llm,
    tokenizer_mt,
    max_seq_len: int,
    max_gen_len: int,
    use_prompt: bool,
    langs_map: dict[str, str],
) -> tuple[float, list[dict]]:
    """Evaluate a MindMerger model on a classification task (exact-match).

    Generates text for every batch and compares the stripped output to the
    target label via exact string match.

    Args:
        model: A trained :class:`MindMerger` instance.
        test_set: DataLoader yielding batches with keys ``"prompt"``,
            ``"target"``, and ``"source_language"``.
        tokenizer_llm: LLM tokenizer used for decoding outputs.
        tokenizer_mt: MT tokenizer used for encoding inputs.
        max_seq_len: Maximum sequence length for the MT tokenizer.
        max_gen_len: Maximum number of tokens the LLM may generate.
        use_prompt: If ``True``, also encode the prompt with the LLM.
        langs_map: Mapping from human-readable language name to MT tag.

    Returns:
        A 2-tuple ``(accuracy, results_list)`` where *accuracy* is the
        percentage of exact-match hits, and *results_list* is a list of
        per-example dicts with keys ``"prompt"``, ``"prediction"``, and
        ``"answer"``.
    """
    model.eval()
    results_list = []
    hit = 0
    step_trange = tqdm(test_set)
    for batch in step_trange:
        prompts = batch["prompt"]
        targets = batch["target"]
        source_languages = batch["source_language"]

        input_ids_m2m, attention_mask_m2m = mt_input_features(
            prompts, tokenizer_mt, max_seq_len, source_languages, langs_map
        )
        input_ids_prompt, mask_prompt = None, None
        if use_prompt:
            input_ids_prompt, mask_prompt = llm_input_features(
                prompts, tokenizer_llm, max_gen_len, add_bos_token=False, add_eos_token=False
            )

        generate_ids = model(
            input_ids_m2m, attention_mask_m2m,
            input_ids_prompt=input_ids_prompt, mask_prompt=mask_prompt,
        )
        results = tokenizer_llm.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for result, prompt, target in zip(results, prompts, targets):
            result = result.strip()
            results_list.append({"prompt": prompt, "prediction": result, "answer": target})
            if target == result:
                hit += 1

        acc = round(hit / len(results_list) * 100, 2)
        step_trange.set_postfix_str(f"Acc:{acc}")

    return round(hit / len(results_list) * 100, 2), results_list


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
        Perplexity (a float ≥ 1.0).
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
