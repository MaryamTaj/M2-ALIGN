import torch


def mt_input_features(
    input_texts: list[str],
    tokenizer_m2m,
    max_seq_len: int,
    source_languages: list[str],
    langs_map: dict[str, str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise a batch of source-language texts with the MT (NLLB/M2M) tokenizer.

    Sets ``tokenizer_m2m.src_lang`` per example so that NLLB inserts the
    correct language-tag token.  Pads all sequences in the batch to the
    length of the longest one and moves the tensors to CUDA.

    Args:
        input_texts: Raw source-language strings, one per example.
        tokenizer_m2m: An instantiated NLLB/M2M tokenizer.
        max_seq_len: Maximum number of tokens; longer sequences are
            right-truncated.
        source_languages: Human-readable language names parallel to
            *input_texts* (e.g. ``["French", "Swahili"]``).
        langs_map: Mapping from human-readable language name to the
            NLLB language tag (e.g. ``{"French": "fra_Latn"}``).

    Returns:
        A 2-tuple ``(input_ids, attention_mask)`` of shape
        ``[batch, seq_len]`` on CUDA.
    """
    input_ids_m2m, attention_mask_m2m = [], []
    for text, lang in zip(input_texts, source_languages):
        tokenizer_m2m.src_lang = langs_map[lang]
        enc = tokenizer_m2m(
            text,
            padding="longest",
            max_length=max_seq_len,
            truncation=True,
        )
        input_ids_m2m.append(enc.input_ids)
        attention_mask_m2m.append(enc.attention_mask)

    max_len = max(len(ids) for ids in input_ids_m2m)
    pad_id = tokenizer_m2m.pad_token_id
    for ids, mask in zip(input_ids_m2m, attention_mask_m2m):
        while len(ids) < max_len:
            ids.append(pad_id)
            mask.append(0)

    return (
        torch.tensor(input_ids_m2m, dtype=torch.long).cuda(),
        torch.tensor(attention_mask_m2m, dtype=torch.long).cuda(),
    )


def bert_t5_input_features(
    input_texts: list[str],
    tokenizer_m2m,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise a batch with a BERT or T5 tokenizer (adds special tokens).

    Unlike :func:`mt_input_features`, this helper does not set a
    ``src_lang`` attribute and always adds special tokens, making it
    suitable for encoder-only or sequence-to-sequence models that follow
    the BERT/T5 convention.

    Args:
        input_texts: Raw strings to tokenize.
        tokenizer_m2m: An instantiated BERT or T5 tokenizer.
        max_seq_len: Maximum sequence length; longer sequences are
            right-truncated.

    Returns:
        A 2-tuple ``(input_ids, attention_mask)`` of shape
        ``[batch, seq_len]`` on CUDA.
    """
    enc = tokenizer_m2m(
        input_texts,
        padding="longest",
        max_length=max_seq_len,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    return enc.input_ids.cuda(), enc.attention_mask.cuda()


def llm_input_features(
    input_texts: list[str],
    tokenizer_llm,
    max_seq_len: int,
    add_bos_token: bool,
    add_eos_token: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise a batch of strings with the LLM tokenizer.

    Controls BOS/EOS insertion via ``tokenizer_llm.add_bos_token`` and
    ``tokenizer_llm.add_eos_token`` before calling the tokenizer.
    Padding is applied to the longest sequence in the batch.

    Args:
        input_texts: Raw strings to tokenize.
        tokenizer_llm: An instantiated LLM tokenizer (e.g. Qwen3-VL).
        max_seq_len: Maximum sequence length; longer sequences are
            right-truncated.
        add_bos_token: Whether to prepend a BOS token to every sequence.
        add_eos_token: Whether to append an EOS token to every sequence.

    Returns:
        A 2-tuple ``(input_ids, attention_mask)`` of shape
        ``[batch, seq_len]`` on CUDA.
    """
    tokenizer_llm.add_bos_token = add_bos_token
    tokenizer_llm.add_eos_token = add_eos_token
    enc = tokenizer_llm(
        input_texts,
        padding="longest",
        max_length=max_seq_len,
        truncation=True,
        return_tensors="pt",
    )
    return enc.input_ids.cuda(), enc.attention_mask.cuda()
