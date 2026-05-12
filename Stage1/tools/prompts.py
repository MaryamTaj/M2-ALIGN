def construct_prompt_math(query: str) -> str:
    """Build an instruction-following prompt for math reasoning tasks.

    Wraps *query* in the Alpaca-style instruction template used to
    prompt the LLM during training and evaluation.

    Args:
        query: The raw math problem text in the source language.

    Returns:
        A formatted prompt string ending with "Let's think step by step."

    Examples:
        >>> p = construct_prompt_math("What is 2 + 2?")
        >>> p.startswith("Below is an instruction")
        True
        >>> "Let's think step by step" in p
        True
    """
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{query}\n\n### Response: Let's think step by step."
    )


def construct_prompt_xnli(sample: dict) -> str:
    """Build a natural-language inference prompt from an XNLI sample.

    Args:
        sample: A dict with keys ``"sentence1"`` (premise) and
            ``"sentence2"`` (hypothesis).

    Returns:
        A string of the form ``"Premise: …\\nHypothesis: …\\nLabel:"``.
    """
    sentence1 = sample["sentence1"]
    sentence2 = sample["sentence2"]
    return f"Premise: {sentence1}\nHypothesis: {sentence2}\nLabel:"


def construct_prompt_x_csqa(sample: dict) -> str:
    """Build a commonsense QA prompt from an X-CSQA sample.

    Args:
        sample: A dict with a ``"question"`` sub-dict containing
            ``"stem"`` (the question text) and ``"choices"``
            (a list of dicts with ``"text"`` keys).

    Returns:
        A string of the form ``"<stem>\\nOptions: A. … B. …\\nAnswer:"``.
    """
    question = sample["question"]["stem"]
    choices = sample["question"]["choices"]
    choice_text = [
        chr(ord("A") + i) + ". " + choice["text"]
        for i, choice in enumerate(choices)
    ]
    return question + "\nOptions:" + "\t".join(choice_text) + "\nAnswer:"


def construct_prompt_mt(query: str, src_lang_name: str, trg_lang_name: str) -> str:
    """Build a translation prompt for machine-translation training.

    Args:
        query: The source-language text to be translated.
        src_lang_name: Human-readable name of the source language
            (e.g. ``"French"``).
        trg_lang_name: Human-readable name of the target language
            (e.g. ``"English"``).

    Returns:
        An Alpaca-style prompt that instructs the model to translate
        *query* from *src_lang_name* to *trg_lang_name*.
    """
    instruction = f"Translate the following sentences from {src_lang_name} to {trg_lang_name}."
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n"
        f"### Instruction:\n{instruction}\n"
        f"### Input:\n{query}\n### Response:"
    )
