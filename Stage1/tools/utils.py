import torch
import os
import re
import random
import numpy as np
import json


def extract_last_num(text: str) -> float:
    """Extract the last number from a text string.

    Strips thousands-separator commas before scanning so that "1,234"
    is treated as 1234 rather than two separate numbers.

    Args:
        text: The string to search for numeric values.

    Returns:
        The last floating-point number found in *text*, or 0.0 if none
        is present.

    Examples:
        >>> extract_last_num("The answer is 42.")
        42.0
        >>> extract_last_num("1,234 items")
        1234.0
        >>> extract_last_num("no numbers here")
        0.0
    """
    text = re.sub(r"(\d),(\d)", r"\g<1>\g<2>", text)
    res = re.findall(r"(\d+(\.\d+)?)", text)
    if res:
        return float(res[-1][0])
    return 0.0


def save_model(output_dir: str, model: torch.nn.Module, model_name: str = "pytorch_model.bin") -> None:
    """Save a model's state dict to *output_dir*/*model_name*.

    Creates *output_dir* if it does not already exist.

    Args:
        output_dir: Directory in which to write the checkpoint file.
        model: The PyTorch module whose ``state_dict`` will be saved.
        model_name: Filename for the checkpoint (default: "pytorch_model.bin").
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, model_name)
    torch.save(
        {"model_state_dict": model.state_dict()},
        output_path,
        _use_new_zipfile_serialization=False,
    )


def save_dataset(path: str, name: str, dataset) -> None:
    """Write *dataset* to *path*/*name* in either plain-text or JSON format.

    Creates *path* if it does not already exist. The format is inferred
    from the file extension: ``.txt`` files are written one item per line;
    all other extensions are serialised as JSON.

    Args:
        path: Directory in which to place the output file.
        name: Filename (including extension) for the output file.
        dataset: The data to write. For ``.txt`` output each element
            must support ``strip()``; otherwise the value must be
            JSON-serialisable.
    """
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, name)
    if file_path.endswith(".txt"):
        with open(file_path, "w", encoding="utf-8") as f:
            for line in dataset:
                f.write(line.strip() + "\n")
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Sets ``PYTHONHASHSEED``, ``torch`` (CPU and all GPUs), ``numpy``, and
    the built-in ``random`` module.  Also enables ``cudnn`` determinism.

    Args:
        seed: The integer seed value to use across all RNG backends.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_hf_format(model: torch.nn.Module, tokenizer, output_dir: str) -> None:
    """Save a model and tokenizer in Hugging Face format.

    Writes model weights and tokenizer files to *output_dir* in the
    standard ``save_pretrained`` layout so the checkpoint can later be
    loaded with ``from_pretrained``.

    Args:
        model: The Hugging Face model to save.
        tokenizer: The corresponding tokenizer to save.
        output_dir: Destination directory (created if absent).
    """
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
