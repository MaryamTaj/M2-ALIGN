import json
import os
import random

from torch.utils.data import Dataset

from .prompts import (
    construct_prompt_math,
    construct_prompt_mt,
    construct_prompt_x_csqa,
    construct_prompt_xnli,
)

langs_map = {
    "English": "en", "Swahili": "sw", "Chinese": "zh", "Bengali": "bn",
    "German": "de", "Spanish": "es", "French": "fr", "Japanese": "ja",
    "Russian": "ru", "Thai": "th", "Telugu": "te", "Greek": "el",
    "Arabic": "ar", "Bulgarian": "bg", "Croatian": "hr", "Hungarian": "hu",
    "Italian": "it", "Lithuanian": "lt", "Macedonian": "mk", "Polish": "pl",
    "Portuguese": "pt", "Albanian": "sq", "Serbian": "sr", "Turkish": "tr",
    "Vietnamese": "vi", "Hindi": "hi", "Flemish": "nl", "Urdu": "ur",
}


class MathDataset(Dataset):
    """PyTorch Dataset that attaches task-specific prompts to raw samples.

    Each sample returned by :meth:`__getitem__` is a copy of the original
    dict augmented with a ``"prompt"`` key (and, for classification tasks,
    a ``"source"`` key derived from the prompt).

    Args:
        dataset: List of raw sample dicts.  Each dict must contain the
            keys expected by the prompt-construction function for *task*
            (see :mod:`tools.prompts`).
        task: Task identifier string.  Recognised prefixes/values:
            ``"translation"``, ``"csqa"``, ``"nli"``; anything else
            triggers the math-reasoning prompt.
    """

    def __init__(self, dataset: list[dict], task: str) -> None:
        super().__init__()
        self.dataset = dataset
        self.task = task

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """Return sample *idx* with its prompt field populated.

        Args:
            idx: Integer index into the dataset.

        Returns:
            A copy of the raw sample dict with ``"prompt"`` (and
            optionally ``"source"``) filled in.
        """
        sample = self.dataset[idx]
        if self.task == "translation":
            sample["prompt"] = construct_prompt_mt(
                sample["source"], sample["source_language"], sample["target_language"]
            )
        elif "csqa" in self.task:
            sample["source"] = sample["prompt"] = construct_prompt_x_csqa(sample)
        elif "nli" in self.task:
            sample["source"] = sample["prompt"] = construct_prompt_xnli(sample)
        else:
            sample["prompt"] = construct_prompt_math(sample["source"])
        return sample


# ---------------------------------------------------------------------------
# Training-set loaders
# ---------------------------------------------------------------------------

def read_lego(train_num: int, languages: list[str]) -> list[dict]:
    """Load bilingual translation pairs from the LEGO dataset files.

    Reads parallel corpora stored in ``./datas/bilingual_pairs/en-{code}/``
    and returns a shuffled list of sample dicts.

    Args:
        train_num: Maximum number of sentence pairs to load per language.
        languages: Human-readable language names (keys in :data:`langs_map`).

    Returns:
        A shuffled list of dicts with keys ``"source"``, ``"target"``,
        ``"source_language"``, and ``"target_language"``.
    """
    dataset_train = []
    for language in languages:
        lang_code = langs_map[language]
        path_base = f"./datas/bilingual_pairs/en-{lang_code}"
        sources = read_dataset(f"{path_base}/train_100k.{lang_code}")[:train_num]
        targets = read_dataset(f"{path_base}/train_100k.en")[:train_num]
        for source, target in zip(sources, targets):
            dataset_train.append({
                "source": source,
                "target": target,
                "source_language": language,
                "target_language": "English",
            })
    random.shuffle(dataset_train)
    return dataset_train


def read_nllb(data_dir: str, train_num: int, languages: list[str]) -> list[dict]:
    """Load NLLB translation pairs from pre-downloaded JSONL files.

    Each JSONL file is expected to contain one JSON object per line with
    ``"source"`` and ``"target"`` keys, as produced by
    ``load_nllb_corpus.py``.

    Args:
        data_dir: Directory containing ``{language}_to_English.jsonl`` files.
        train_num: Maximum number of pairs to load per language.
        languages: Human-readable language names matching the filenames.

    Returns:
        A shuffled list of dicts with keys ``"source"``, ``"target"``,
        ``"source_language"``, and ``"target_language"``.

    Raises:
        FileNotFoundError: If a required JSONL file is missing in *data_dir*.
    """
    dataset_train = []
    for language in languages:
        path = os.path.join(data_dir, f"{language}_to_English.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"NLLB file not found: {path}")
        for pair in read_dataset(path)[:train_num]:
            dataset_train.append({
                "source": pair["source"],
                "target": pair["target"],
                "source_language": language,
                "target_language": "English",
            })
    random.shuffle(dataset_train)
    return dataset_train


def read_x_csqa_train() -> list[dict]:
    """Load X-CSQA training splits for all supported languages.

    Returns:
        A shuffled list of sample dicts with keys ``"source_language"``,
        ``"target_language"``, ``"target"`` (the answer key), plus the
        original X-CSQA fields.
    """
    language_names = [
        "Urdu", "Swahili", "Hindi", "Arabic", "Vietnamese", "Japanese",
        "Polish", "Chinese", "Flemish", "Russian", "Italian", "German",
        "Portuguese", "French", "Spanish", "English",
    ]
    dataset_train = []
    for name in language_names:
        code = langs_map[name]
        for sample in read_dataset(f"./datas/query_translation/x-csqa/train_{code}.jsonl"):
            sample["source_language"] = name
            sample["target_language"] = "English"
            sample["target"] = sample["answerKey"]
            dataset_train.append(sample)
    random.shuffle(dataset_train)
    return dataset_train


def read_xnli_train() -> list[dict]:
    """Load the XNLI development set for training.

    Returns:
        A shuffled list of sample dicts with keys ``"target"``,
        ``"source_language"``, and ``"target_language"`` appended to
        the original XNLI fields.
    """
    dataset = read_dataset("./datas/query_translation/xnli_dev.json")
    train_set = []
    for sample in dataset:
        sample["target"] = sample["label"]
        sample["source_language"] = sample["language"]
        sample["target_language"] = "English"
        train_set.append(sample)
    random.shuffle(train_set)
    return train_set


def read_math_train(train_num: int) -> list[dict]:
    """Load multilingual math training examples, capping per language.

    Args:
        train_num: Maximum number of examples to retain per language.

    Returns:
        A shuffled list of sample dicts with keys ``"source"``,
        ``"target"``, ``"source_language"``, and ``"target_language"``.
    """
    raw = read_dataset("./datas/query_translation/math.json")
    per_lang: dict[str, list[dict]] = {}
    for sample in raw:
        lang = sample["lang"]
        entry = {
            "source": sample["query"],
            "target": sample["response"],
            "source_language": lang,
            "target_language": "English",
        }
        bucket = per_lang.setdefault(lang, [])
        if len(bucket) < train_num:
            bucket.append(entry)

    dataset_train = [s for bucket in per_lang.values() for s in bucket]
    random.shuffle(dataset_train)
    return dataset_train


# ---------------------------------------------------------------------------
# Evaluation-set loaders
# ---------------------------------------------------------------------------

def read_msvamp() -> dict[str, list[dict]]:
    """Load the MSVAMP test sets for all supported languages.

    Returns:
        A dict mapping language name → list of sample dicts with keys
        ``"source"``, ``"target"``, ``"source_language"``, and
        ``"target_language"``.
    """
    test_list = [
        "Bengali", "Thai", "Swahili", "Japanese", "Chinese",
        "German", "French", "Russian", "Spanish", "English",
    ]
    datasets_test = {}
    for name in test_list:
        raw = read_dataset(f"./datas/evaluation/msvamp/test_{name}.jsonl")
        datasets_test[name] = [
            {
                "source": s["m_query"],
                "target": s["response"],
                "source_language": name,
                "target_language": name,
            }
            for s in raw
        ]
    return datasets_test


def read_mgsms() -> dict[str, list[dict]]:
    """Load MGSM test sets for all supported languages.

    Returns:
        A dict mapping language name → list of sample dicts with keys
        ``"source"``, ``"target"``, ``"source_language"``, and
        ``"target_language"``.
    """
    test_list = [
        "Telugu", "Bengali", "Thai", "Swahili", "Japanese", "Chinese",
        "German", "French", "Russian", "Spanish", "English",
    ]
    datasets_test = {}
    for name in test_list:
        code = langs_map[name]
        raw = read_mgsm(f"./datas/evaluation/mgsm/mgsm_{code}.tsv", name)
        datasets_test[name] = [
            {
                "source": s["question"],
                "target": s["answer"],
                "source_language": name,
                "target_language": name,
            }
            for s in raw
        ]
    return datasets_test


def read_mgsm(path: str, language: str) -> list[dict]:
    """Parse a single MGSM TSV file into a list of sample dicts.

    Each line must be tab-separated with the question in column 0 and
    the answer in column 1.  Commas are stripped from numeric answers.

    Args:
        path: Path to the ``.tsv`` file.
        language: Human-readable language name for metadata fields.

    Returns:
        A list of dicts with keys ``"id"``, ``"question"``,
        ``"explanation"``, ``"answer"``, ``"language"``,
        ``"source_language"``, and ``"target_language"``.
    """
    lines = read_docs(path)
    dataset = []
    for i, line in enumerate(lines):
        parts = line.split("\t")
        dataset.append({
            "id": i,
            "question": parts[0],
            "explanation": "",
            "answer": parts[1].replace(",", "").strip(),
            "language": language,
            "source_language": language,
            "target_language": language,
        })
    return dataset


def read_x_csqa() -> dict[str, list[dict]]:
    """Load X-CSQA evaluation splits for all supported languages.

    Returns:
        A dict mapping language name → list of sample dicts with keys
        ``"source_language"``, ``"target_language"``, ``"target"``,
        plus the original X-CSQA fields.
    """
    language_names = [
        "Urdu", "Swahili", "Hindi", "Arabic", "Vietnamese", "Japanese",
        "Polish", "Chinese", "Flemish", "Russian", "Italian", "German",
        "Portuguese", "French", "Spanish", "English",
    ]
    dataset_tests = {}
    for name in language_names:
        code = langs_map[name]
        raw = read_dataset(f"./datas/evaluation/x-csqa/{code}/dev.jsonl")
        samples = []
        for sample in raw:
            sample["source_language"] = name
            sample["target_language"] = "English"
            sample["target"] = sample["answerKey"]
            samples.append(sample)
        dataset_tests[name] = samples
    return dataset_tests


def read_xnli() -> dict[str, list[dict]]:
    """Load the XNLI test set, grouped by language.

    Returns:
        A dict mapping language name → list of sample dicts with keys
        ``"target"``, ``"source_language"``, and ``"target_language"``
        appended to the original XNLI fields.
    """
    raw = read_dataset("./datas/evaluation/xnli/test.json")
    test_sets: dict[str, list[dict]] = {}
    for sample in raw:
        sample["target"] = sample["label"]
        lang = sample["language"]
        sample["source_language"] = lang
        sample["target_language"] = "English"
        test_sets.setdefault(lang, []).append(sample)
    return test_sets


# ---------------------------------------------------------------------------
# Low-level file readers
# ---------------------------------------------------------------------------

def read_docs(path: str) -> list[str]:
    """Read a plain-text file and return stripped lines.

    Args:
        path: Path to the text file.

    Returns:
        A list of non-empty stripped lines.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def read_dataset(path: str):
    """Load a dataset from a JSONL, JSON, or plain-text file.

    Args:
        path: Path to the file.  Format is inferred from the extension:
            ``.jsonl`` → one JSON object per line;
            ``.json`` → a JSON array or ``{"data": [...]}`` object;
            everything else → list of raw text lines.

    Returns:
        A list of parsed records (dicts for JSON variants, strings for
        plain text).
    """
    if "jsonl" in path:
        dataset = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line))
        return dataset
    if "json" in path:
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        if isinstance(dataset, dict) and "data" in dataset:
            dataset = dataset["data"]
        return dataset
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()
