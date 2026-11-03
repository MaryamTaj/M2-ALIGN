"""Frozen multilingual text encoders that feed the mapping layer.

An encoder turns a batch of source-language texts into hidden states of shape
``[batch, seq, dim]``. The mapping layer in ``model.py`` projects those into
the backbone's embedding space. Every encoder is frozen.
"""
from __future__ import annotations

import os
import torch
from torch import nn
from transformers import AutoTokenizer, M2M100Model, MT5EncoderModel, NllbTokenizer

HF_HOME = os.path.join(os.environ.get("SCRATCH", "."), "huggingface")

ENCODERS = {
    "nllb-600m": ("nllb", os.path.join(HF_HOME, "nllb-200-distilled-600M-full")),
    "nllb-3.3b": ("nllb", "facebook/nllb-200-3.3B"),
    "mt5-large": ("mt5", "google/mt5-large"),
    "mt5-xl": ("mt5", "google/mt5-xl"),
}

# ISO 639-1 code -> NLLB language tag.
NLLB_TAGS = {
    "en": "eng_Latn",
    "bn": "ben_Beng",
    "de": "deu_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "pt": "por_Latn",
    "id": "ind_Latn",
    "ko": "kor_Hang",
    "jv": "jav_Latn",
    "mn": "khk_Cyrl",  
    "si": "sin_Sinh",
    "ga": "gle_Latn",
    "am": "amh_Ethi",
    "ig": "ibo_Latn",
    "om": "gaz_Latn",  
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ja": "jpn_Jpan",
    "sw": "swh_Latn",
    "th": "tha_Thai",
    "te": "tel_Telu",
}


class TextEncoder(nn.Module):
    """A frozen tokenizer + encoder pair.

    Subclasses set ``self.tokenizer`` and ``self.encoder`` and implement
    ``tokenize``. ``encode`` is shared.
    """

    tokenizer = None
    encoder: nn.Module

    def freeze(self) -> None:
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @property
    def dim(self) -> int:
        return self.encoder.config.d_model

    def tokenize(self, texts: list[str], langs: list[str], max_len: int) -> dict[str, torch.Tensor]:
        """Return ``input_ids`` and ``attention_mask``, both ``[batch, seq]`` on CPU."""
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the encoder on a tokenized batch and return ``[batch, seq, dim]``.

        No gradient is needed here: the encoder is frozen, and only the
        mapping layer after it is trained.
        """
        out = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return out.last_hidden_state


class NLLBEncoder(TextEncoder):
    """The encoder half of NLLB-200.

    NLLB prepends a language tag token, so each text is tokenized with its
    own ``src_lang`` and the batch is right-padded afterwards.
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self.tokenizer = NllbTokenizer.from_pretrained(path)
        self.encoder = M2M100Model.from_pretrained(path).get_encoder()
        self.freeze()

    def tokenize(self, texts: list[str], langs: list[str], max_len: int) -> dict[str, torch.Tensor]:
        ids = []
        for text, lang in zip(texts, langs):
            self.tokenizer.src_lang = NLLB_TAGS[lang]
            ids.append(self.tokenizer(text, truncation=True, max_length=max_len)["input_ids"])

        longest = max(len(x) for x in ids)
        pad_id = self.tokenizer.pad_token_id
        input_ids = [x + [pad_id] * (longest - len(x)) for x in ids]
        attention_mask = [[1] * len(x) + [0] * (longest - len(x)) for x in ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class MT5Encoder(TextEncoder):
    """The encoder half of mT5.

    mT5 has no language tags, so ``langs`` is ignored. Its weights are
    loaded in bf16 because mT5 overflows in fp16.
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.encoder = MT5EncoderModel.from_pretrained(path, dtype=torch.bfloat16)
        self.freeze()

    def tokenize(self, texts: list[str], langs: list[str], max_len: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(texts, truncation=True, max_length=max_len, padding="longest", return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def load_encoder(name: str) -> TextEncoder:
    family, path = ENCODERS[name]
    if family == "nllb":
        return NLLBEncoder(path)
    return MT5Encoder(path)
