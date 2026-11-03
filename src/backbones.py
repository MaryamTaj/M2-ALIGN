"""Frozen vision-language backbones.

A backbone takes ``LLMInputs`` (embeddings plus the bookkeeping the LLM
needs) and returns a loss, logits or generated text. ``model.py`` builds
those inputs by concatenating the backbone's own pieces (token embeddings,
vision features, a rendered chat) with the mapped encoder output.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, replace

import torch
from PIL import Image
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

HF_HOME = os.path.join(os.environ.get("SCRATCH", "."), "huggingface")
VISUAL_PIXELS = 256 * 256

BACKBONES = {
    "qwen-2b": ("qwen3vl", "Qwen/Qwen3-VL-2B-Instruct"),
    "qwen-4b": ("qwen3vl", "Qwen/Qwen3-VL-4B-Instruct"),
    "qwen-8b": ("qwen3vl", os.path.join(
        HF_HOME, "hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")),
}


@dataclass
class LLMInputs:
    """A batch of LLM input embeddings with the tensors that travel with them.

    ``token_types`` marks image positions (1) and text positions (0).
    ``image_grid`` and ``deepstack`` are only set when the batch has images,
    and ``positions`` is filled in just before the LLM call.
    """

    embeds: torch.Tensor  # [B, S, D]
    mask: torch.Tensor  # [B, S], 1 = real token
    token_types: torch.Tensor  # [B, S]
    image_grid: torch.Tensor | None = None  # [B, 3]
    deepstack: list[torch.Tensor] | None = None  # per layer, [n_image_tokens, D]
    positions: torch.Tensor | None = None  # [3, B, S]

    @staticmethod
    def cat(parts: list[LLMInputs]) -> LLMInputs:
        """Join parts along the sequence. At most one part may carry an image."""
        with_image = [p for p in parts if p.image_grid is not None]
        if len(with_image) > 1:
            raise ValueError("Only one part of the sequence can carry images.")
        image = with_image[0] if with_image else None
        return LLMInputs(
            embeds=torch.cat([p.embeds for p in parts], dim=1),
            mask=torch.cat([p.mask for p in parts], dim=1),
            token_types=torch.cat([p.token_types for p in parts], dim=1),
            image_grid=image.image_grid if image else None,
            deepstack=image.deepstack if image else None,
        )


def squeeze_pad(inputs: LLMInputs, labels: torch.Tensor | None = None) -> tuple[LLMInputs, torch.Tensor | None]:
    """Move each row's padding to the front, then drop columns that are padding in every row.

    Prefix pieces are padded separately, so a row can have padding in the
    middle. Real tokens keep their order, so image tokens still line up with
    ``deepstack``. ``labels`` and ``positions`` are reordered the same way.
    """
    seq_len = inputs.mask.size(1)
    order_key = (inputs.mask != 0).long() * torch.arange(1, seq_len + 1, device=inputs.mask.device)
    idx = order_key.argsort(dim=1)
    keep = inputs.mask.gather(1, idx).sum(dim=0) > 0

    def reorder(t: torch.Tensor) -> torch.Tensor:
        return t.gather(1, idx)[:, keep]

    embeds_idx = idx.unsqueeze(-1).expand_as(inputs.embeds)
    positions = None
    if inputs.positions is not None:
        positions = inputs.positions.gather(2, idx.unsqueeze(0).expand_as(inputs.positions))[:, :, keep]
    squeezed = LLMInputs(
        embeds=inputs.embeds.gather(1, embeds_idx)[:, keep],
        mask=reorder(inputs.mask),
        token_types=reorder(inputs.token_types),
        image_grid=inputs.image_grid,
        deepstack=inputs.deepstack,
        positions=positions,
    )
    return squeezed, (reorder(labels) if labels is not None else None)


@contextlib.contextmanager
def _deepstack_injection(text_model: nn.Module, visual_mask: torch.Tensor, deepstack_embeds: list[torch.Tensor] | None):
    """Pass DeepStack features to the first call of ``text_model.forward`` only.

    Qwen3-VL adds multi-scale vision features into its first decoder layers,
    but only when the outer model runs the vision tower itself. We run the
    tower ourselves and pass ``inputs_embeds``, so the features are injected
    here. Only the first call (the full prompt) gets them. Later generation
    steps read from the KV cache and have no image positions.
    """
    if deepstack_embeds is None:
        yield
        return

    original_forward = text_model.forward
    pending = {"visual_pos_masks": visual_mask, "deepstack_visual_embeds": deepstack_embeds}

    def patched_forward(*args, **kwargs):
        if pending:
            kwargs.update(pending)
            pending.clear()
        return original_forward(*args, **kwargs)

    text_model.forward = patched_forward
    try:
        yield
    finally:
        del text_model.forward


class VLMBackbone(nn.Module):
    """A frozen VLM with its processor.

    Subclasses load ``self.model`` and ``self.processor`` and implement the
    model-specific parts: ``embed_vision``, ``merge_vision`` and ``_call``.
    """

    model: nn.Module
    processor = None

    def freeze(self) -> None:
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def dim(self) -> int:
        return self.model.get_input_embeddings().embedding_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.model.get_input_embeddings().weight.dtype

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    @property
    def bos_id(self) -> int:
        # Qwen3-VL has no BOS token; the mapped prefix starts with PAD instead.
        return self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.pad_token_id

    def embed_tokens(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> LLMInputs:
        mask = torch.ones_like(ids) if mask is None else mask
        embeds = self.model.get_input_embeddings()(ids).to(self.dtype)
        return LLMInputs(embeds, mask, torch.zeros_like(ids))

    def text_part(self, embeds: torch.Tensor, mask: torch.Tensor) -> LLMInputs:
        """Wrap embeddings from outside the backbone (the mapped encoder output) as text."""
        return LLMInputs(embeds.to(self.dtype), mask, torch.zeros_like(mask))

    def render_chat(self, messages: list[dict], image: Image.Image | None = None, max_len: int | None = None) -> dict:
        """Tokenize a chat that ends with the assistant turn opening.

        ``messages`` use plain-string content. If ``image`` is given, it goes
        at the start of the last user turn. Returns CPU tensors with a batch
        dimension of 1.
        """
        truncation = {"truncation": True, "max_length": max_len} if max_len else {}
        if image is None:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                **truncation,
            )

        parts = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        parts[-1]["content"].insert(0, {"type": "image", "image": image})
        return self.processor.apply_chat_template(
            parts, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            processor_kwargs={"min_pixels": VISUAL_PIXELS, "max_pixels": VISUAL_PIXELS, **truncation},
        )

    def batch_chats(self, chats: list[dict]) -> dict:
        """Right-pad rendered chats into one batch. Images are concatenated."""
        n = len(chats)
        longest = max(c["input_ids"].size(1) for c in chats)
        batch = {
            "input_ids": torch.full((n, longest), self.tokenizer.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((n, longest), dtype=torch.long),
            "mm_token_type_ids": torch.zeros((n, longest), dtype=torch.long),
        }
        for i, c in enumerate(chats):
            length = c["input_ids"].size(1)
            for key in batch:
                if key in c:
                    batch[key][i, :length] = c[key][0]
        if "pixel_values" in chats[0]:
            batch["pixel_values"] = torch.cat([c["pixel_values"] for c in chats], dim=0)
            batch["image_grid_thw"] = torch.cat([c["image_grid_thw"] for c in chats], dim=0)
        return batch

    def embed_vision(self, pixel_values: torch.Tensor, image_grid: torch.Tensor) -> LLMInputs:
        """Stage 2: the image as a standalone block of vision tokens."""
        raise NotImplementedError

    def merge_vision(self, chat: dict) -> LLMInputs:
        """Stage 3 and evaluation: a rendered chat with image features placed in its image slots."""
        raise NotImplementedError

    def _call(self, inputs: LLMInputs, **kwargs):
        """Run the LLM on prepared inputs. ``kwargs`` go to ``forward`` or, with ``generate=True``, ``generate``."""
        raise NotImplementedError

    def loss(self, inputs: LLMInputs, labels: dict[str, torch.Tensor]) -> torch.Tensor:
        """Cross-entropy on ``labels`` (``input_ids``, ``attention_mask``) appended after ``inputs``."""
        target = self.embed_tokens(labels["input_ids"], labels["attention_mask"])
        full = LLMInputs.cat([inputs, target])
        label_ids = labels["input_ids"].masked_fill(labels["attention_mask"] == 0, -100)
        label_ids = torch.cat([torch.full_like(inputs.mask, -100), label_ids], dim=1)
        return self._call(full, labels=label_ids).loss

    def logits(self, inputs: LLMInputs) -> torch.Tensor:
        return self._call(inputs).logits

    def generate(self, inputs: LLMInputs, max_new_tokens: int, **kwargs) -> list[str]:
        ids = self._call(
            inputs, generate=True, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, **kwargs,
        )
        # Given only inputs_embeds, generate() returns only the new tokens.
        texts = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [t.strip() for t in texts]


class Qwen3VLBackbone(VLMBackbone):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype="auto")
        self.freeze()

    def _image_features(self, pixel_values: torch.Tensor, image_grid: torch.Tensor):
        out = self.model.model.get_image_features(pixel_values, image_grid, return_dict=True)
        features = torch.cat(out.pooler_output, dim=0).to(self.dtype)
        deepstack = [f.to(self.dtype) for f in out.deepstack_features]
        n_tokens = [len(f) for f in out.pooler_output]
        return features, deepstack, n_tokens

    def embed_vision(self, pixel_values: torch.Tensor, image_grid: torch.Tensor) -> LLMInputs:
        """``<|vision_start|> + image tokens + <|vision_end|>``, right-padded per image."""
        features, deepstack, n_tokens = self._image_features(pixel_values, image_grid)
        batch, longest = len(n_tokens), max(n_tokens)
        embeds = features.new_zeros(batch, longest, features.size(-1))
        mask = torch.zeros(batch, longest, dtype=torch.long, device=features.device)
        for i, chunk in enumerate(features.split(n_tokens)):
            embeds[i, :len(chunk)] = chunk
            mask[i, :len(chunk)] = 1

        config = self.model.config
        start = self.embed_tokens(torch.full((batch, 1), config.vision_start_token_id, device=features.device))
        end = self.embed_tokens(torch.full((batch, 1), config.vision_end_token_id, device=features.device))
        image = LLMInputs(embeds, mask, torch.ones_like(mask), image_grid=image_grid, deepstack=deepstack)
        return LLMInputs.cat([start, image, end])

    def merge_vision(self, chat: dict) -> LLMInputs:
        inputs = self.embed_tokens(chat["input_ids"], chat["attention_mask"])
        if "pixel_values" not in chat:
            return inputs

        features, deepstack, _ = self._image_features(chat["pixel_values"], chat["image_grid_thw"])
        slots, _ = self.model.model.get_placeholder_mask(
            chat["input_ids"], inputs_embeds=inputs.embeds, image_features=features,
        )
        inputs.embeds = inputs.embeds.masked_scatter(slots, features)
        inputs.token_types = chat["mm_token_type_ids"]
        inputs.image_grid = chat["image_grid_thw"]
        inputs.deepstack = deepstack
        return inputs

    def _rope_positions(self, inputs: LLMInputs) -> torch.Tensor:
        """M-RoPE positions, so image tokens keep their 2D layout.

        Without ``input_ids``, Qwen3-VL falls back to flat positions.
        ``get_rope_index`` only reads token types and the mask, so dummy ids
        are enough.
        """
        positions, _ = self.model.model.get_rope_index(
            input_ids=torch.zeros_like(inputs.token_types),
            mm_token_type_ids=inputs.token_types,
            image_grid_thw=inputs.image_grid,
            attention_mask=inputs.mask,
        )
        return positions

    def _call(self, inputs: LLMInputs, labels: torch.Tensor | None = None, generate: bool = False, **kwargs):
        inputs = replace(inputs, positions=self._rope_positions(inputs))
        inputs, labels = squeeze_pad(inputs, labels)
        visual_mask = (inputs.token_types == 1) & (inputs.mask != 0)
        run = self.model.generate if generate else self.model
        if labels is not None:
            kwargs["labels"] = labels
        with _deepstack_injection(self.model.model.language_model, visual_mask, inputs.deepstack):
            return run(inputs_embeds=inputs.embeds, attention_mask=inputs.mask, position_ids=inputs.positions, **kwargs)


def load_backbone(name: str) -> VLMBackbone:
    family, path = BACKBONES[name]
    if family == "qwen3vl":
        return Qwen3VLBackbone(path)
    raise ValueError(f"Unknown backbone family: {family}")
