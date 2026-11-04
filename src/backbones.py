"""The frozen vision-language model (the "backbone") that the mapping feeds into.

The backbone never trains. model.py builds its input as one sequence of
embeddings. The sequence mixes three things: the backbone's own token
embeddings, its image features, and the mapping's output. This file wraps
those pieces as LLMInputs, runs the backbone on them, and returns a loss,
logits or generated text.
"""
from __future__ import annotations

import contextlib
import functools
import os
from dataclasses import dataclass, replace

import torch
from PIL import Image
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

HF_CACHE = os.path.join(os.environ.get("SCRATCH", "."), "huggingface", "hub")
# Every image is resized to exactly this many pixels, so at 32x32 px per merged
# visual token this is the token budget the backbone sees: 256*256 is 64 tokens,
# against roughly 1,400 for the processor's own default on a CVQA photograph.
#
# The budget is not neutral on CVQA. Scoring one checkpoint on identical items at
# 588 merged tokens instead of 64 is worth +3.35 accuracy points averaged over
# bn/ga/jv/mn/si (35.26 -> 38.61), and it changes the sign of the comparison
# against the zero-shot backbone: below its backbone at 64 tokens, above it at
# 588. On xGQA the same sweep is flat (+0.11, p = 0.17), so this is
# benchmark-dependent rather than a general "use more pixels" -- the curve is
# steep below roughly 187 merged tokens and flat above.
#
# Overridable so the budget can be swept without editing the source; the default
# is unchanged, so every number produced before this reproduces exactly.
VISUAL_PIXELS = int(os.environ.get("VISUAL_PIXELS", 256 * 256))

BACKBONES = {
    "qwen-2b": ("qwen3vl", "Qwen/Qwen3-VL-2B-Instruct"),
    "qwen-8b": ("qwen3vl", "Qwen/Qwen3-VL-8B-Instruct"),
    "qwen-32b": ("qwen3vl", "Qwen/Qwen3-VL-32B-Instruct"),
}


@dataclass
class LLMInputs:
    """One batch of input embeddings, plus the extra tensors Qwen3-VL needs.

    Each row is one example, stored as one embedding per position.
    - mask: 1 for a real token, 0 for padding.
    - token_types: 1 for an image token, 0 for a text token. Qwen3-VL uses
      this to give image tokens 2D positions.
    - image_grid: each image's size in patches (time, height, width).
    - deepstack: extra image features taken from earlier vision layers.
      Qwen3-VL adds them into its first few decoder layers.
    - positions: the position of every token. _call fills this in.
    image_grid and deepstack are None when the batch has no image.
    """

    embeds: torch.Tensor  # [B, S, D]
    mask: torch.Tensor  # [B, S]
    token_types: torch.Tensor  # [B, S]
    image_grid: torch.Tensor | None = None  # [B, 3]
    deepstack: list[torch.Tensor] | None = None  # one per layer, [n_image_tokens, D]
    positions: torch.Tensor | None = None  # [3, B, S]

    @staticmethod
    def cat(parts: list[LLMInputs]) -> LLMInputs:
        """Join pieces end to end. Only one piece may hold the image.

        A batch has one image_grid and one deepstack, so they can only
        describe one image block per row.
        """
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
    """Move each row's padding to the left, then drop columns that are all padding.

    model.py pads each piece of the sequence on its own. So padding can sit
    in the middle of a row, like [prefix, pad, pad, chat]. This turns it into
    [pad, pad, prefix, chat]. generate() needs this, because it appends new
    tokens on the right.

    Real tokens keep their order. That matters for images: deepstack lists
    image features in order, and they must still line up with the image
    tokens. labels and positions are moved the same way as the embeddings.
    """
    is_real = inputs.mask != 0
    # Padding gets key 0. Real tokens get their position: 1, 2, 3, ...
    # Sorting by this key puts padding first and keeps real tokens in order.
    sort_key = is_real.long() * torch.arange(1, is_real.size(1) + 1, device=is_real.device)
    new_order = sort_key.argsort(dim=1)
    # Keep a column if at least one row has a real token there.
    keep = is_real.gather(1, new_order).any(dim=0)

    # positions is [3, B, S]. Put the batch first so it moves like the rest.
    positions = _reorder_columns(inputs.positions.permute(1, 2, 0), new_order, keep).permute(2, 0, 1)
    squeezed = replace(
        inputs,
        embeds=_reorder_columns(inputs.embeds, new_order, keep),
        mask=_reorder_columns(inputs.mask, new_order, keep),
        token_types=_reorder_columns(inputs.token_types, new_order, keep),
        positions=positions,
    )
    if labels is not None:
        labels = _reorder_columns(labels, new_order, keep)
    return squeezed, labels


def _reorder_columns(t: torch.Tensor, new_order: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Rearrange the columns of t ([B, S, ...]) into new_order, then keep only the kept columns."""
    extra_dims = (1,) * (t.dim() - 2)
    index = new_order.reshape(*new_order.shape, *extra_dims).expand_as(t)
    return t.gather(1, index)[:, keep]


@contextlib.contextmanager
def _deepstack_injection(text_model: nn.Module, inputs: LLMInputs):
    """Hand the DeepStack features to the decoder on its first call only.

    Qwen3-VL adds DeepStack features into its first decoder layers. But it
    only does this when it runs the vision tower itself, from pixel_values.
    We run the vision tower ourselves and pass ready-made embeddings
    instead, so Qwen3-VL never sees the features. This hook adds them back
    as arguments to the decoder.

    Only the first call gets them. In training, that is the only call. In
    generation, the first call reads the whole prompt, including the image.
    Each later call reads just one new text token, so there is no image to
    add features to.
    """
    if inputs.deepstack is None:
        yield
        return

    pending = {
        "visual_pos_masks": (inputs.token_types == 1) & (inputs.mask != 0),
        "deepstack_visual_embeds": inputs.deepstack,
    }
    hook = text_model.register_forward_pre_hook(functools.partial(_add_deepstack_once, pending), with_kwargs=True)
    try:
        yield
    finally:
        hook.remove()


def _add_deepstack_once(pending: dict, module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Copy the DeepStack arguments into this call, then empty pending so later calls run unchanged."""
    kwargs.update(pending)
    pending.clear()
    return args, kwargs


class VLMBackbone(nn.Module):
    """A frozen vision-language model and its processor.

    This class holds the parts that work the same for any model. A subclass
    loads self.model and self.processor, and fills in the model-specific
    methods: embed_vision, merge_vision and _call.
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
        """The token that starts the mapped prefix. Qwen3-VL has no BOS token, so it uses PAD."""
        return self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.pad_token_id

    def embed_tokens(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> LLMInputs:
        """Look up the backbone's embeddings for token ids. All positions are marked as text."""
        mask = torch.ones_like(ids) if mask is None else mask
        embeds = self.model.get_input_embeddings()(ids).to(self.dtype)
        return LLMInputs(embeds, mask, torch.zeros_like(ids))

    def text_part(self, embeds: torch.Tensor, mask: torch.Tensor) -> LLMInputs:
        """Wrap the mapping's output as LLMInputs. The backbone treats it as text."""
        return LLMInputs(embeds.to(self.dtype), mask, torch.zeros_like(mask))

    def render_chat(self, messages: list[dict], image: Image.Image | None = None, max_len: int | None = None) -> dict:
        """Turn a chat into token ids. The ids end where the assistant's reply would start.

        Each message's content is a plain string. If an image is given, it
        goes before the text of the last message. The processor then adds
        image placeholder tokens and computes pixel_values. merge_vision
        later fills those placeholders with image features.
        Returns CPU tensors for a batch of one.
        """
        truncation = {"truncation": True, "max_length": max_len} if max_len else {}
        if image is None:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                **truncation,
            )

        # The processor wants content as a list of typed parts, so the image can be one of them.
        parts = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        parts[-1]["content"].insert(0, {"type": "image", "image": image})
        return self.processor.apply_chat_template(
            parts, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            processor_kwargs={"min_pixels": VISUAL_PIXELS, "max_pixels": VISUAL_PIXELS, **truncation},
        )

    def batch_chats(self, chats: list[dict]) -> dict:
        """Combine chats from render_chat into one batch.

        Shorter chats get padding on the right. Image tensors are joined
        end to end, because Qwen3-VL keeps all images of a batch in one tensor.
        """
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

    def process_images(self, images: list[Image.Image]) -> dict:
        """Stage 2: turn images into pixel_values and image_grid_thw, outside any chat.

        Every image is resized to VISUAL_PIXELS, so all images in a batch
        get the same number of image tokens.
        """
        out = self.processor(
            images=images, text=[""] * len(images), return_tensors="pt",
            min_pixels=VISUAL_PIXELS, max_pixels=VISUAL_PIXELS,
        )
        return {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"]}

    def embed_vision(self, pixel_values: torch.Tensor, image_grid: torch.Tensor) -> LLMInputs:
        """Stage 2: embed each image as its own block, outside any chat."""
        raise NotImplementedError

    def merge_vision(self, chat: dict) -> LLMInputs:
        """Stage 3 and evaluation: embed a chat and put the image features into its placeholders."""
        raise NotImplementedError

    def _call(self, inputs: LLMInputs, **kwargs):
        """Run the backbone. kwargs go to forward(), or to generate() when generate=True."""
        raise NotImplementedError

    def loss(self, inputs: LLMInputs, labels: dict[str, torch.Tensor]) -> torch.Tensor:
        """Cross-entropy on the target tokens in labels, placed right after inputs.

        Only the target tokens are scored. The inputs and the padding get
        label -100, which the loss skips.
        """
        target = self.embed_tokens(labels["input_ids"], labels["attention_mask"])
        full = LLMInputs.cat([inputs, target])
        label_ids = labels["input_ids"].masked_fill(labels["attention_mask"] == 0, -100)
        label_ids = torch.cat([torch.full_like(inputs.mask, -100), label_ids], dim=1)
        return self._call(full, labels=label_ids).loss

    def logits(self, inputs: LLMInputs) -> torch.Tensor:
        return self._call(inputs).logits

    def generate(self, inputs: LLMInputs, max_new_tokens: int, **kwargs) -> list[str]:
        """Greedy decoding. Returns one answer string per row."""
        ids = self._call(
            inputs, generate=True, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, **kwargs,
        )
        # We pass embeddings, not token ids, so generate() returns only the new tokens.
        # There is no prompt to cut off.
        texts = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [t.strip() for t in texts]


class Qwen3VLBackbone(VLMBackbone):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=HF_CACHE)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, cache_dir=HF_CACHE, dtype="auto")
        self.freeze()

    def _image_features(self, pixel_values: torch.Tensor, image_grid: torch.Tensor):
        """Run the vision tower.

        Returns three things: the image tokens of all images stacked into
        one tensor, the DeepStack features, and how many tokens each image has.
        """
        out = self.model.model.get_image_features(pixel_values, image_grid, return_dict=True)
        features = torch.cat(out.pooler_output, dim=0).to(self.dtype)
        deepstack = [f.to(self.dtype) for f in out.deepstack_features]
        n_tokens = [len(f) for f in out.pooler_output]
        return features, deepstack, n_tokens

    def embed_vision(self, pixel_values: torch.Tensor, image_grid: torch.Tensor) -> LLMInputs:
        """Each row becomes <|vision_start|>, the image tokens, then <|vision_end|>.

        The rows are padded to the same length. model.py puts this block
        next to the mapped caption.
        """
        features, deepstack, n_tokens = self._image_features(pixel_values, image_grid)
        batch, longest = len(n_tokens), max(n_tokens)
        embeds = features.new_zeros(batch, longest, features.size(-1))
        mask = torch.zeros(batch, longest, dtype=torch.long, device=features.device)
        for i, image_tokens in enumerate(features.split(n_tokens)):
            embeds[i, :len(image_tokens)] = image_tokens
            mask[i, :len(image_tokens)] = 1

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
        # placeholders is True wherever the chat holds an image placeholder token.
        placeholders, _ = self.model.model.get_placeholder_mask(
            chat["input_ids"], inputs_embeds=inputs.embeds, image_features=features,
        )
        inputs.embeds = inputs.embeds.masked_scatter(placeholders, features)
        inputs.token_types = chat["mm_token_type_ids"]
        inputs.image_grid = chat["image_grid_thw"]
        inputs.deepstack = deepstack
        return inputs

    def _rope_positions(self, inputs: LLMInputs) -> torch.Tensor:
        """Compute Qwen3-VL's 3D token positions (M-RoPE).

        Text tokens count up in a line. Image tokens get a (time, row,
        column) position, so the model knows where each patch sits in the
        image. Qwen3-VL normally works these out from input_ids. We pass
        embeddings instead, and without this every token would get a plain
        1D position. get_rope_index only reads the token types and the mask,
        so zeros work as input_ids.
        """
        positions, _ = self.model.model.get_rope_index(
            input_ids=torch.zeros_like(inputs.token_types),
            mm_token_type_ids=inputs.token_types,
            image_grid_thw=inputs.image_grid,
            attention_mask=inputs.mask,
        )
        return positions

    def _call(self, inputs: LLMInputs, labels: torch.Tensor | None = None, generate: bool = False, **kwargs):
        # Positions come first. get_rope_index uses the mask to skip padding,
        # so padding in the middle of a row is fine here. squeeze_pad then
        # moves the padding to the left, and the positions move with it.
        inputs = replace(inputs, positions=self._rope_positions(inputs))
        inputs, labels = squeeze_pad(inputs, labels)
        if labels is not None:
            kwargs["labels"] = labels
        run = self.model.generate if generate else self.model
        with _deepstack_injection(self.model.model.language_model, inputs):
            return run(inputs_embeds=inputs.embeds, attention_mask=inputs.mask, position_ids=inputs.positions, **kwargs)


def load_backbone(name: str) -> VLMBackbone:
    family, model_id = BACKBONES[name]
    if family == "qwen3vl":
        return Qwen3VLBackbone(model_id)
    raise ValueError(f"Unknown backbone family: {family}")
