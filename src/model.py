"""The full model: a frozen text encoder, a trainable mapping, and a frozen backbone.

The encoder reads the source-language text. The mapping turns the encoder's
output into embeddings the backbone can read. The backbone then answers in
English. Only the mapping trains.

Every stage feeds the backbone the same kind of sequence:

    [BOS] + image block + mapped source + [end] + chat

- The image block is only used in Stage 2. There the backbone sees an
  image and the mapped caption, and learns to write the English caption.
- The chat is only used in Stage 3, in math replay and in evaluation. It
  is the backbone's normal chat prompt, with the image inside it.
- Stage 1 and translation replay use neither. The backbone sees only the
  mapped source and learns to write the English translation.
"""
from __future__ import annotations

import torch
from torch import nn

from src.backbones import LLMInputs, load_backbone
from src.encoders import load_encoder


class MLP(nn.Module):
    """Two linear layers with a ReLU between them. The hidden layer is twice the encoder's width."""

    def __init__(self, encoder_dim: int, backbone_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(encoder_dim, encoder_dim * 2)
        self.linear2 = nn.Linear(encoder_dim * 2, backbone_dim)
        self.relu = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(hidden_states)))


class Mapping(nn.Module):
    """The only trained part: the MLP, plus a learned embedding that marks the end of the mapped source.
    """

    def __init__(self, encoder_dim: int, backbone_dim: int) -> None:
        super().__init__()
        self.mlp = MLP(encoder_dim, backbone_dim)
        self.end_boundary = nn.Parameter(torch.zeros(1, 1, backbone_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


def move_to(batch: dict[str, dict[str, torch.Tensor]], device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    """Move every tensor in a batch to device. The batch holds one dict of tensors per part."""
    return {part: {key: t.to(device) for key, t in tensors.items()} for part, tensors in batch.items()}


class MappedVLM(nn.Module):
    """The encoder, mapping and backbone put together.

    A batch is a dict with up to four parts. Each part is a dict of tensors.
    - "source": the source text, from encoder.tokenize. Always present.
    - "image": pixel_values and image_grid_thw. Stage 2 only.
    - "chat": the chat prompt, from backbone.batch_chats. Stage 3, math
      replay and evaluation. A row with no chat can be all padding.
    - "target": the English answer the backbone should write, as
      input_ids and attention_mask. Training only.
    """

    def __init__(self, backbone_name: str, encoder_name: str) -> None:
        super().__init__()
        self.encoder = load_encoder(encoder_name)
        self.backbone = load_backbone(backbone_name)
        self.mapping = Mapping(self.encoder.dim, self.backbone.dim)

    def build_inputs(self, batch: dict) -> LLMInputs:
        """Build [BOS] + image block + mapped source + [end] + chat. Parts missing from the batch are skipped."""
        batch = move_to(batch, self.backbone.device)
        source = batch["source"]
        n_rows = source["input_ids"].size(0)
        one_per_row = torch.ones(n_rows, 1, dtype=torch.long, device=self.backbone.device)

        # The mapping trains in fp32. The encoder may run in bf16, so convert first.
        mapped = self.mapping(self.encoder.encode(source).float())

        parts = [self.backbone.embed_tokens(one_per_row * self.backbone.bos_id)]
        if "image" in batch:
            parts.append(self.backbone.embed_vision(batch["image"]["pixel_values"], batch["image"]["image_grid_thw"]))
        parts.append(self.backbone.text_part(mapped, source["attention_mask"]))
        parts.append(self.backbone.text_part(self.mapping.end_boundary.expand(n_rows, 1, -1), one_per_row))
        if "chat" in batch:
            parts.append(self.backbone.merge_vision(batch["chat"]))
        return LLMInputs.cat(parts)

    def forward(self, batch: dict) -> torch.Tensor:
        """The training loss: how well the backbone predicts batch["target"]."""
        batch = move_to(batch, self.backbone.device)
        return self.backbone.loss(self.build_inputs(batch), batch["target"])

    @torch.inference_mode()
    def generate(self, batch: dict, max_new_tokens: int, **kwargs) -> list[str]:
        """Greedy decoding. Returns one answer string per row."""
        return self.backbone.generate(self.build_inputs(batch), max_new_tokens, **kwargs)
