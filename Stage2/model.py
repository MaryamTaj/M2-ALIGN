from __future__ import annotations

import torch
from torch import nn
from transformers import M2M100Model, NllbTokenizer, Qwen3VLForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessorList


def _squeeze_pad(
    hidden_states: torch.Tensor,
    masks: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Remove padding columns that are zero for every example in the batch.

    Args:
        hidden_states: Float tensor of shape ``[batch, seq, dim]``.
        masks: Long tensor of shape ``[batch, seq]`` (1 = real, 0 = pad).
        position_ids: Optional M-RoPE position ids ``[3, batch, seq]``
            (see :meth:`VisualMindMerger._compute_position_ids`), reindexed
            identically to *hidden_states*/*masks* when given.

    Returns:
        ``(hidden_states, masks, keep_idx, position_ids)`` with padding
        columns removed. ``keep_idx`` is a boolean mask over the original
        sequence positions. ``position_ids`` is ``None`` unless supplied.
    """
    x_01 = (masks != 0).long()
    seq_len = x_01.size(1)
    offset = (
        torch.arange(1, seq_len + 1, dtype=torch.long, device=x_01.device)
        .unsqueeze(0)
        .expand_as(x_01)
    )
    x_01 = x_01 * offset
    _, idx = x_01.sort(1, descending=False)
    masks = masks.gather(1, idx)
    idx_ex = idx.unsqueeze(-1).expand_as(hidden_states)
    hidden_states = hidden_states.gather(1, idx_ex)
    bs, _, dim = hidden_states.size()
    masks_sum = masks.sum(dim=0)
    keep_idx = (masks_sum > 0).unsqueeze(0).expand_as(masks)
    masks = masks[keep_idx].view(bs, -1)
    hidden_states = hidden_states[keep_idx.unsqueeze(-1).expand_as(hidden_states)].view(bs, -1, dim)

    if position_ids is not None:
        idx_pos = idx.unsqueeze(0).expand_as(position_ids)
        position_ids = position_ids.gather(2, idx_pos)
        keep_idx_pos = keep_idx.unsqueeze(0).expand_as(position_ids)
        position_ids = position_ids[keep_idx_pos].view(3, bs, -1)

    return hidden_states, masks, keep_idx, position_ids


class PresencePenaltyGeneratedOnly:
    """Logits processor that penalises tokens seen only in the generated suffix.

    Subtracts *penalty* from the logit of any token that already appeared
    after position ``prompt_len`` in ``input_ids``.  Tokens in the original
    prompt are left untouched.  Mirrors Stage 1's processor so both stages
    use consistent decoding.

    Args:
        penalty: Magnitude of the penalty to subtract from repeated logits.
        prompt_len: Number of tokens in the conditioning prefix; positions
            at or before this index are considered part of the prompt.
    """

    def __init__(self, penalty: float, prompt_len: int) -> None:
        self.penalty = float(penalty)
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply the presence penalty to *scores* and return updated logits.

        Args:
            input_ids: Token ids of shape ``[batch, seq]``.
            scores: Logit tensor of shape ``[batch, vocab_size]``.

        Returns:
            Updated *scores* with the penalty applied to repeated tokens.
        """
        if self.penalty == 0.0:
            return scores
        if input_ids.size(1) <= self.prompt_len:
            return scores
        gen_part = input_ids[:, self.prompt_len:]
        for b in range(input_ids.size(0)):
            seen = torch.unique(gen_part[b])
            scores[b, seen] -= self.penalty
        return scores


class MLP(nn.Module):
    """Two-layer MLP: Linear(mt_dim, mt_dim*2) → ReLU → Linear(mt_dim*2, llm_dim).

    Args:
        mt_dim: Input dimension (MT encoder hidden size).
        llm_dim: Output dimension (LLM embedding size).
    """

    def __init__(self, mt_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(mt_dim, mt_dim * 2)
        self.linear2 = nn.Linear(mt_dim * 2, llm_dim)
        self.relu = nn.ReLU()

    def forward(self, mt_hidden_state: torch.Tensor) -> torch.Tensor:
        """Project *mt_hidden_state* into LLM embedding space.

        Args:
            mt_hidden_state: Float tensor of shape ``[batch, seq, mt_dim]``.

        Returns:
            Float tensor of shape ``[batch, seq, llm_dim]``.
        """
        return self.linear2(self.relu(self.linear1(mt_hidden_state)))


class Mapping(nn.Module):
    """Trainable mapping: MLP projection + learnable end-boundary token.

    Args:
        mt_dim: MT encoder hidden-state dimension.
        llm_dim: LLM embedding dimension.
    """

    def __init__(self, mt_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.mlp = MLP(mt_dim, llm_dim)
        self.end_boundary = nn.Parameter(torch.zeros(1, 1, llm_dim), requires_grad=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the MLP projection to *hidden_states*.

        Args:
            hidden_states: MT encoder output of shape ``[batch, seq, mt_dim]``.

        Returns:
            Projected tensor of shape ``[batch, seq, llm_dim]``.
        """
        return self.mlp(hidden_states)

    def get_embed(self) -> torch.Tensor:
        """Return the learnable end-boundary embedding.

        Returns:
            Parameter tensor of shape ``[1, 1, llm_dim]``.
        """
        return self.end_boundary



# ---------------------------------------------------------------------------
# Stage 2 visual grounding model
#
# AugmentedMindMerger (text-task augmentation, Stage 3a) moved to
# Stage3/model.py — only the vision-grounding mapping model lives here now.
# ---------------------------------------------------------------------------

class VisualMindMerger(nn.Module):
    """Stage 2 model: image + multilingual NLLB caption → Qwen3-VL decoder.

    The LLM input prefix is built as:
        ``[BOS] + visual_tokens + X_m + [end_boundary]``

    where:
        ``visual_tokens`` = ``model_llm.visual(pixel_values, image_grid_thw)`` (frozen)
        ``X_m``           = ``mapping(encoder_mt(caption_in_source_language))``   (trainable)

    Mirrors Stage 1's base (non-augmented) structure: no text prompt T is
    appended after the boundary token.  Only the :class:`Mapping` parameters
    are trained.  NLLB encoder and Qwen3-VL (including vision tower) are
    fully frozen.

    Args:
        mt_path: HF id or local path for the NLLB model.
        llm_path: HF id or local path for Qwen3-VL.
        max_gen_len: Maximum new tokens at inference.
        llm_bos_token_id: LLM BOS token id; falls back to pad when ``None``.
        llm_pad_token_id: LLM PAD token id.
        local_files_only: Skip Hub downloads when ``True``.
    """

    def __init__(
        self,
        mt_path: str,
        llm_path: str,
        max_gen_len: int,
        llm_bos_token_id: int | None,
        llm_pad_token_id: int | None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.max_gen_len = max_gen_len

        # NLLB encoder (frozen)
        self.model_mt = M2M100Model.from_pretrained(mt_path, local_files_only=local_files_only)
        self.encoder_mt = self.model_mt.get_encoder()
        for p in self.model_mt.parameters():
            p.requires_grad = False

        # Qwen3-VL: LLM + vision tower (both frozen)
        self.model_llm = Qwen3VLForConditionalGeneration.from_pretrained(
            llm_path, local_files_only=local_files_only
        )
        for p in self.model_llm.parameters():
            p.requires_grad = False
        self.llm_embedding_layer = self.model_llm.get_input_embeddings()

        # Spatial merge size: Qwen3-VL merges adjacent visual patches by this factor.
        # After the visual encoder, each image produces (t*h*w) // merge_size^2 tokens.
        vis_cfg = getattr(getattr(self.model_llm, "config", None), "vision_config", None)
        self._spatial_merge_size: int = getattr(vis_cfg, "spatial_merge_size", 2)

        # Trainable mapping only
        mt_dim = self.model_mt.config.d_model
        llm_dim = getattr(self.llm_embedding_layer, "embedding_dim",
                          self.llm_embedding_layer.weight.shape[1])
        self.mapping = Mapping(mt_dim, llm_dim)

        self.llm_pad_token_id = llm_pad_token_id
        self.llm_bos_token_id = llm_bos_token_id if llm_bos_token_id is not None else llm_pad_token_id
        if self.llm_bos_token_id is None:
            raise ValueError("Need at least one of llm_bos_token_id or llm_pad_token_id.")

    @property
    def llm_dtype(self) -> torch.dtype:
        """Dtype of the frozen LLM weights (typically bf16)."""
        return self.llm_embedding_layer.weight.dtype

    def _embed_visual(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the vision encoder and pad the batch to a uniform token count.

        Args:
            pixel_values: Preprocessed image patches ``[total_patches, C, pH, pW]``.
            image_grid_thw: Patch-grid dimensions ``[B, 3]`` (temporal, h, w).

        Returns:
            ``(vis_padded, vis_mask)`` of shapes ``[B, max_vis, llm_dim]``
            and ``[B, max_vis]`` (long, 1 = real token).
        """
        ms = self._spatial_merge_size
        n_tokens = [
            int(g[0] * g[1] * g[2]) // (ms * ms)
            for g in image_grid_thw
        ]

        raw = self.model_llm.model.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
        # raw: [sum(n_tokens), llm_dim]

        B = image_grid_thw.size(0)
        max_vis = max(n_tokens)
        llm_dim = raw.size(-1)
        device = raw.device

        vis_padded = torch.zeros(B, max_vis, llm_dim, dtype=raw.dtype, device=device)
        vis_mask = torch.zeros(B, max_vis, dtype=torch.long, device=device)
        for i, (chunk, n) in enumerate(zip(raw.split(n_tokens, dim=0), n_tokens)):
            vis_padded[i, :n] = chunk
            vis_mask[i, :n] = 1

        return vis_padded, vis_mask

    def _compute_position_ids(
        self,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """M-RoPE position ids for a mixed vision/text prefix.

        Calling ``model_llm`` with only ``inputs_embeds``/``attention_mask``
        (no ``input_ids``) makes ``Qwen3VLModel`` fall back to a flat arange
        tiled across all 3 RoPE axes, so image tokens lose their 2D spatial
        layout (see Stage3/check_mrope_positions.py). ``get_rope_index``
        only needs ``input_ids`` for its shape/attention-mask bookkeeping,
        not its content, so a dummy tensor is enough.

        Args:
            mm_token_type_ids: ``[B, seq]`` (0=text, 1=image), matching the
                layout ``get_rope_index`` expects.
            image_grid_thw: ``[B, 3]`` patch grid dimensions.
            attention_mask: ``[B, seq]``, 1 = real token.

        Returns:
            Long tensor of shape ``[3, B, seq]``.
        """
        dummy_ids = torch.zeros_like(mm_token_type_ids)
        position_ids, _ = self.model_llm.model.get_rope_index(
            input_ids=dummy_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        return position_ids

    def _build_prefix_raw(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids_mt: torch.Tensor,
        attention_mask_mt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble ``[BOS] + vis + X_m + [end_boundary]`` without squeeze_pad.

        squeeze_pad is applied separately in :meth:`forward` (over the full
        sequence including labels) and in :meth:`generate` (over the prefix).

        Returns:
            ``(llm_embeds, llm_mask, mm_token_type_ids)`` of shapes
            ``[B, prefix_len, llm_dim]``, ``[B, prefix_len]`` and
            ``[B, prefix_len]`` (0=text, 1=image; see
            :meth:`_compute_position_ids`).
        """
        B = input_ids_mt.size(0)
        device = input_ids_mt.device
        dtype = self.llm_dtype

        vis_padded, vis_mask = self._embed_visual(pixel_values, image_grid_thw)
        vis_padded = vis_padded.to(dtype)
        n_vis = vis_padded.size(1)

        mt_out = self.encoder_mt(
            input_ids=input_ids_mt,
            attention_mask=attention_mask_mt,
            output_hidden_states=False,
        )
        x_m = self.mapping(mt_out[0]).to(dtype)

        bos = torch.full((B,), self.llm_bos_token_id, dtype=torch.long, device=device)
        bos_embed = self.llm_embedding_layer(bos).view(B, 1, -1).to(dtype)
        end_boundary = self.mapping.get_embed().expand(B, 1, -1).to(dtype)

        ones1 = torch.ones(B, 1, dtype=torch.long, device=device)
        zeros1 = torch.zeros(B, 1, dtype=torch.long, device=device)
        llm_embeds = torch.cat([bos_embed, vis_padded, x_m, end_boundary], dim=1)
        llm_mask = torch.cat([ones1, vis_mask, attention_mask_mt, ones1], dim=1)
        mm_token_type_ids = torch.cat(
            [
                zeros1,
                torch.ones(B, n_vis, dtype=torch.long, device=device),
                torch.zeros_like(attention_mask_mt),
                zeros1,
            ],
            dim=1,
        )
        return llm_embeds, llm_mask, mm_token_type_ids

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids_mt: torch.Tensor,
        attention_mask_mt: torch.Tensor,
        labels: torch.Tensor,
        mask_label: torch.Tensor,
    ) -> torch.Tensor:
        """Compute teacher-forcing cross-entropy loss for image-captioning.

        Args:
            pixel_values: ``[total_patches, C, pH, pW]`` from the processor.
            image_grid_thw: ``[B, 3]`` patch grid dimensions.
            input_ids_mt: NLLB token ids ``[B, mt_seq]``.
            attention_mask_mt: NLLB attention mask ``[B, mt_seq]``.
            labels: English caption token ids ``[B, label_seq]``.
            mask_label: Caption attention mask ``[B, label_seq]``.

        Returns:
            Scalar cross-entropy loss.
        """
        B = input_ids_mt.size(0)
        dtype = self.llm_dtype

        llm_embeds, llm_mask, mm_token_type_ids = self._build_prefix_raw(
            pixel_values, image_grid_thw,
            input_ids_mt, attention_mask_mt,
        )

        pad_labels = torch.full_like(llm_mask, -100)
        label_embedding = self.llm_embedding_layer(labels).to(dtype)
        llm_embeds = torch.cat([llm_embeds, label_embedding], dim=1)
        llm_mask = torch.cat([llm_mask, mask_label], dim=1)
        mm_token_type_ids = torch.cat([mm_token_type_ids, torch.zeros_like(mask_label)], dim=1)
        labels_masked = labels * mask_label + (-100) * (1 - mask_label)
        labels_full = torch.cat([pad_labels, labels_masked], dim=1)

        position_ids = self._compute_position_ids(mm_token_type_ids, image_grid_thw, llm_mask)
        llm_embeds, llm_mask, cut_idx, position_ids = _squeeze_pad(llm_embeds, llm_mask, position_ids)
        labels_full = labels_full[cut_idx].view(B, -1)

        out = self.model_llm(
            inputs_embeds=llm_embeds,
            attention_mask=llm_mask,
            position_ids=position_ids,
            labels=labels_full,
        )
        return out.loss

    @torch.inference_mode()
    def generate(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids_mt: torch.Tensor,
        attention_mask_mt: torch.Tensor,
        tokenizer_llm,
        generation_kwargs: dict | None = None,
    ) -> list[str]:
        """Generate English captions for a batch of image-caption pairs.

        Args:
            pixel_values: ``[total_patches, C, pH, pW]``.
            image_grid_thw: ``[B, 3]``.
            input_ids_mt: NLLB token ids ``[B, mt_seq]``.
            attention_mask_mt: NLLB attention mask ``[B, mt_seq]``.
            tokenizer_llm: LLM tokenizer for decoding.
            generation_kwargs: Extra kwargs forwarded to ``model_llm.generate``.

        Returns:
            List of decoded caption strings, one per batch example.
        """
        llm_embeds, llm_mask, mm_token_type_ids = self._build_prefix_raw(
            pixel_values, image_grid_thw,
            input_ids_mt, attention_mask_mt,
        )
        position_ids = self._compute_position_ids(mm_token_type_ids, image_grid_thw, llm_mask)
        llm_embeds, llm_mask, _, position_ids = _squeeze_pad(llm_embeds, llm_mask, position_ids)
        prefix_len = llm_embeds.size(1)

        gen_kw: dict = dict(
            inputs_embeds=llm_embeds,
            attention_mask=llm_mask,
            position_ids=position_ids,
            max_new_tokens=self.max_gen_len,
            pad_token_id=self.llm_pad_token_id,
            do_sample=False,
        )
        if generation_kwargs:
            gen_kw.update(generation_kwargs)

        ids = self.model_llm.generate(**gen_kw)
        new_ids = ids[:, prefix_len:] if ids.shape[1] > prefix_len else ids
        return tokenizer_llm.batch_decode(new_ids, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
