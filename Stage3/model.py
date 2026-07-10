from __future__ import annotations

import torch
from torch import nn
from transformers import M2M100Model, NllbTokenizer, Qwen3VLForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessorList


def _squeeze_pad(
    hidden_states: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove padding columns that are zero for every example in the batch.

    Args:
        hidden_states: Float tensor of shape ``[batch, seq, dim]``.
        masks: Long tensor of shape ``[batch, seq]`` (1 = real, 0 = pad).

    Returns:
        ``(hidden_states, masks, keep_idx)`` with padding columns removed.
        ``keep_idx`` is a boolean mask over the original sequence positions.
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
    return hidden_states, masks, keep_idx


class PresencePenaltyGeneratedOnly:
    """Logits processor that penalises tokens seen only in the generated suffix.

    Subtracts *penalty* from the logit of any token that already appeared
    after position ``prompt_len`` in ``input_ids``.  Tokens in the original
    prompt are left untouched.  Mirrors Stage 1/2's processor so all stages
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


class AugmentedMindMerger(nn.Module):
    """Stage 3a augmentation model with an LLM-side query prefix (text reasoning).

    Extends the Stage-1-style prefix by appending the LLM token embedding of
    the query text, giving the model direct access to the original query in
    addition to the mapped MT encoder output:

        ``LLM inputs_embeds = [BOS] + X_m + [end_boundary] + T``

    where:
        - ``X_m = mapping(encoder_mt(query_tokens_mt))``
        - ``T   = llm_embedding(query_tokens_llm)``

    Only the :class:`Mapping` parameters are trainable; both the MT encoder
    and the LLM are frozen. Initialised from Stage 2's vision-mapping
    checkpoint (Stage 1 → Stage 2 → Stage 3a). Stage 3a is optional/standalone
    now -- it is no longer chained into Stage 3b, which instead initialises
    directly from Stage 2 (see :class:`AugmentedVisualMindMerger`).

    Args:
        mt_path: HF model ID or local path for the NLLB/M2M MT model.
        llm_path: HF model ID or local path for Qwen3-VL.
        max_gen_len: Maximum number of new tokens to generate at inference.
        llm_bos_token_id: LLM BOS token id.  Falls back to
            *llm_pad_token_id* when ``None`` (Qwen3-VL has no native BOS).
        llm_pad_token_id: LLM PAD token id.
        local_files_only: If ``True``, do not download weights from the Hub.
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

        self.model_mt = M2M100Model.from_pretrained(mt_path, local_files_only=local_files_only)
        self.encoder_mt = self.model_mt.get_encoder()
        for p in self.model_mt.parameters():
            p.requires_grad = False

        self.model_llm = Qwen3VLForConditionalGeneration.from_pretrained(
            llm_path, local_files_only=local_files_only
        )
        for p in self.model_llm.parameters():
            p.requires_grad = False
        self.llm_embedding_layer = self.model_llm.get_input_embeddings()

        mt_dim = self.model_mt.config.d_model
        llm_dim = getattr(self.llm_embedding_layer, "embedding_dim", self.llm_embedding_layer.weight.shape[1])
        self.mapping = Mapping(mt_dim, llm_dim)

        self.llm_pad_token_id = llm_pad_token_id
        self.llm_bos_token_id = llm_bos_token_id if llm_bos_token_id is not None else llm_pad_token_id
        if self.llm_bos_token_id is None:
            raise ValueError("Need at least one of llm_bos_token_id or llm_pad_token_id.")

    @staticmethod
    def squeeze_pad(
        hidden_states: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Delegates to the module-level :func:`_squeeze_pad`."""
        return _squeeze_pad(hidden_states, masks)

    @property
    def llm_dtype(self) -> torch.dtype:
        """Return the dtype of the LLM embedding weights.

        The trainable Mapping stays in fp32 for stable AdamW updates while
        the frozen LLM lives in its checkpoint dtype (typically bf16).
        Casting the Mapping output to this dtype before concatenation
        prevents the forward pass from promoting the full sequence to fp32.
        """
        return self.llm_embedding_layer.weight.dtype

    def forward(
        self,
        input_ids_mt: torch.Tensor,
        attention_mask_mt: torch.Tensor,
        input_ids_query_llm: torch.Tensor,
        mask_query_llm: torch.Tensor,
        labels: torch.Tensor,
        mask_label: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the cross-entropy training loss.

        Builds the prefix ``[BOS] + X_m + [end_boundary] + T``, appends
        the label embeddings, and runs the LLM forward pass.

        Args:
            input_ids_mt: MT token ids of shape ``[batch, mt_seq]``.
            attention_mask_mt: MT attention mask of shape ``[batch, mt_seq]``.
            input_ids_query_llm: LLM token ids for the query,
                shape ``[batch, query_seq]``.
            mask_query_llm: Attention mask for *input_ids_query_llm*.
            labels: LLM token ids for the answer, shape ``[batch, label_seq]``.
            mask_label: Attention mask for *labels*.

        Returns:
            Scalar cross-entropy loss tensor.
        """
        device = input_ids_mt.device
        bs = input_ids_mt.size(0)
        llm_dtype = self.llm_dtype

        end_boundary = self.mapping.get_embed().expand(bs, 1, -1)
        bos = torch.full((bs,), self.llm_bos_token_id, dtype=torch.long, device=device)
        bos_embedding = self.llm_embedding_layer(bos).view(bs, 1, -1)
        ones = torch.ones([bs, 1], dtype=torch.long, device=device)

        mt_out = self.encoder_mt(
            input_ids=input_ids_mt,
            attention_mask=attention_mask_mt,
            output_hidden_states=False,
        )
        x_m = self.mapping(mt_out[0]).to(llm_dtype)
        t_embed = self.llm_embedding_layer(input_ids_query_llm).to(llm_dtype)
        end_boundary = end_boundary.to(llm_dtype)
        bos_embedding = bos_embedding.to(llm_dtype)

        llm_embeds = torch.cat([bos_embedding, x_m, end_boundary, t_embed], dim=1)
        llm_mask = torch.cat([ones, attention_mask_mt, ones, mask_query_llm], dim=1)

        pad_labels = torch.full_like(llm_mask, -100)
        label_embedding = self.llm_embedding_layer(labels).to(llm_dtype)
        llm_embeds = torch.cat([llm_embeds, label_embedding], dim=1)
        llm_mask = torch.cat([llm_mask, mask_label], dim=1)
        labels = labels * mask_label + (-100) * (1 - mask_label)
        labels = torch.cat([pad_labels, labels], dim=1)

        llm_embeds, llm_mask, cut_pad_idx = self.squeeze_pad(llm_embeds, llm_mask)
        labels = labels[cut_pad_idx].view(bs, -1)

        out = self.model_llm(
            inputs_embeds=llm_embeds,
            attention_mask=llm_mask,
            labels=labels,
        )
        return out.loss

    def _build_inputs_embeds_for_generate(
        self,
        input_ids_mt: torch.Tensor,
        attention_mask_mt: torch.Tensor,
        input_ids_query_llm: torch.Tensor,
        mask_query_llm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the generation prefix ``[BOS] + X_m + [end_boundary] + T``.

        Mirrors the training-time construction so that the prefix fed to
        ``model_llm.generate`` is identical to the one seen during training.

        Args:
            input_ids_mt: MT token ids of shape ``[batch, mt_seq]``.
            attention_mask_mt: MT attention mask of shape ``[batch, mt_seq]``.
            input_ids_query_llm: LLM token ids for the query,
                shape ``[batch, query_seq]``.
            mask_query_llm: Attention mask for *input_ids_query_llm*.

        Returns:
            A 2-tuple ``(inputs_embeds, attention_mask)`` ready for
            ``model_llm.generate``.
        """
        device = input_ids_mt.device
        bs = input_ids_mt.size(0)
        llm_dtype = self.llm_dtype

        end_boundary = self.mapping.get_embed().expand(bs, 1, -1)
        bos = torch.full((bs,), self.llm_bos_token_id, dtype=torch.long, device=device)
        bos_embedding = self.llm_embedding_layer(bos).view(bs, 1, -1)
        ones = torch.ones([bs, 1], dtype=torch.long, device=device)

        mt_out = self.encoder_mt(
            input_ids=input_ids_mt,
            attention_mask=attention_mask_mt,
            output_hidden_states=False,
        )
        x_m = self.mapping(mt_out[0]).to(llm_dtype)
        t_embed = self.llm_embedding_layer(input_ids_query_llm).to(llm_dtype)
        end_boundary = end_boundary.to(llm_dtype)
        bos_embedding = bos_embedding.to(llm_dtype)

        llm_embeds = torch.cat([bos_embedding, x_m, end_boundary, t_embed], dim=1)
        llm_mask = torch.cat([ones, attention_mask_mt, ones, mask_query_llm], dim=1)
        llm_embeds, llm_mask, _ = self.squeeze_pad(llm_embeds, llm_mask)
        return llm_embeds, llm_mask

    @torch.inference_mode()
    def generate(
        self,
        input_ids_mt: torch.Tensor,
        attention_mask_mt: torch.Tensor,
        input_ids_query_llm: torch.Tensor,
        mask_query_llm: torch.Tensor,
        tokenizer_llm,
        generation_kwargs: dict | None = None,
        presence_penalty: float | None = None,
    ) -> list[str]:
        """Decode answers from the augmentation prefix ``BOS + X_m + boundary + T``.

        Args:
            input_ids_mt: MT token ids of shape ``[batch, mt_seq]``.
            attention_mask_mt: MT attention mask of shape ``[batch, mt_seq]``.
            input_ids_query_llm: LLM token ids for the query.
            mask_query_llm: Attention mask for *input_ids_query_llm*.
            tokenizer_llm: LLM tokenizer used for decoding.
            generation_kwargs: Extra keyword arguments forwarded to
                ``model_llm.generate`` (e.g. temperature, top_p).
            presence_penalty: If non-zero, apply
                :class:`PresencePenaltyGeneratedOnly`.

        Returns:
            A list of decoded strings, one per batch row.
        """
        llm_embeds, llm_mask = self._build_inputs_embeds_for_generate(
            input_ids_mt, attention_mask_mt, input_ids_query_llm, mask_query_llm
        )
        prefix_len = llm_embeds.size(1)
        gen_kwargs: dict = dict(
            inputs_embeds=llm_embeds,
            attention_mask=llm_mask,
            max_new_tokens=self.max_gen_len,
            pad_token_id=self.llm_pad_token_id,
            do_sample=False,
        )
        if presence_penalty is not None and presence_penalty != 0.0:
            gen_kwargs["logits_processor"] = LogitsProcessorList(
                [PresencePenaltyGeneratedOnly(presence_penalty, prefix_len)]
            )
        if generation_kwargs is not None:
            gen_kwargs.update(generation_kwargs)

        generate_ids = self.model_llm.generate(**gen_kwargs)
        # Some HF versions return only new tokens when using inputs_embeds.
        new_ids = generate_ids[:, prefix_len:] if generate_ids.shape[1] > prefix_len else generate_ids
        return tokenizer_llm.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


# ---------------------------------------------------------------------------
# Stage 3b augmentation model: image + VQA question -> Qwen3-VL decoder
# ---------------------------------------------------------------------------

class AugmentedVisualMindMerger(nn.Module):
    """Stage 3b model: VQA task-specialization with an LLM-side query prefix.

    Extends Stage 2's ``VisualMindMerger`` (image + mapped caption → English
    caption, no query prompt) exactly the way :class:`AugmentedMindMerger`
    extends the base text mapping model: by appending the LLM token
    embedding of the question, giving the model direct access to the raw
    question in addition to the mapped MT encoder output.

        ``LLM inputs_embeds = [BOS] + vis + X_m + [end_boundary] + T``

    where:
        - ``vis   = model_llm.visual(pixel_values, image_grid_thw)`` (frozen)
        - ``X_m   = mapping(encoder_mt(question_tokens_mt))``          (trainable)
        - ``T     = llm_embedding(question_tokens_llm)``                (frozen embedding table)

    ``T`` is built from the *untranslated* source-language question
    (tokenized directly by the LLM's own tokenizer), matching how Stage 3a's
    `T` term is built from the untranslated source query rather than an
    English translation. Only the :class:`Mapping` parameters are
    trainable; the MT encoder, the vision tower, and the LLM are all
    frozen. Initialised directly from Stage 2's checkpoint (chained lineage:
    Stage 1 → Stage 2 → Stage 3 (VQA)); Stage 3a is no longer chained in.

    Args:
        mt_path: HF model ID or local path for the NLLB model.
        llm_path: HF model ID or local path for Qwen3-VL.
        max_gen_len: Maximum number of new tokens to generate at inference.
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

    def _build_prefix_raw(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids_mt: torch.Tensor,
        attention_mask_mt: torch.Tensor,
        input_ids_query_llm: torch.Tensor,
        mask_query_llm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble ``[BOS] + vis + X_m + [end_boundary] + T`` without squeeze_pad.

        squeeze_pad is applied separately in :meth:`forward` (over the full
        sequence including labels) and in :meth:`generate` (over the prefix).

        Args:
            pixel_values: ``[total_patches, C, pH, pW]``.
            image_grid_thw: ``[B, 3]`` patch grid dimensions.
            input_ids_mt: NLLB token ids for the question, ``[B, mt_seq]``.
            attention_mask_mt: NLLB attention mask, ``[B, mt_seq]``.
            input_ids_query_llm: LLM token ids for the *untranslated*
                question, ``[B, query_seq]``.
            mask_query_llm: Attention mask for *input_ids_query_llm*.

        Returns:
            ``(llm_embeds, llm_mask)`` of shapes ``[B, prefix_len, llm_dim]``
            and ``[B, prefix_len]``.
        """
        B = input_ids_mt.size(0)
        device = input_ids_mt.device
        dtype = self.llm_dtype

        vis_padded, vis_mask = self._embed_visual(pixel_values, image_grid_thw)
        vis_padded = vis_padded.to(dtype)

        mt_out = self.encoder_mt(
            input_ids=input_ids_mt,
            attention_mask=attention_mask_mt,
            output_hidden_states=False,
        )
        x_m = self.mapping(mt_out[0]).to(dtype)
        t_embed = self.llm_embedding_layer(input_ids_query_llm).to(dtype)

        bos = torch.full((B,), self.llm_bos_token_id, dtype=torch.long, device=device)
        bos_embed = self.llm_embedding_layer(bos).view(B, 1, -1).to(dtype)
        end_boundary = self.mapping.get_embed().expand(B, 1, -1).to(dtype)

        ones1 = torch.ones(B, 1, dtype=torch.long, device=device)
        llm_embeds = torch.cat([bos_embed, vis_padded, x_m, end_boundary, t_embed], dim=1)
        llm_mask = torch.cat([ones1, vis_mask, attention_mask_mt, ones1, mask_query_llm], dim=1)
        return llm_embeds, llm_mask

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids_mt: torch.Tensor,
        attention_mask_mt: torch.Tensor,
        input_ids_query_llm: torch.Tensor,
        mask_query_llm: torch.Tensor,
        labels: torch.Tensor,
        mask_label: torch.Tensor,
    ) -> torch.Tensor:
        """Compute teacher-forcing cross-entropy loss for VQA.

        Args:
            pixel_values: ``[total_patches, C, pH, pW]`` from the processor.
            image_grid_thw: ``[B, 3]`` patch grid dimensions.
            input_ids_mt: NLLB token ids for the question, ``[B, mt_seq]``.
            attention_mask_mt: NLLB attention mask, ``[B, mt_seq]``.
            input_ids_query_llm: LLM token ids for the untranslated question,
                ``[B, query_seq]``.
            mask_query_llm: Attention mask for *input_ids_query_llm*.
            labels: Short English answer token ids, ``[B, label_seq]``.
            mask_label: Attention mask for *labels*.

        Returns:
            Scalar cross-entropy loss.
        """
        B = input_ids_mt.size(0)
        dtype = self.llm_dtype

        llm_embeds, llm_mask = self._build_prefix_raw(
            pixel_values, image_grid_thw,
            input_ids_mt, attention_mask_mt,
            input_ids_query_llm, mask_query_llm,
        )

        pad_labels = torch.full_like(llm_mask, -100)
        label_embedding = self.llm_embedding_layer(labels).to(dtype)
        llm_embeds = torch.cat([llm_embeds, label_embedding], dim=1)
        llm_mask = torch.cat([llm_mask, mask_label], dim=1)
        labels_masked = labels * mask_label + (-100) * (1 - mask_label)
        labels_full = torch.cat([pad_labels, labels_masked], dim=1)

        llm_embeds, llm_mask, cut_idx = _squeeze_pad(llm_embeds, llm_mask)
        labels_full = labels_full[cut_idx].view(B, -1)

        out = self.model_llm(
            inputs_embeds=llm_embeds,
            attention_mask=llm_mask,
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
        input_ids_query_llm: torch.Tensor,
        mask_query_llm: torch.Tensor,
        tokenizer_llm,
        generation_kwargs: dict | None = None,
    ) -> list[str]:
        """Generate short answers for a batch of VQA examples.

        Args:
            pixel_values: ``[total_patches, C, pH, pW]``.
            image_grid_thw: ``[B, 3]``.
            input_ids_mt: NLLB token ids for the question, ``[B, mt_seq]``.
            attention_mask_mt: NLLB attention mask, ``[B, mt_seq]``.
            input_ids_query_llm: LLM token ids for the untranslated question.
            mask_query_llm: Attention mask for *input_ids_query_llm*.
            tokenizer_llm: LLM tokenizer for decoding.
            generation_kwargs: Extra kwargs forwarded to ``model_llm.generate``.

        Returns:
            List of decoded answer strings, one per batch example.
        """
        llm_embeds, llm_mask = self._build_prefix_raw(
            pixel_values, image_grid_thw,
            input_ids_mt, attention_mask_mt,
            input_ids_query_llm, mask_query_llm,
        )
        llm_embeds, llm_mask, _ = _squeeze_pad(llm_embeds, llm_mask)
        prefix_len = llm_embeds.size(1)

        gen_kw: dict = dict(
            inputs_embeds=llm_embeds,
            attention_mask=llm_mask,
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
