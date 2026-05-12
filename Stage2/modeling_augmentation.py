from __future__ import annotations

import torch
from torch import nn
from transformers import M2M100Model, Qwen3VLForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessorList


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


class AugmentedMindMerger(nn.Module):
    """Stage 2 augmentation model with an LLM-side query prefix.

    Extends the Stage 1 prefix by appending the LLM token embedding of the
    query text, giving the model direct access to the original query in
    addition to the mapped MT encoder output:

        ``LLM inputs_embeds = [BOS] + X_m + [end_boundary] + T``

    where:
        - ``X_m = mapping(encoder_mt(query_tokens_mt))``
        - ``T   = llm_embedding(query_tokens_llm)``

    Only the :class:`Mapping` parameters are trainable; both the MT encoder
    and the LLM are frozen.

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
        """Remove padding columns that are zero for every example in the batch.

        Args:
            hidden_states: Float tensor of shape ``[batch, seq, dim]``.
            masks: Long tensor of shape ``[batch, seq]`` (1 = real, 0 = pad).

        Returns:
            A 3-tuple ``(hidden_states, masks, keep_idx)`` with padding
            columns removed.  *keep_idx* is a boolean mask that can be
            used to strip the same positions from the label tensor.
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
