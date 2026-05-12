from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, M2M100Model, Qwen3VLForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessorList


class MLP(nn.Module):
    """Two-layer MLP that projects MT encoder hidden states into LLM embedding space.

    Architecture: Linear(mt_dim, mt_dim*2) → ReLU → Linear(mt_dim*2, llm_dim).

    Args:
        mt_dim: Hidden-state dimension of the MT encoder (e.g. 1024 for NLLB-600M).
        llm_dim: Embedding dimension of the target LLM (e.g. ~3584 for Qwen3-VL-8B).
    """

    def __init__(self, mt_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(mt_dim, mt_dim * 2)
        self.linear2 = nn.Linear(mt_dim * 2, llm_dim)
        self.relu = nn.ReLU()

    def forward(self, mt_hidden_state: torch.Tensor) -> torch.Tensor:
        """Project *mt_hidden_state* from MT space to LLM embedding space.

        Args:
            mt_hidden_state: Float tensor of shape ``[batch, seq, mt_dim]``.

        Returns:
            Float tensor of shape ``[batch, seq, llm_dim]``.
        """
        return self.linear2(self.relu(self.linear1(mt_hidden_state)))


class Mapping(nn.Module):
    """Trainable mapping layer: MLP projection + learnable end-boundary token.

    The end-boundary is a learnable parameter of shape ``[1, 1, llm_dim]``
    that is expanded and concatenated after the projected MT hidden states
    to signal the end of the MT prefix to the LLM.

    Args:
        mt_dim: Input dimension (MT encoder hidden size).
        llm_dim: Output dimension (LLM embedding size).
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


class PresencePenaltyGeneratedOnly:
    """Logits processor that applies a presence penalty to generated tokens only.

    Subtracts *penalty* from the logits of any token that has already
    appeared in the *generated* portion of ``input_ids`` (i.e. tokens
    after position ``prompt_len``).  Tokens present in the original
    prompt are not penalised.

    Args:
        penalty: Penalty magnitude to subtract from repeated token logits.
        prompt_len: Length of the conditioning prefix in tokens; positions
            at or before this index are considered part of the prompt.
    """

    def __init__(self, penalty: float, prompt_len: int) -> None:
        self.penalty = float(penalty)
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply the presence penalty and return updated logits.

        Args:
            input_ids: Token ids of shape ``[batch, seq]`` including both
                prompt and generated tokens.
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


class MindMerger(nn.Module):
    """Stage 1 model: NLLB encoder → trainable mapping → frozen Qwen3-VL LLM.

    Builds the LLM input as:
        ``[BOS] + mapping(NLLB_encoder(source)) + [end_boundary]``
    optionally followed by an LLM-side prompt embedding when *augmentation*
    is enabled.  Only the :class:`Mapping` parameters are trainable; both
    the MT encoder and the LLM are frozen.

    Args:
        mt_path: Hugging Face model ID or local path for the MT encoder
            (NLLB-200 or M2M-100).
        llm_path: Hugging Face model ID or local path for Qwen3-VL.
        max_gen_len: Maximum number of new tokens to generate at inference.
        llm_bos_token_id: BOS token id for the LLM vocabulary.  Falls back
            to *llm_pad_token_id* if ``None`` (Qwen3-VL has no native BOS).
        llm_pad_token_id: PAD token id for the LLM vocabulary.
        local_files_only: If ``True``, do not attempt to download weights
            from the Hub.
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

        # NLLB checkpoints use the M2M100 architecture; load it directly to
        # avoid AutoConfig dispatch failures when model_type is missing.
        mt_path_lower = (mt_path or "").lower()
        if "nllb" in mt_path_lower or "m2m_100" in mt_path_lower:
            model_mt = M2M100Model.from_pretrained(mt_path, local_files_only=local_files_only)
        else:
            model_mt = AutoModel.from_pretrained(mt_path, local_files_only=local_files_only)

        print("MT model size:", sum(p.numel() for p in model_mt.parameters()) / 1e6, "M")
        self.model_mt = model_mt
        for p in self.model_mt.parameters():
            p.requires_grad = False

        if "bert" in mt_path or "GPT" in mt_path:
            self.encoder_mt = self.model_mt
        else:
            self.encoder_mt = self.model_mt.get_encoder()
        print("Encoder size:", sum(p.numel() for p in self.encoder_mt.parameters()) / 1e6, "M")

        model_llm = Qwen3VLForConditionalGeneration.from_pretrained(
            llm_path, local_files_only=local_files_only
        )
        self.model_llm = model_llm
        self.llm_embedding_layer = self.model_llm.get_input_embeddings()
        for p in self.model_llm.parameters():
            p.requires_grad = False

        if "bert" in mt_path:
            d_model = model_mt.config.hidden_size
        elif "GPT" in mt_path:
            d_model = model_mt.config.n_embd
        else:
            d_model = model_mt.config.d_model

        llm_dim = getattr(self.llm_embedding_layer, "embedding_dim", None)
        if llm_dim is None:
            llm_dim = self.llm_embedding_layer.weight.shape[1]

        self.mapping = Mapping(d_model, llm_dim)
        self.llm_pad_token_id = llm_pad_token_id
        # Qwen3-VL has no native BOS token; fall back to PAD so the prefix can still be built.
        self.llm_bos_token_id = llm_bos_token_id if llm_bos_token_id is not None else llm_pad_token_id
        if self.llm_bos_token_id is None:
            raise ValueError(
                "Both llm_bos_token_id and llm_pad_token_id are None; cannot build BOS embedding."
            )
        print("Mapping layer size:", sum(p.numel() for p in self.mapping.parameters()) / 1e6, "M")

    def squeeze_pad(
        self,
        hidden_states: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove padding positions that are all-zero across the batch.

        Sorts each sequence so that padding tokens (mask == 0) come first,
        then discards columns that are padding for every example.  This
        compresses variable-length sequences into a dense tensor without
        copying data across the batch dimension.

        Args:
            hidden_states: Float tensor of shape ``[batch, seq, dim]``.
            masks: Long tensor of shape ``[batch, seq]`` with 1 for real
                tokens and 0 for padding.

        Returns:
            A 3-tuple ``(hidden_states, masks, keep_idx)`` where
            *hidden_states* and *masks* have padding columns removed, and
            *keep_idx* is a boolean mask of shape ``[batch, seq]`` used to
            invert the operation on label tensors.
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

    def _build_inputs_embeds_for_generate(
        self,
        input_ids_mt: torch.Tensor,
        attention_mask_mt: torch.Tensor,
        input_ids_prompt: torch.Tensor | None = None,
        mask_prompt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct the LLM input prefix for generation (no labels).

        Builds: ``[BOS] + mapping(enc(source)) + [end_boundary]`` and
        optionally appends an LLM-tokenised prompt embedding.

        Args:
            input_ids_mt: MT token ids of shape ``[batch, mt_seq]``.
            attention_mask_mt: MT attention mask of shape ``[batch, mt_seq]``.
            input_ids_prompt: Optional LLM token ids for the prompt,
                shape ``[batch, prompt_seq]``.
            mask_prompt: Attention mask for *input_ids_prompt*,
                shape ``[batch, prompt_seq]``.

        Returns:
            A 2-tuple ``(inputs_embeds, attention_mask)`` ready for
            ``model_llm.generate``.
        """
        device = input_ids_mt.device
        bs = input_ids_mt.size(0)
        end_boundary = self.mapping.get_embed().expand(bs, 1, -1)

        bos = torch.full((bs,), self.llm_bos_token_id, dtype=torch.long, device=device)
        bos_embedding = self.llm_embedding_layer(bos).view(bs, 1, -1)
        ones = torch.ones([bs, 1], dtype=torch.long, device=device)

        mt_out = self.encoder_mt(
            input_ids=input_ids_mt,
            attention_mask=attention_mask_mt,
            output_hidden_states=True,
        )
        mt_hidden = self.mapping(mt_out[0])

        llm_embeds = torch.cat([bos_embedding, mt_hidden, end_boundary], dim=1)
        llm_mask = torch.cat([ones, attention_mask_mt, ones], dim=1)

        if input_ids_prompt is not None:
            prompt_embeds = self.llm_embedding_layer(input_ids_prompt)
            llm_embeds = torch.cat([llm_embeds, prompt_embeds], dim=1)
            llm_mask = torch.cat([llm_mask, mask_prompt], dim=1)

        llm_embeds, llm_mask, _ = self.squeeze_pad(llm_embeds, llm_mask)
        return llm_embeds, llm_mask

    @torch.inference_mode()
    def generate_from_mt(
        self,
        input_ids_mt: torch.Tensor,
        attention_mask_mt: torch.Tensor,
        tokenizer_llm,
        input_ids_prompt: torch.Tensor | None = None,
        mask_prompt: torch.Tensor | None = None,
        generation_kwargs: dict | None = None,
        presence_penalty: float | None = None,
    ) -> str:
        """Generate text conditioned on MT-encoded input.

        Builds the LLM embedding prefix, runs ``model_llm.generate``, and
        decodes the newly generated tokens to a string.

        Args:
            input_ids_mt: MT token ids of shape ``[1, mt_seq]``.
            attention_mask_mt: MT attention mask of shape ``[1, mt_seq]``.
            tokenizer_llm: LLM tokenizer used for decoding.
            input_ids_prompt: Optional LLM prompt token ids.
            mask_prompt: Attention mask for *input_ids_prompt*.
            generation_kwargs: Extra keyword arguments forwarded to
                ``model_llm.generate`` (e.g. temperature, top_p).
            presence_penalty: If non-zero, apply a
                :class:`PresencePenaltyGeneratedOnly` logits processor.

        Returns:
            Decoded generation string (stripped of special tokens and
            leading/trailing whitespace).
        """
        llm_embeds, llm_mask = self._build_inputs_embeds_for_generate(
            input_ids_mt, attention_mask_mt,
            input_ids_prompt=input_ids_prompt, mask_prompt=mask_prompt,
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
        )[0].strip()

    def forward(
        self,
        input_ids_mt: torch.Tensor,
        attention_mask_mt: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask_label: torch.Tensor | None = None,
        input_ids_prompt: torch.Tensor | None = None,
        mask_prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: return loss (training) or generated ids (inference).

        During training (*labels* is not ``None``) computes the
        cross-entropy loss over the label tokens.  During inference
        (*labels* is ``None``) runs greedy generation.

        Args:
            input_ids_mt: MT token ids of shape ``[batch, mt_seq]``.
            attention_mask_mt: MT attention mask of shape ``[batch, mt_seq]``.
            labels: LLM token ids for the target sequence,
                shape ``[batch, label_seq]``.  ``None`` → inference mode.
            mask_label: Attention mask for *labels*,
                shape ``[batch, label_seq]``.
            input_ids_prompt: Optional LLM prompt token ids.
            mask_prompt: Attention mask for *input_ids_prompt*.

        Returns:
            Scalar loss tensor when *labels* is provided; otherwise a
            generated-id tensor of shape ``[batch, gen_seq]``.
        """
        device = input_ids_mt.device
        bs = input_ids_mt.size(0)
        end_boundary = self.mapping.get_embed().expand(bs, 1, -1)

        bos = torch.full((bs,), self.llm_bos_token_id, dtype=torch.long, device=device)
        bos_embedding = self.llm_embedding_layer(bos).view(bs, 1, -1)
        ones = torch.ones([bs, 1], dtype=torch.long, device=device)

        mt_out = self.encoder_mt(
            input_ids=input_ids_mt,
            attention_mask=attention_mask_mt,
            output_hidden_states=True,
        )
        mt_hidden = self.mapping(mt_out[0])

        llm_embeds = torch.cat([bos_embedding, mt_hidden, end_boundary], dim=1)
        llm_mask = torch.cat([ones, attention_mask_mt, ones], dim=1)

        if input_ids_prompt is not None:
            prompt_embeds = self.llm_embedding_layer(input_ids_prompt)
            llm_embeds = torch.cat([llm_embeds, prompt_embeds], dim=1)
            llm_mask = torch.cat([llm_mask, mask_prompt], dim=1)

        if labels is not None:
            pad_labels = llm_mask * -100 + (1 - llm_mask) * -100
            label_embedding = self.llm_embedding_layer(labels)
            llm_embeds = torch.cat([llm_embeds, label_embedding], dim=1)
            llm_mask = torch.cat([llm_mask, mask_label], dim=1)
            labels = labels * mask_label - 100 * (1 - mask_label)
            labels = torch.cat([pad_labels, labels], dim=1)

        llm_embeds, llm_mask, cut_pad_idx = self.squeeze_pad(llm_embeds, llm_mask)

        if labels is None:
            return self.model_llm.generate(
                inputs_embeds=llm_embeds,
                attention_mask=llm_mask,
                max_new_tokens=self.max_gen_len,
                pad_token_id=self.llm_pad_token_id,
                do_sample=False,
            )

        bs, _ = labels.size()
        labels = labels[cut_pad_idx].view(bs, -1)
        output = self.model_llm(
            inputs_embeds=llm_embeds,
            attention_mask=llm_mask,
            labels=labels,
        )
        return output.loss
