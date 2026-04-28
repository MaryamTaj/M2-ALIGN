from __future__ import annotations

import torch
from torch import nn
from transformers import M2M100Model, Qwen3VLForConditionalGeneration


class MLP(nn.Module):
    def __init__(self, mt_dim: int, llm_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(mt_dim, mt_dim * 2)
        self.linear2 = nn.Linear(mt_dim * 2, llm_dim)
        self.relu = nn.ReLU()

    def forward(self, mt_hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(mt_hidden_state)))


class Mapping(nn.Module):
    def __init__(self, mt_dim: int, llm_dim: int):
        super().__init__()
        self.mlp = MLP(mt_dim, llm_dim)
        self.end_boundary = nn.Parameter(torch.zeros(1, 1, llm_dim), requires_grad=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)

    def get_embed(self) -> torch.Tensor:
        return self.end_boundary


class AugmentedMindMerger(nn.Module):
    """
    Stage2 augmentation model:
      LLM inputs_embeds = [BOS] + [X_m] + [end_boundary] + [T]
    where:
      X_m = mapping(encoder_mt(query_tokens_mt))
      T   = embedding(query_tokens_llm)
    """

    def __init__(
        self,
        mt_path: str,
        llm_path: str,
        max_gen_len: int,
        llm_bos_token_id: int | None,
        llm_pad_token_id: int | None,
        local_files_only: bool = False,
    ):
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
    def squeeze_pad(hidden_states: torch.Tensor, masks: torch.Tensor):
        x_01 = (masks != 0).long()
        seq_num_len = x_01.size(1)
        offset = torch.arange(1, seq_num_len + 1, dtype=torch.long, device=x_01.device).unsqueeze(0).expand_as(x_01)
        x_01 *= offset
        _, idx = x_01.sort(1, descending=False)

        masks = masks.gather(1, idx)
        idx_ex = idx.unsqueeze(dim=-1).expand_as(hidden_states)
        hidden_states = hidden_states.gather(1, idx_ex)

        bs, _, dim = hidden_states.size()
        masks_sum = torch.sum(masks, dim=0)
        keep_idx = (masks_sum > 0).unsqueeze(dim=0).expand_as(masks)
        masks = masks[keep_idx].view(bs, -1)
        hidden_states = hidden_states[keep_idx.unsqueeze(dim=-1).expand_as(hidden_states)].view(bs, -1, dim)
        return hidden_states, masks, keep_idx

    @property
    def llm_dtype(self) -> torch.dtype:
        # The LLM is loaded in its checkpoint dtype (typically bf16 for Qwen3-VL)
        # while the trainable Mapping stays in fp32 for stable AdamW updates.
        # Use the LLM embedding dtype as the canonical "compute" dtype for the LLM.
        return self.llm_embedding_layer.weight.dtype

    def forward(
        self,
        input_ids_mt: torch.Tensor,
        attention_mask_mt: torch.Tensor,
        input_ids_query_llm: torch.Tensor,
        mask_query_llm: torch.Tensor,
        labels: torch.Tensor,
        mask_label: torch.Tensor,
    ):
        device = input_ids_mt.device
        bs = input_ids_mt.size(0)
        llm_dtype = self.llm_dtype

        end_boundary = self.mapping.get_embed().expand([bs, 1, -1])
        bos = torch.full((bs,), self.llm_bos_token_id, dtype=torch.long, device=device)
        bos_embedding = self.llm_embedding_layer(bos).view(bs, 1, -1)
        ones = torch.ones([bs, 1], dtype=torch.long, device=device)

        mt_encoder_outputs = self.encoder_mt(
            input_ids=input_ids_mt,
            attention_mask=attention_mask_mt,
            output_hidden_states=False,
        )
        x_m = self.mapping(mt_encoder_outputs[0])
        t_embed = self.llm_embedding_layer(input_ids_query_llm)

        # Align dtypes: the trainable mapping (fp32) feeds into a frozen LLM
        # whose weights live in bf16. Cast the prefix pieces produced by the
        # mapping to the LLM dtype so the concat doesn't promote to fp32.
        x_m = x_m.to(llm_dtype)
        end_boundary = end_boundary.to(llm_dtype)
        bos_embedding = bos_embedding.to(llm_dtype)
        t_embed = t_embed.to(llm_dtype)

        llm_input_embedding = torch.cat([bos_embedding, x_m, end_boundary, t_embed], dim=1)
        llm_input_mask = torch.cat([ones, attention_mask_mt, ones, mask_query_llm], dim=1)

        pad_labels = torch.full_like(llm_input_mask, -100)
        label_embedding = self.llm_embedding_layer(labels).to(llm_dtype)
        llm_input_embedding = torch.cat([llm_input_embedding, label_embedding], dim=1)
        llm_input_mask = torch.cat([llm_input_mask, mask_label], dim=1)

        labels = labels * mask_label + (-100) * (1 - mask_label)
        labels = torch.cat([pad_labels, labels], dim=1)

        llm_input_embedding, llm_input_mask, cut_pad_idx = self.squeeze_pad(llm_input_embedding, llm_input_mask)
        labels = labels[cut_pad_idx].view(bs, -1)

        out = self.model_llm(
            inputs_embeds=llm_input_embedding,
            attention_mask=llm_input_mask,
            labels=labels,
        )
        return out.loss
