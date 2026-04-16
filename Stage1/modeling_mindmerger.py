
from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
from transformers import Qwen3VLForConditionalGeneration
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, mt_dim, llm_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(mt_dim, mt_dim * 2)
        self.linear2 = nn.Linear(mt_dim * 2, llm_dim)
        self.relu = nn.ReLU()
    def forward(self, mt_hidden_state):
        output = self.linear1(mt_hidden_state)
        output = self.relu(output)
        output = self.linear2(output)
        return output

class Mapping(nn.Module):
    def __init__(self, mt_dim, llm_dim):
        super(Mapping, self).__init__()
        self.mlp = MLP(mt_dim, llm_dim)
        self.end_boundary = nn.Parameter(
            torch.zeros(1, 1, llm_dim), requires_grad=True
        )
    def forward(self, hidden_states):
        hidden_states = self.mlp(hidden_states)
        return hidden_states

    def get_embed(self):
        return self.end_boundary

class MindMerger(nn.Module):
    def __init__(self, mt_path, llm_path, max_gen_len, llm_bos_token_id,
                 llm_pad_token_id):
        super(MindMerger, self).__init__()
        self.max_gen_len = max_gen_len

        model_mt = AutoModel.from_pretrained(mt_path)
        print('MT model size:', sum(param.numel() for param in model_mt.parameters()) / 1000000)
        self.model_mt = model_mt
        for name, parameter in self.model_mt.named_parameters():
            parameter.requires_grad = False
        if 'bert' in mt_path or 'GPT' in mt_path:
            self.encoder_mt = self.model_mt
        else:
            self.encoder_mt = self.model_mt.get_encoder()
        print('used size:', sum(param.numel() for param in self.encoder_mt.parameters()) / 1000000)
        # Qwen3-VL uses a different Transformers model class than a causal-LM.
        # We keep the existing AutoModelForCausalLM path for all other LLMs.
        model_llm = Qwen3VLForConditionalGeneration.from_pretrained(llm_path)

        self.model_llm = model_llm
        self.llm_embedding_layer = self.model_llm.get_input_embeddings()
        for name, parameter in self.model_llm.named_parameters():
            parameter.requires_grad = False
        if 'bert' in mt_path:
            d_model = model_mt.config.hidden_size
        elif 'GPT' in mt_path:
            d_model = model_mt.config.n_embd
        else:
            d_model = model_mt.config.d_model
        # Be robust to model config differences across LLM families.
        # Qwen3-VL often stores the text hidden size under `text_config`.
        llm_dim = getattr(self.llm_embedding_layer, "embedding_dim", None)
        if llm_dim is None:
            # Fallback: embedding weights have shape [vocab, hidden]
            llm_dim = self.llm_embedding_layer.weight.shape[1]

        self.mapping = Mapping(d_model, llm_dim)
        self.llm_pad_token_id = llm_pad_token_id
        self.llm_bos_token_id = llm_bos_token_id
        print('mapping layer size:', sum(param.numel() for param in self.mapping.parameters()) / 1000000)

    def squeeze_pad(self, hidden_states, masks):
        x_01 = (masks != 0).long()

        seq_num_len = x_01.size(1)
        offset = torch.tensor([(i + 1) for i in range(seq_num_len)], dtype=torch.long).to(x_01.device)
        offset = offset.unsqueeze(dim=0).expand_as(x_01)
        x_01 *= offset
        _, idx = x_01.sort(1, descending=False)

        masks = masks.gather(1, idx)
        idx = idx.unsqueeze(dim=-1).expand_as(hidden_states)
        hidden_states = hidden_states.gather(1, idx)

        bs, seq_len, dim = hidden_states.size()
        masks_sum = torch.sum(masks, dim=0)
        idx = masks_sum > 0
        idx = idx.unsqueeze(dim=0).expand_as(masks)
        masks = masks[idx]
        idx_ex = idx.unsqueeze(dim=-1).expand_as(hidden_states)
        hidden_states = hidden_states[idx_ex]
        hidden_states = hidden_states.view(bs, -1, dim)
        masks = masks.view(bs, -1)

        return hidden_states, masks, idx

    def forward(self, input_ids_mt, attention_mask_mt,
                labels=None, mask_label=None, input_ids_prompt=None, mask_prompt=None):
        end_boundary = self.mapping.get_embed()
        bs = input_ids_mt.size(0)
        end_boundary = end_boundary.expand([bs, 1, end_boundary.size(-1)])

        bos = torch.tensor([self.llm_bos_token_id for i in range(bs)], dtype=torch.long).cuda()
        bos_embedding = self.llm_embedding_layer(bos)
        bos_embedding = bos_embedding.view(bs, 1, -1)
        mask = torch.ones([bs, 1], dtype=torch.long).cuda()
        llm_input_embedding = bos_embedding
        llm_input_mask = mask

        mt_encoder_outputs = self.encoder_mt(input_ids=input_ids_mt,
                                             attention_mask=attention_mask_mt,
                                             output_hidden_states=True)
        encoder_last_hidden_state = mt_encoder_outputs[0]
        mt_hidden_state = self.mapping(encoder_last_hidden_state)
        llm_input_embedding = torch.cat([llm_input_embedding, mt_hidden_state, end_boundary],
                                        dim=1)
        llm_input_mask = torch.cat([llm_input_mask, attention_mask_mt, mask], dim=1)

        if input_ids_prompt is not None:

            hidden_states_prompt = self.llm_embedding_layer(input_ids_prompt)
            llm_input_embedding = torch.cat([llm_input_embedding, hidden_states_prompt], dim=1)
            llm_input_mask = torch.cat([llm_input_mask, mask_prompt], dim=1)
        if labels is not None:
            pad_labels = llm_input_mask * -100 + (1 - llm_input_mask) * -100
            label_embedding = self.llm_embedding_layer(labels)
            llm_input_embedding = torch.cat([llm_input_embedding, label_embedding], dim=1)
            llm_input_mask = torch.cat([llm_input_mask, mask_label], dim=1)
            labels = labels * mask_label - 100 * (1 - mask_label)
            labels = torch.cat([pad_labels, labels], dim=1)

        llm_input_embedding, llm_input_mask, cut_pad_idx \
            = self.squeeze_pad(llm_input_embedding, llm_input_mask)

        if labels is None:
            generate_ids = self.model_llm.generate(inputs_embeds=llm_input_embedding,
                                                   attention_mask=llm_input_mask,
                                                   max_new_tokens=self.max_gen_len,
                                                   pad_token_id=self.llm_pad_token_id,
                                                   do_sample=False)
            return generate_ids
        else:
            bs, seq_len = labels.size()
            labels = labels[cut_pad_idx]
            labels = labels.view(bs, -1)
            output = self.model_llm(inputs_embeds=llm_input_embedding,
                                    attention_mask=llm_input_mask,
                                    labels=labels)
            return output.loss



# from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
# from transformers import Qwen3VLForConditionalGeneration
# from transformers.generation.logits_process import LogitsProcessorList
# import torch
# from torch import nn

# class MLP(nn.Module):
#     def __init__(self, mt_dim, llm_dim):
#         super(MLP, self).__init__()
#         self.linear1 = nn.Linear(mt_dim, mt_dim * 2)
#         self.linear2 = nn.Linear(mt_dim * 2, llm_dim)
#         self.relu = nn.ReLU()
#     def forward(self, mt_hidden_state):
#         output = self.linear1(mt_hidden_state)
#         output = self.relu(output)
#         output = self.linear2(output)
#         return output

# class PresencePenaltyGeneratedOnly:
#     """Subtract penalty from logits of tokens seen in the generated suffix (HF-compatible)."""

#     def __init__(self, penalty: float, prompt_len: int):
#         self.penalty = float(penalty)
#         self.prompt_len = int(prompt_len)

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         if self.penalty == 0.0:
#             return scores
#         if input_ids.size(1) <= self.prompt_len:
#             return scores
#         gen_part = input_ids[:, self.prompt_len:]
#         for b in range(input_ids.size(0)):
#             seen = torch.unique(gen_part[b])
#             scores[b, seen] -= self.penalty
#         return scores


# class Mapping(nn.Module):
#     def __init__(self, mt_dim, llm_dim):
#         super(Mapping, self).__init__()
#         self.mlp = MLP(mt_dim, llm_dim)
#         self.end_boundary = nn.Parameter(
#             torch.zeros(1, 1, llm_dim), requires_grad=True
#         )
#     def forward(self, hidden_states):
#         hidden_states = self.mlp(hidden_states)
#         return hidden_states

#     def get_embed(self):
#         return self.end_boundary

# class MindMerger(nn.Module):
#     def __init__(self, mt_path, llm_path, max_gen_len, llm_bos_token_id,
#                  llm_pad_token_id, local_files_only: bool = False):
#         super(MindMerger, self).__init__()
#         self.max_gen_len = max_gen_len

#         model_mt = AutoModel.from_pretrained(mt_path, local_files_only=local_files_only)
#         print('MT model size:', sum(param.numel() for param in model_mt.parameters()) / 1000000)
#         self.model_mt = model_mt
#         for name, parameter in self.model_mt.named_parameters():
#             parameter.requires_grad = False
#         if 'bert' in mt_path or 'GPT' in mt_path:
#             self.encoder_mt = self.model_mt
#         else:
#             self.encoder_mt = self.model_mt.get_encoder()
#         print('used size:', sum(param.numel() for param in self.encoder_mt.parameters()) / 1000000)
#         # Qwen3-VL uses a different Transformers model class than a causal-LM.
#         # We keep the existing AutoModelForCausalLM path for all other LLMs.
#         model_llm = Qwen3VLForConditionalGeneration.from_pretrained(
#             llm_path, local_files_only=local_files_only
#         )

#         self.model_llm = model_llm
#         self.llm_embedding_layer = self.model_llm.get_input_embeddings()
#         for name, parameter in self.model_llm.named_parameters():
#             parameter.requires_grad = False
#         if 'bert' in mt_path:
#             d_model = model_mt.config.hidden_size
#         elif 'GPT' in mt_path:
#             d_model = model_mt.config.n_embd
#         else:
#             d_model = model_mt.config.d_model
#         # Be robust to model config differences across LLM families.
#         # Qwen3-VL often stores the text hidden size under `text_config`.
#         llm_dim = getattr(self.llm_embedding_layer, "embedding_dim", None)
#         if llm_dim is None:
#             # Fallback: embedding weights have shape [vocab, hidden]
#             llm_dim = self.llm_embedding_layer.weight.shape[1]

#         self.mapping = Mapping(d_model, llm_dim)
#         self.llm_pad_token_id = llm_pad_token_id
#         self.llm_bos_token_id = llm_bos_token_id
#         print('mapping layer size:', sum(param.numel() for param in self.mapping.parameters()) / 1000000)

#     def squeeze_pad(self, hidden_states, masks):
#         x_01 = (masks != 0).long()

#         seq_num_len = x_01.size(1)
#         offset = torch.tensor([(i + 1) for i in range(seq_num_len)], dtype=torch.long).to(x_01.device)
#         offset = offset.unsqueeze(dim=0).expand_as(x_01)
#         x_01 *= offset
#         _, idx = x_01.sort(1, descending=False)

#         masks = masks.gather(1, idx)
#         idx = idx.unsqueeze(dim=-1).expand_as(hidden_states)
#         hidden_states = hidden_states.gather(1, idx)

#         bs, seq_len, dim = hidden_states.size()
#         masks_sum = torch.sum(masks, dim=0)
#         idx = masks_sum > 0
#         idx = idx.unsqueeze(dim=0).expand_as(masks)
#         masks = masks[idx]
#         idx_ex = idx.unsqueeze(dim=-1).expand_as(hidden_states)
#         hidden_states = hidden_states[idx_ex]
#         hidden_states = hidden_states.view(bs, -1, dim)
#         masks = masks.view(bs, -1)

#         return hidden_states, masks, idx

#     def _build_inputs_embeds_for_generate(
#         self, input_ids_mt, attention_mask_mt, input_ids_prompt=None, mask_prompt=None
#     ):
#         device = input_ids_mt.device
#         end_boundary = self.mapping.get_embed()
#         bs = input_ids_mt.size(0)
#         end_boundary = end_boundary.expand([bs, 1, end_boundary.size(-1)])

#         bos = torch.tensor([self.llm_bos_token_id for i in range(bs)], dtype=torch.long, device=device)
#         bos_embedding = self.llm_embedding_layer(bos)
#         bos_embedding = bos_embedding.view(bs, 1, -1)
#         mask = torch.ones([bs, 1], dtype=torch.long, device=device)
#         llm_input_embedding = bos_embedding
#         llm_input_mask = mask

#         mt_encoder_outputs = self.encoder_mt(input_ids=input_ids_mt,
#                                              attention_mask=attention_mask_mt,
#                                              output_hidden_states=True)
#         encoder_last_hidden_state = mt_encoder_outputs[0]
#         mt_hidden_state = self.mapping(encoder_last_hidden_state)
#         llm_input_embedding = torch.cat([llm_input_embedding, mt_hidden_state, end_boundary],
#                                         dim=1)
#         llm_input_mask = torch.cat([llm_input_mask, attention_mask_mt, mask], dim=1)

#         if input_ids_prompt is not None:
#             hidden_states_prompt = self.llm_embedding_layer(input_ids_prompt)
#             llm_input_embedding = torch.cat([llm_input_embedding, hidden_states_prompt], dim=1)
#             llm_input_mask = torch.cat([llm_input_mask, mask_prompt], dim=1)

#         llm_input_embedding, llm_input_mask, _ = self.squeeze_pad(llm_input_embedding, llm_input_mask)
#         return llm_input_embedding, llm_input_mask

#     @torch.inference_mode()
#     def generate_from_mt(
#         self,
#         input_ids_mt,
#         attention_mask_mt,
#         tokenizer_llm,
#         input_ids_prompt=None,
#         mask_prompt=None,
#         generation_kwargs=None,
#         presence_penalty: float | None = None,
#     ):
#         llm_input_embedding, llm_input_mask = self._build_inputs_embeds_for_generate(
#             input_ids_mt, attention_mask_mt, input_ids_prompt=input_ids_prompt, mask_prompt=mask_prompt
#         )
#         prefix_len = llm_input_embedding.size(1)
#         gen_kwargs = dict(
#             inputs_embeds=llm_input_embedding,
#             attention_mask=llm_input_mask,
#             max_new_tokens=self.max_gen_len,
#             pad_token_id=self.llm_pad_token_id,
#             do_sample=False,
#         )
#         if presence_penalty is not None and presence_penalty != 0.0:
#             gen_kwargs["logits_processor"] = LogitsProcessorList(
#                 [PresencePenaltyGeneratedOnly(presence_penalty, prefix_len)]
#             )
#         if generation_kwargs is not None:
#             gen_kwargs.update(generation_kwargs)
#         generate_ids = self.model_llm.generate(**gen_kwargs)
#         # Some HF versions return only newly generated token ids when using inputs_embeds.
#         if generate_ids.shape[1] > prefix_len:
#             new_ids = generate_ids[:, prefix_len:]
#         else:
#             new_ids = generate_ids
#         text = tokenizer_llm.batch_decode(
#             new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )[0].strip()
#         return text

#     def forward(self, input_ids_mt, attention_mask_mt,
#                 labels=None, mask_label=None, input_ids_prompt=None, mask_prompt=None,
#                 generation_kwargs=None):
#         device = input_ids_mt.device
#         end_boundary = self.mapping.get_embed()
#         bs = input_ids_mt.size(0)
#         end_boundary = end_boundary.expand([bs, 1, end_boundary.size(-1)])

#         bos = torch.tensor([self.llm_bos_token_id for i in range(bs)], dtype=torch.long, device=device)
#         bos_embedding = self.llm_embedding_layer(bos)
#         bos_embedding = bos_embedding.view(bs, 1, -1)
#         mask = torch.ones([bs, 1], dtype=torch.long, device=device)
#         llm_input_embedding = bos_embedding
#         llm_input_mask = mask

#         mt_encoder_outputs = self.encoder_mt(input_ids=input_ids_mt,
#                                              attention_mask=attention_mask_mt,
#                                              output_hidden_states=True)
#         encoder_last_hidden_state = mt_encoder_outputs[0]
#         mt_hidden_state = self.mapping(encoder_last_hidden_state)
#         llm_input_embedding = torch.cat([llm_input_embedding, mt_hidden_state, end_boundary],
#                                         dim=1)
#         llm_input_mask = torch.cat([llm_input_mask, attention_mask_mt, mask], dim=1)

#         if input_ids_prompt is not None:

#             hidden_states_prompt = self.llm_embedding_layer(input_ids_prompt)
#             llm_input_embedding = torch.cat([llm_input_embedding, hidden_states_prompt], dim=1)
#             llm_input_mask = torch.cat([llm_input_mask, mask_prompt], dim=1)
#         if labels is not None:
#             pad_labels = llm_input_mask * -100 + (1 - llm_input_mask) * -100
#             label_embedding = self.llm_embedding_layer(labels)
#             llm_input_embedding = torch.cat([llm_input_embedding, label_embedding], dim=1)
#             llm_input_mask = torch.cat([llm_input_mask, mask_label], dim=1)
#             labels = labels * mask_label - 100 * (1 - mask_label)
#             labels = torch.cat([pad_labels, labels], dim=1)

#         llm_input_embedding, llm_input_mask, cut_pad_idx \
#             = self.squeeze_pad(llm_input_embedding, llm_input_mask)

#         if labels is None:
#             gen_kwargs = dict(
#                 inputs_embeds=llm_input_embedding,
#                 attention_mask=llm_input_mask,
#                 max_new_tokens=self.max_gen_len,
#                 pad_token_id=self.llm_pad_token_id,
#                 do_sample=False,
#             )
#             if generation_kwargs is not None:
#                 gen_kwargs.update(generation_kwargs)
#             generate_ids = self.model_llm.generate(**gen_kwargs)
#             return generate_ids
#         else:
#             bs, seq_len = labels.size()
#             labels = labels[cut_pad_idx]
#             labels = labels.view(bs, -1)
#             output = self.model_llm(inputs_embeds=llm_input_embedding,
#                                     attention_mask=llm_input_mask,
#                                     labels=labels)
#             return output.loss