"""M-RoPE position-id diagnostic for the M2-ALIGN VQA pipeline.

Compares three ways of computing Qwen3-VL position_ids for the same real
xGQA example (n161313.jpg + its Bengali question from Stage3/data/xgqa/bn.jsonl):

  Path A: what Stage2/model.py + Stage3/model.py currently do -- call the
          LLM with only `inputs_embeds`/`attention_mask`, no `input_ids`,
          no `mm_token_type_ids`. Reproduces Qwen3VLTextModel.forward's
          fallback (modeling_qwen3_vl.py:888-891): a flat arange tiled
          identically across all 3 RoPE axes.

  Path B: the proposed fix -- build a `mm_token_type_ids` tag tensor
          (0=text, 1=image) for the same BOS+vis+X_m+boundary+T layout and
          call `Qwen3VLModel.get_rope_index` directly.

  Path C: reference -- the real `processor.apply_chat_template(...)`
          pipeline that Baseline/evaluate.py already uses, run through the
          same `get_rope_index` call, to confirm Path B's image-block shape
          matches what a known-correct path produces.

No model weights are loaded. `get_rope_index` only reads `self.config`, so
we bind it to a lightweight stand-in instead of instantiating the real
8B-parameter model.

Usage (from the repo root, inside a salloc allocation with the venv active):
    python3 Stage3/check_mrope_positions.py
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("HF_HOME", os.path.join(os.environ.get("SCRATCH", "."), "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

MODEL_ID = os.environ.get("LLM_PATH", "Qwen/Qwen3-VL-8B-Instruct")
SCRATCH = os.environ.get("SCRATCH", ".")
IMAGE_PATH = f"{SCRATCH}/M2-ALIGN/Stage3/data/gqa/images/n161313.jpg"
XGQA_BN = f"{SCRATCH}/M2-ALIGN/Stage3/data/xgqa/bn.jsonl"

_VQA_SYSTEM = "You are a helpful assistant that answers questions about images."


def build_open_ended_prompt(question: str) -> str:
    """Must match Stage3/evaluate.py's build_open_ended_prompt."""
    return f"Question: {question}\nAnswer with a single word or short phrase, in English."


class _RopeOnly:
    """Exposes only what get_rope_index touches (self.config + the helper it
    calls), so we can call it without allocating the real 8B-param model."""

    get_vision_position_ids = Qwen3VLModel.get_vision_position_ids
    get_rope_index = Qwen3VLModel.get_rope_index

    def __init__(self, config):
        self.config = config


def load_example() -> tuple[Image.Image, str]:
    with open(XGQA_BN, "r", encoding="utf-8") as f:
        row = json.loads(f.readline())
    assert row["vg_image_id"] == "n161313", row["vg_image_id"]
    image = Image.open(IMAGE_PATH).convert("RGB")
    return image, row["query"]


def path_a_flat(seq_len: int) -> torch.Tensor:
    """Reproduce the position_ids Qwen3VLTextModel.forward builds when
    position_ids is None (modeling_qwen3_vl.py:888-891): a single flat
    arange, tiled identically across all 3 RoPE axes. Shape [3, 1, seq_len]."""
    flat = torch.arange(seq_len).view(1, -1)
    return flat.unsqueeze(0).expand(3, 1, seq_len).clone()


def describe(pos: torch.Tensor, vis_start: int, vis_len: int, label: str) -> None:
    block = pos[:, 0, vis_start : vis_start + vis_len]
    t_axis, h_axis, w_axis = block[0], block[1], block[2]
    flat = torch.equal(t_axis, h_axis) and torch.equal(h_axis, w_axis)
    print(f"\n--- {label} ---")
    print("  image block positions (t,h,w) for first 6 tokens:")
    for i in range(min(6, vis_len)):
        print(f"    tok {i}: ({t_axis[i].item()}, {h_axis[i].item()}, {w_axis[i].item()})")
    print(f"  flat (all 3 axes identical)?           {flat}")
    print(f"  h-axis unique values: {h_axis.unique().numel()}   w-axis unique values: {w_axis.unique().numel()}")
    if vis_start + vis_len < pos.shape[-1]:
        nxt = tuple(pos[:, 0, vis_start + vis_len].tolist())
        print(f"  position right after image block:       {nxt}")


def main() -> None:
    print("Step 1/5: loading example (image + question) ...", flush=True)
    image, question = load_example()
    print(f"  Loaded {IMAGE_PATH}")
    print(f"  Question (bn): {question!r}")

    print("Step 2/5: loading config + processor (local cache only, no weights) ...", flush=True)
    config = Qwen3VLConfig.from_pretrained(MODEL_ID, local_files_only=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    rope = _RopeOnly(config)
    spatial_merge_size = config.vision_config.spatial_merge_size

    print("Step 3/5: preprocessing image ...", flush=True)
    image_inputs = processor(images=[image], text=[""], return_tensors="pt")
    image_grid_thw = image_inputs["image_grid_thw"]
    t, h, w = image_grid_thw[0].tolist()
    grid_h, grid_w = h // spatial_merge_size, w // spatial_merge_size
    n_vis = int(image_grid_thw.prod(-1).item()) // (spatial_merge_size**2)
    print(
        f"  image_grid_thw={image_grid_thw.tolist()} -> {n_vis} visual tokens "
        f"({grid_h} rows x {grid_w} cols after merge size {spatial_merge_size})"
    )

    print("Step 4/5: building the BOS+vis+X_m+boundary+T skeleton and computing positions ...", flush=True)
    # [BOS] + vis(n_vis) + X_m(n_xm) + [boundary] + T(n_t) -- mirrors what
    # Stage2/3's _build_prefix_raw actually constructs.
    n_xm = 6  # stand-in length for the mapped NLLB-encoder output
    prompt_ids = processor.tokenizer(build_open_ended_prompt(question), return_tensors="pt")["input_ids"]
    n_t = prompt_ids.shape[1]
    seq_len = 1 + n_vis + n_xm + 1 + n_t

    mm_type = torch.zeros(1, seq_len, dtype=torch.long)
    mm_type[:, 1 : 1 + n_vis] = 1  # 0=text, 1=image (get_rope_index docstring)
    attn_mask = torch.ones(1, seq_len, dtype=torch.long)
    dummy_ids = torch.zeros(1, seq_len, dtype=torch.long)  # content unused by get_rope_index

    pos_a = path_a_flat(seq_len)  # Path A: current Stage2/3 behavior

    pos_b, _ = rope.get_rope_index(  # Path B: proposed fix
        input_ids=dummy_ids,
        mm_token_type_ids=mm_type,
        image_grid_thw=image_grid_thw,
        attention_mask=attn_mask,
    )

    print("Step 5/5: cross-checking against the real processor.apply_chat_template pipeline ...", flush=True)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _VQA_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_open_ended_prompt(question)},
            ],
        },
    ]
    ref_inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )
    pos_c, _ = rope.get_rope_index(  # Path C: reference
        input_ids=ref_inputs["input_ids"],
        mm_token_type_ids=ref_inputs["mm_token_type_ids"],
        image_grid_thw=ref_inputs["image_grid_thw"],
        attention_mask=ref_inputs["attention_mask"],
    )
    ref_types = ref_inputs["mm_token_type_ids"][0]
    vis_idx = (ref_types == 1).nonzero(as_tuple=True)[0]
    c_start, c_len = vis_idx[0].item(), vis_idx.numel()

    print("\n================ RESULTS ================")
    describe(pos_a, vis_start=1, vis_len=n_vis, label="Path A (current Stage2/3 code: no input_ids/mm_token_type_ids)")
    describe(pos_b, vis_start=1, vis_len=n_vis, label="Path B (proposed fix: explicit get_rope_index call)")
    describe(pos_c, vis_start=c_start, vis_len=c_len, label="Path C (reference: real processor.apply_chat_template pipeline)")

    print("\n================ VERDICT ================")
    a_block = pos_a[:, 0, 1 : 1 + n_vis]
    b_block = pos_b[:, 0, 1 : 1 + n_vis]
    c_block = pos_c[:, 0, c_start : c_start + c_len]

    a_flat = torch.equal(a_block[0], a_block[1]) and torch.equal(a_block[1], a_block[2])
    b_flat = torch.equal(b_block[0], b_block[1]) and torch.equal(b_block[1], b_block[2])

    b_h_pat, c_h_pat = b_block[1] - b_block[1].min(), c_block[1] - c_block[1].min()
    b_w_pat, c_w_pat = b_block[2] - b_block[2].min(), c_block[2] - c_block[2].min()
    structural_match = (
        b_block.shape == c_block.shape
        and torch.equal(b_h_pat, c_h_pat)
        and torch.equal(b_w_pat, c_w_pat)
    )

    print(f"Path A image block is flat 1D (bug reproduced):     {a_flat}")
    print(f"Path B image block is a real 2D grid (fix works):   {not b_flat}")
    print(f"Path B structurally matches Path C's grid pattern:  {structural_match}")


if __name__ == "__main__":
    main()
