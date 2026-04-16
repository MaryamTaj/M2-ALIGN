# import argparse
# import os

# import torch
# from transformers import AutoTokenizer, AutoConfig


# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Smoke test: can this repo-style training loss path load Qwen3-VL and run with inputs_embeds?"
#     )
#     parser.add_argument("--llm_path", type=str, required=True, help="HF model id or local path to Qwen3-VL.")
#     parser.add_argument("--prompt", type=str, default="Hello, world!", help="Text-only prompt for smoke test.")
#     parser.add_argument("--max_new_tokens", type=int, default=5, help="Unused for loss-only smoke test.")
#     parser.add_argument(
#         "--device_map",
#         type=str,
#         default="auto",
#         help="Passed to from_pretrained(device_map=...). Use 'cpu' if needed.",
#     )
#     parser.add_argument(
#         "--dtype",
#         type=str,
#         default="auto",
#         help="Torch dtype for loading: one of auto, float16, bfloat16, float32.",
#     )
#     args = parser.parse_args()

#     # Avoid permission issues when the runtime can't write to ~/.cache/huggingface.
#     default_cache_dir = os.path.join(os.path.dirname(__file__), ".hf_cache")
#     os.environ.setdefault("HF_HOME", default_cache_dir)
#     os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(default_cache_dir, "transformers"))

#     dtype_map = {
#         "auto": "auto",
#         "float16": torch.float16,
#         "bfloat16": torch.bfloat16,
#         "float32": torch.float32,
#     }
#     torch_dtype = dtype_map.get(args.dtype, "auto")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"[smoke] device={device}, dtype={args.dtype}, device_map={args.device_map}")

#     tokenizer = AutoTokenizer.from_pretrained(args.llm_path, use_fast=True)
#     # Repo code expects pad_token_id to exist; it sets pad_token=eos_token.
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "left"

#     # Use the correct Qwen3-VL model class (the repo currently expects a causal-LM-like interface).
#     llm_cfg = AutoConfig.from_pretrained(args.llm_path)
#     model_type = getattr(llm_cfg, "model_type", None)
#     if model_type != "qwen3_vl":
#         raise ValueError(f"Expected a Qwen3-VL model (model_type='qwen3_vl'), got model_type={model_type}")

#     from transformers import Qwen3VLForConditionalGeneration

#     print("[smoke] loading Qwen3VLForConditionalGeneration...")
#     model = Qwen3VLForConditionalGeneration.from_pretrained(
#         args.llm_path,
#         torch_dtype=torch_dtype if torch_dtype != "auto" else None,
#         device_map=args.device_map,
#     )
#     model.eval()

#     # Must exist for MindMerger.forward().
#     _ = model.get_input_embeddings()
#     print("[smoke] model.get_input_embeddings(): OK")

#     print("[smoke] tokenizing prompt...")
#     enc = tokenizer(args.prompt, return_tensors="pt", truncation=True)
#     input_ids = enc["input_ids"].to(device)
#     attention_mask = enc["attention_mask"].to(device)

#     # MindMerger uses inputs_embeds, not input_ids.
#     inputs_embeds = model.get_input_embeddings()(input_ids)

#     # 1) Forward with labels => tests CE loss path.
#     print("[smoke] forward(inputs_embeds=..., labels=...)...")
#     labels = input_ids.clone()
#     with torch.no_grad():
#         out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
#     loss = out.loss
#     print(f"[smoke] forward loss: {loss.item():.6f}")

#     print("[smoke] SUCCESS: Qwen3-VL loads and returns a finite loss with inputs_embeds + labels.")


# if __name__ == "__main__":
#     main()

import argparse
import os

print("[smoke] importing torch...", flush=True)
import torch
print("[smoke] torch imported", flush=True)

print("[smoke] importing transformers basics...", flush=True)
from transformers import AutoTokenizer, AutoConfig
print("[smoke] transformers basics imported", flush=True)


def main() -> None:
    print("[smoke] entering main()", flush=True)

    parser = argparse.ArgumentParser(
        description="Smoke test: can this repo-style training loss path load Qwen3-VL and run with inputs_embeds?"
    )
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Hello, world!")
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    args = parser.parse_args()

    print("[smoke] args parsed", flush=True)

    # Cache setup
    default_cache_dir = os.path.join(os.path.dirname(__file__), ".hf_cache")
    os.environ.setdefault("HF_HOME", default_cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(default_cache_dir, "transformers"))

    print(f"[smoke] HF_HOME={os.environ.get('HF_HOME')}", flush=True)

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, "auto")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}, dtype={args.dtype}, device_map={args.device_map}", flush=True)

    print("[smoke] loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_path,
        use_fast=True,
        local_files_only=True,
    )
    print("[smoke] tokenizer loaded", flush=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("[smoke] loading config...", flush=True)
    llm_cfg = AutoConfig.from_pretrained(
        args.llm_path,
        local_files_only=True,
    )
    model_type = getattr(llm_cfg, "model_type", None)
    print(f"[smoke] model_type={model_type}", flush=True)

    if model_type != "qwen3_vl":
        raise ValueError(f"Expected qwen3_vl, got {model_type}")

    print("[smoke] importing Qwen3VLForConditionalGeneration...", flush=True)
    from transformers import Qwen3VLForConditionalGeneration
    print("[smoke] import OK", flush=True)

    print("[smoke] loading model (this may take time)...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.llm_path,
        torch_dtype=torch_dtype if torch_dtype != "auto" else None,
        device_map=args.device_map,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    print("[smoke] model loaded", flush=True)
    
    model.eval()

    print("[smoke] checking embeddings...", flush=True)
    _ = model.get_input_embeddings()
    print("[smoke] embeddings OK", flush=True)

    print("[smoke] tokenizing prompt...", flush=True)
    enc = tokenizer(args.prompt, return_tensors="pt", truncation=True)

    # safer device handling
    if args.device_map == "auto":
        model_device = next(model.parameters()).device
    else:
        model_device = torch.device(device)
        model.to(model_device)

    input_ids = enc["input_ids"].to(model_device)
    attention_mask = enc["attention_mask"].to(model_device)

    print("[smoke] building inputs_embeds...", flush=True)
    inputs_embeds = model.get_input_embeddings()(input_ids)

    print("[smoke] running forward...", flush=True)
    labels = input_ids.clone()
    with torch.no_grad():
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    loss = out.loss
    print(f"[smoke] forward loss: {loss.item():.6f}", flush=True)

    print("[smoke] SUCCESS ✅", flush=True)


if __name__ == "__main__":
    main()