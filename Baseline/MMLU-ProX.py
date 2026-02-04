import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# ---- Device selection (Compute Canada) ----
assert torch.cuda.is_available(), "CUDA not available – did you request a GPU?"
device = torch.device("cuda")

# A100 supports bfloat16; V100/T4 use float16
cap_major = torch.cuda.get_device_capability(0)[0]
dtype = torch.bfloat16 if cap_major >= 8 else torch.float16

# ---- Load model efficiently ----
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)

@torch.inference_mode()
def generate_text_response(user_text: str, max_new_tokens: int = 8) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
    )

    gen_only = generated_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        gen_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

# ---- Smoke test ----
print(
    generate_text_response(
        "Answer with just one letter: A, B, C, or D. What is 2+2? "
        "A)3 B)4 C)5 D)6"
    )
)
