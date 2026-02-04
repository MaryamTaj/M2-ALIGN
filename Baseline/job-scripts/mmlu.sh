import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset

# Access the MMLU-ProX dataset:
# LANGS = ["en","zh","ja","ko","fr","de","es","pt","ar","th","hi","bn","sw","af","cs",
#          "hu","id","it","mr","ne","ru", "sr", "te", "uk", "ur", "vi", "wo", "yo", "zu"]
# test = {}
# for language in LANGS:
#     test[language] = load_dataset("li-lab/MMLU-ProX", language, split="test")

# Access Qwen3-VL-8B-Instruct:
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# Use GPU, if possible
if torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)
model.to(device)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)

@torch.inference_mode()
def generate_text_response(user_text: str, max_new_tokens: int = 32) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
            ],
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

    prompt_len = inputs["input_ids"].shape[1]
    gen_only = generated_ids[:, prompt_len:]

    out = processor.batch_decode(
        gen_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return out.strip()


# Quick smoke test
print(generate_text_response("Answer with just one letter: A, B, C, or D. What is 2+2? A)3 B)4 C)5 D)6", max_new_tokens=8))
