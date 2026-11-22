
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local Qwen model directory
MODEL_PATH = (
    r"C:\Users\Sheru\Downloads\NLP-20261121T074752Z-1-001"
    r"\NLP\phishing_papers\models\Qwen2.5-1.5B-Instruct"
)

print("Loading Qwen tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Loading Qwen model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

model.eval()

print("\nQwen model loaded successfully!")
print("Model device:", model.device)
print("GPU:", torch.cuda.get_device_name(0))


# Test question
question = "What are common characteristics of phishing attacks?"

messages = [
    {
        "role": "user",
        "content": question
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

print("\nGenerating test answer...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

answer = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("\nQWEN TEST ANSWER")
print("-" * 50)
print(answer)
print("-" * 50)

print("\nGPU memory allocated:",
      round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")
