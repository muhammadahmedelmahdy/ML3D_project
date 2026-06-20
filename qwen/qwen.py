import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Define model path
model_id = "Qwen/Qwen3-8B"

print("Loading tokenizer and model (this may take a few minutes)...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # Automatically utilizes your allocated RTX 3090
)

# 2. Structure your prompt with the standard chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "when is your knowledge cutoff"}
]

# Formatting using the model's token template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 3. Configure the generator parameters
# IMPORTANT: For Qwen3 thinking mode, avoid greedy decoding (do not set do_sample=False). 
# Use a slight temperature to prevent generation loop bugs.
print("Generating response...")
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        top_k=20
    )

# Trim out the prompt text from the generated output tokens
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 4. Decode and print the result
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\n--- Model Output ---")
print(response)