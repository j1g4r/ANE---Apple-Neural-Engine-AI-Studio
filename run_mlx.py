import time
from mlx_lm import load, generate

print("Loading model. This will download the model on the first run...")

# We use a quantized version of Qwen 2.5 0.5B for a fast, lightweight demo
model_name = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
model, tokenizer = load(model_name)

prompt = "What is the Apple Neural Engine?"

# Apply the chat template if standard for this model, otherwise simple format
if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

print(f"\nPrompt: {prompt}\n")
print("Generating response...")

start = time.time()
response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=256)
end = time.time()

print(f"\nResponse generated in {end - start:.2f} seconds.")
