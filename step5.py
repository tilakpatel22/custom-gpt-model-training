from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import torch


def test_lora_model(prompt, max_length=150):
    # Load base model
    base_model = GPT2LMHeadModel.from_pretrained('gpt2-large')

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, './book-gpt2-large-lora')

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    tokenizer.pad_token = tokenizer.eos_token

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Encode with attention mask
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + max_length,
            num_return_sequences=1,
            temperature=0.7,  # More conservative for better quality
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1  # Reduce repetition
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def multiple_prompts():
    # Test with prompts related to your geopolitics book
    prompts = [
        "The US-China relationship",
        "Great power politics involves",
        "The balance of power in international relations",
        "Offensive realism argues that",
        "International relations theory suggests",
        "The tragedy of great power politics",
        "Security competition between nations"
    ]

    print("=== Testing LoRA Fine-tuned GPT-2 Large ===")
    print("Based on Mearsheimer's 'The Tragedy of Great Power Politics'\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. Prompt: '{prompt}'")
        result = test_lora_model(prompt)
        print(f"Generated: {result}\n")
        print("-" * 80 + "\n")


if __name__ == "__main__":
    multiple_prompts()