from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import torch


def generate_with_fine_tuned_model(prompt, max_length=150):
    """Generate text using the LoRA fine-tuned model"""
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

    # Encode and generate
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def generate_with_base_model(prompt, max_length=150):
    """Generate text using the base GPT-2 Large model (no fine-tuning)"""
    # Load base model only
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    tokenizer.pad_token = tokenizer.eos_token

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Encode and generate
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def compare_model_responses():
    """Compare responses from fine-tuned vs base model"""

    # Mearsheimer-specific questions to test book knowledge
    questions = [
        "Explain buckpassing in great power politics",
        "Why can't great powers achieve global hegemony?"
    ]

    print("=" * 80)
    print("COMPARING FINE-TUNED vs BASE GPT-2 LARGE MODEL")
    print("=" * 80)

    for i, question in enumerate(questions, 1):
        print(f"\n{i}. QUESTION: '{question}'")
        print("-" * 60)

        # Generate with fine-tuned model
        print("🔹 FINE-TUNED MODEL (trained on Mearsheimer's book):")
        fine_tuned_response = generate_with_fine_tuned_model(question)
        print(fine_tuned_response)

        print("\n" + "-" * 60)

        # Generate with base model
        print("🔸 BASE MODEL (no fine-tuning):")
        base_response = generate_with_base_model(question)
        print(base_response)

        print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_model_responses()