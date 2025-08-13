from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader, Dataset
import time


class BookDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        print(f"DEBUG: Dataset created with {len(sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)


def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"DEBUG: GPU Memory - Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f}MB, "
              f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.1f}MB")


def fine_tune_mistral_lora():
    print("DEBUG: Starting LoRA fine-tuning with Mistral-7B...")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEBUG: Using device: {device}")
    print_gpu_memory()

    # Load model and tokenizer
    print("DEBUG: Loading Mistral-7B model...")
    model_name = "mistralai/Mistral-7B-v0.1"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision to save memory
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"DEBUG: Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print_gpu_memory()

    # Re-tokenize data with Mistral tokenizer
    print("DEBUG: Re-tokenizing data with Mistral tokenizer...")
    with open('cleaned_text.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize with Mistral tokenizer
    tokens = tokenizer.encode(text)
    print(f"DEBUG: Total tokens with Mistral: {len(tokens)}")

    # Create new sequences
    max_length = 512
    sequences = []
    for i in range(0, len(tokens) - max_length, max_length):
        sequence = tokens[i:i + max_length]
        sequences.append(sequence)

    print(f"DEBUG: Created {len(sequences)} sequences for Mistral")

    # Setup LoRA configuration
    print("DEBUG: Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Mistral attention modules
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print_gpu_memory()

    # Setup training
    print("DEBUG: Setting up training components...")
    dataset = BookDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Small batch for 7B model
    optimizer = AdamW(model.parameters(), lr=2e-4)

    model.train()

    # Training loop
    total_start_time = time.time()

    for epoch in range(2):  # Fewer epochs for larger model
        print(f"\nDEBUG: Starting Epoch {epoch + 1}/2")
        epoch_start_time = time.time()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)

            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"DEBUG: Epoch {epoch + 1}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}")
                print_gpu_memory()

        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(dataloader)
        print(f"DEBUG: Epoch {epoch + 1} completed in {epoch_time:.1f}s. Average Loss: {avg_loss:.4f}")

    total_time = time.time() - total_start_time
    print(f"\nDEBUG: Total training time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")

    # Save LoRA model
    print("DEBUG: Saving Mistral LoRA model...")
    model.save_pretrained('./book-mistral-lora')
    tokenizer.save_pretrained('./book-mistral-lora')
    print("DEBUG: Mistral LoRA model saved successfully!")
    print("Training completed!")


if __name__ == "__main__":
    fine_tune_mistral_lora()