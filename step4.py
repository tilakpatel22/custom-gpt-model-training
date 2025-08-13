from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from peft import LoraConfig, get_peft_model  # Changed LoRAConfig to LoraConfig
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


def fine_tune_model_lora():
    print("DEBUG: Starting LoRA fine-tuning with GPT-2 Large...")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEBUG: Using device: {device}")
    print_gpu_memory()

    # Load data
    print("DEBUG: Loading training sequences...")
    sequences = torch.load('training_sequences.pt')
    print(f"DEBUG: Loaded {len(sequences)} sequences")

    # Load GPT-2 Large model
    print("DEBUG: Loading GPT-2 Large model...")
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')

    # Setup LoRA configuration
    print("DEBUG: Setting up LoRA configuration...")
    lora_config = LoraConfig(  # Changed from LoRAConfig to LoraConfig
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model = model.to(device)
    print("DEBUG: LoRA model moved to device")
    print_gpu_memory()

    # Setup training
    print("DEBUG: Setting up training components...")
    dataset = BookDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    print(f"DEBUG: DataLoader created. Batches per epoch: {len(dataloader)}")

    model.train()

    # Training loop
    total_start_time = time.time()

    for epoch in range(3):
        print(f"\nDEBUG: Starting Epoch {epoch + 1}/3")
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

            if batch_idx % 25 == 0:
                print(f"DEBUG: Epoch {epoch + 1}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}")
                print_gpu_memory()

        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(dataloader)
        print(f"DEBUG: Epoch {epoch + 1} completed in {epoch_time:.1f}s. Average Loss: {avg_loss:.4f}")

    total_time = time.time() - total_start_time
    print(f"\nDEBUG: Total training time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")

    # Save LoRA model
    print("DEBUG: Saving LoRA model...")
    model.save_pretrained('./book-gpt2-large-lora')
    print("DEBUG: LoRA model saved successfully!")
    print("Training completed!")


if __name__ == "__main__":
    fine_tune_model_lora()