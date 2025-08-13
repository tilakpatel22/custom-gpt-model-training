from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
import time
import psutil


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


def fine_tune_model():
    print("DEBUG: Starting fine-tuning with GPT-2 Large...")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEBUG: Using device: {device}")
    if torch.cuda.is_available():
        print(f"DEBUG: GPU: {torch.cuda.get_device_name(0)}")
        print_gpu_memory()

    # Load data
    print("DEBUG: Loading training sequences...")
    sequences = torch.load('training_sequences.pt')
    print(f"DEBUG: Loaded {len(sequences)} sequences")

    # Load GPT-2 Large model
    print("DEBUG: Loading GPT-2 Large model (774M parameters)...")
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    print(f"DEBUG: Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model = model.to(device)
    print("DEBUG: Model moved to device")
    print_gpu_memory()

    # Setup training with smaller batch size for large model
    print("DEBUG: Setting up training components...")
    dataset = BookDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Reduced batch size
    optimizer = AdamW(model.parameters(), lr=1e-5)  # Lower learning rate
    print(f"DEBUG: DataLoader created. Batches per epoch: {len(dataloader)}")

    model.train()
    print("DEBUG: Model set to training mode")

    # Training loop
    total_start_time = time.time()

    for epoch in range(2):
        print(f"\nDEBUG: Starting Epoch {epoch + 1}/2")
        epoch_start_time = time.time()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()

            # Move to device
            batch = batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_time = time.time() - batch_start_time

            if batch_idx % 50 == 0:
                print(f"DEBUG: Epoch {epoch + 1}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, Batch Time: {batch_time:.2f}s")
                print_gpu_memory()

        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(dataloader)
        print(f"DEBUG: Epoch {epoch + 1} completed in {epoch_time:.1f}s. Average Loss: {avg_loss:.4f}")

    total_time = time.time() - total_start_time
    print(f"\nDEBUG: Total training time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")

    # Save model
    print("DEBUG: Saving model...")
    model.save_pretrained('./book-gpt2-large-final')
    print("DEBUG: Model saved successfully!")
    print("Training completed!")


# Usage
if __name__ == "__main__":
    fine_tune_model()