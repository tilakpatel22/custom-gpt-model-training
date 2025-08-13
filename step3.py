from transformers import GPT2Tokenizer
import torch


def prepare_training_data(input_file='cleaned_text.txt', max_length=512):
    # Load GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Read cleaned text
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize the entire text
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens)}")

    # Create training sequences
    sequences = []
    for i in range(0, len(tokens) - max_length, max_length):
        sequence = tokens[i:i + max_length]
        sequences.append(sequence)

    # Save training data
    torch.save(sequences, 'training_sequences.pt')
    print(f"Created {len(sequences)} training sequences")

    return sequences, tokenizer


# Usage
sequences, tokenizer = prepare_training_data()