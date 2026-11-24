import torch
from models.gpt_model import GPTLanguageModel

vocab_size = 65

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=64,
    block_size=64,
    num_heads=4,
    num_layers=2
)

idx = torch.randint(0, vocab_size, (2, 8))

logits, _ = model(idx)

print("Input shape:", idx.shape)
print("Output shape:", logits.shape)
print("Total parameters:", sum(p.numel() for p in model.parameters()))

print("\nMiniGPT model created successfully!")