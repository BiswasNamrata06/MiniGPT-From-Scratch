
import torch
from models.gpt_model import GPTLanguageModel

vocab_size = 65

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=128,
    block_size=128,
    num_heads=4,
    num_layers=4
)

# Two batches, eight tokens each
idx = torch.randint(0, vocab_size, (2, 8))

logits = model(idx)

print("Input shape:", idx.shape)
print("Output shape:", logits.shape)
print("Total parameters:", sum(p.numel() for p in model.parameters()))

print("\nMiniGPT model created successfully!")

