
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

# Sample input and next-token targets
x = torch.randint(0, vocab_size, (2, 8))
y = torch.randint(0, vocab_size, (2, 8))

# Forward pass
logits, loss = model(x, y)

print("Logits shape:", logits.shape)
print("Training loss:", loss.item())

# Test text generation
start = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(start, max_new_tokens=20)

print("Generated token shape:", generated.shape)
print("\nGPT training test successful!")
