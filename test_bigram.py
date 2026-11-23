
import torch
from models.bigram_model import BigramLanguageModel

# Our vocabulary has 5 characters
vocab_size = 5

# Create model
model = BigramLanguageModel(vocab_size)

# Input character IDs
input_ids = torch.tensor([0, 1, 2, 3])

# Pass input through model
logits = model(input_ids)

print("Input:", input_ids)
print("Output logits:")
print(logits)

print("Output shape:", logits.shape)
