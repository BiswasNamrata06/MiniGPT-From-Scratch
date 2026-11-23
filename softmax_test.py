
import torch
import torch.nn.functional as F

# Scores predicted by a model
logits = torch.tensor([1.2, 2.8, 0.4])

# Convert scores into probabilities
probabilities = F.softmax(logits, dim=0)

print("Logits:", logits)
print("Probabilities:", probabilities)
print("Total probability:", probabilities.sum())
