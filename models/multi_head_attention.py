
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Causal mask
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Calculate attention scores
        wei = q @ k.transpose(-2, -1)

        # Scale attention scores
        wei = wei * (k.shape[-1] ** -0.5)

        # Apply causal mask
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # Convert scores into probabilities
        wei = F.softmax(wei, dim=-1)

        # Weighted sum of values
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size)
            for _ in range(num_heads)
        ])

        self.proj = nn.Sequential(
            nn.Linear(num_heads * head_size, n_embd),
            nn.Dropout(0.2)
        )

    def forward(self, x):

        # Run all attention heads in parallel
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        # Combine their outputs
        out = self.proj(out)

        return out
