
import torch
import torch.nn as nn

from models.multi_head_attention import MultiHeadAttention
from models.feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()

        # Calculate the size of each attention head
        head_size = n_embd // num_heads

        # Multi-head attention
        self.attention = MultiHeadAttention(
            num_heads,
            head_size,
            n_embd,
            block_size
        )

        # Feed-forward network
        self.feed_forward = FeedForward(n_embd)

        # Layer normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        # Attention + residual connection
        x = x + self.attention(self.ln1(x))

        # Feed-forward + residual connection
        x = x + self.feed_forward(self.ln2(x))

        return x
