
import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout=0.2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)
