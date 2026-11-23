
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_block import TransformerBlock


class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_embd=128,
        block_size=128,
        num_heads=4,
        num_layers=4
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(n_embd, num_heads, block_size)
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        if T > self.block_size:
            raise ValueError("Input exceeds block_size")

        token_emb = self.token_embedding(idx)

        positions = torch.arange(T, device=idx.device)
        position_emb = self.position_embedding(positions)

        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

   
    
    
    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None
    ):

        if temperature <= 0:
            raise ValueError("Temperature must be greater than zero")

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Keep only top-k predictions
            if top_k is not None:
                k = min(top_k, logits.size(-1))

                values, indices = torch.topk(logits, k)

                filtered_logits = torch.full_like(
                    logits,
                    float("-inf")
                )

                filtered_logits.scatter_(
                    -1,
                    indices,
                    values
                )

                logits = filtered_logits

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat((idx, next_token), dim=1)

        return idx
