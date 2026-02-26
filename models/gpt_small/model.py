# models/gpt_small/model.py
"""Small GPT-style transformer for tinyfacts-learn."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, context_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # Causal mask: upper triangle is -inf
        mask = torch.triu(torch.ones(context_size, context_size), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).split(C, dim=2)  # 3 × (B, T, C)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = map(split_heads, qkv)  # (B, nh, T, hd)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, nh, T, T)
        attn = attn.masked_fill(self.mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, context_size: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_heads, context_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTSmall(nn.Module):
    def __init__(self, vocab_size: int, context_size: int, n_embd: int,
                 n_heads: int, n_layers: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(context_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_heads, context_size, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # Tie embedding and output weights
        self.head.weight = self.tok_emb.weight
        self._context_size = context_size
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab_size)


def build_model(config: dict, vocab_size: int) -> GPTSmall:
    """Factory function — required interface for all model folders."""
    return GPTSmall(
        vocab_size=vocab_size,
        context_size=config["context_size"],
        n_embd=config["n_embd"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        ffn_dim=config.get("ffn_dim", config["n_embd"] * 4),
        dropout=config.get("dropout", 0.1),
    )
