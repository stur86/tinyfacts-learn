# models/gpt_rope/model.py
"""GPT-style transformer using rotary position embeddings (RoPE).

This is `gpt_small` with exactly one thing changed: position is injected by
rotating queries and keys inside every attention head instead of by adding a
learned vector to the token embeddings.

Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding",
           Su et al., 2021 (arXiv:2104.09864)

`nn.TransformerEncoderLayer` gives no access to Q and K between projection and
attention, so the block is written out here. Everything the layer does is kept:
pre-norm, GeLU, the same dropout placement, and the same parameter
initialisation (xavier on the fused QKV weight, zeroed attention biases,
PyTorch's default `nn.Linear` init elsewhere), so a difference against
`gpt_small` is the positional scheme and nothing else.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_tables(context_size: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin lookup tables for RoPE, both `[context_size, head_dim]`.

    Dimension pair `i` rotates at frequency `theta ** (-2i/head_dim)`: the first
    pairs turn once per token, the last ones once per thousands of tokens, so
    together they encode position across every scale the context spans.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE needs an even head dimension, got {head_dim}")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(context_size, dtype=torch.float32), inv_freq)
    # Halves convention (GPT-NeoX): pair dimension i with i + head_dim/2, so the
    # angle table is the half-width one repeated rather than interleaved.
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate `x` `[..., T, head_dim]` by the angle of its position.

    Each dimension pair is turned by `position * frequency`. A dot product
    between two rotated vectors then depends on their *difference* in position
    only, which is what makes the encoding relative.
    """
    out = x.float() * cos + _rotate_half(x.float()) * sin
    return out.to(x.dtype)


class RopeCausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE applied to Q and K."""

    def __init__(self, n_embd: int, n_heads: int, dropout: float):
        super().__init__()
        if n_embd % n_heads != 0:
            raise ValueError(f"n_embd={n_embd} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        # Match nn.MultiheadAttention's initialisation of the same tensors.
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, H, T, head_dim]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(y)


class RopeBlock(nn.Module):
    """Pre-norm transformer block — the norm_first=True layout of nn.TransformerEncoderLayer."""

    def __init__(self, n_embd: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.attn = RopeCausalSelfAttention(n_embd, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x), cos, sin))
        return x + self.ffn(self.norm2(x))


class GPTRoPE(nn.Module):
    def __init__(self, vocab_size: int, context_size: int, n_embd: int,
                 n_heads: int, n_layers: int, ffn_dim: int, dropout: float,
                 rope_theta: float):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            RopeBlock(n_embd, n_heads, ffn_dim, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # Tie embedding and output weights
        self.head.weight = self.tok_emb.weight
        self._context_size = context_size
        nn.init.normal_(self.tok_emb.weight, std=0.02)

        cos, sin = rope_tables(context_size, n_embd // n_heads, rope_theta)
        # Not persistent: they are a pure function of the config, so keeping them
        # out of the state_dict leaves checkpoints carrying weights only.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        if T > self._context_size:
            raise ValueError(
                f"Sequence of {T} tokens exceeds context_size={self._context_size}"
            )
        # [T, head_dim] broadcasts against the [B, H, T, head_dim] queries and keys.
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.norm(x))  # (B, T, vocab_size)


def build_model(config: dict, vocab_size: int) -> GPTRoPE:
    """Factory function — required interface for all model folders."""
    return GPTRoPE(
        vocab_size=vocab_size,
        context_size=config["context_size"],
        n_embd=config["n_embd"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        ffn_dim=config.get("ffn_dim", config["n_embd"] * 4),
        dropout=config.get("dropout", 0.1),
        rope_theta=config.get("rope_theta", 10000.0),
    )
