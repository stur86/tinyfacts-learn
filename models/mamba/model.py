# models/mamba/model.py
"""Mamba (S6) selective state-space model for tinyfacts-learn.

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
           Gu & Dao, 2023 (arXiv:2312.00752)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = expand * d_model
        dt_rank = math.ceil(d_model / 16)
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank

        self.norm = nn.LayerNorm(d_model)
        # Project input to main branch x and gate z
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        # Causal depthwise conv1d for local context mixing
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )
        # Project x to low-rank dt, B, C (all input-dependent — this is the "selective" part)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        # Expand dt from low-rank to full d_inner
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # A: [d_inner, d_state] stored as log so A = -exp(A_log) stays strictly negative
        A = torch.arange(1, d_state + 1, dtype=torch.float).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        # D: per-channel skip connection weight
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self._init_dt_bias()

    def _init_dt_bias(self):
        """Initialise dt_proj bias so softplus(dt) starts in [0.001, 0.1]."""
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        # inv_softplus: dt_bias such that softplus(dt_bias) = dt
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Selective state-space scan (S6).

        Args:
            x: [B, L, d_inner]
        Returns:
            y: [B, L, d_inner]
        """
        B, L, _ = x.shape
        A = -torch.exp(self.A_log)  # [d_inner, d_state], strictly negative

        # Compute input-dependent dt, B, C in one projection then split
        xz = self.x_proj(x)  # [B, L, dt_rank + 2*d_state]
        dt, B_mat, C = xz.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # [B, L, d_inner], strictly positive

        # Discretize via zero-order hold (simplified):
        #   dA = exp(dt * A),  dB = dt * B
        # dt: [B, L, d_inner, 1],  A: [d_inner, d_state] → broadcasts to [B, L, d_inner, d_state]
        dA = torch.exp(dt.unsqueeze(-1) * A)        # [B, L, d_inner, d_state]
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(2)  # [B, L, d_inner, d_state]

        # Sequential scan: h_t = dA_t * h_{t-1} + dB_t * x_t,  y_t = C_t · h_t
        # For L=128, a Python loop is fast enough; no custom CUDA kernel needed.
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)  # [B, d_inner]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # [B, L, d_inner]

        return y + x * self.D  # skip connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
        Returns:
            [B, L, d_model]
        """
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # each [B, L, d_inner]

        # Causal depthwise conv1d: pad left with d_conv-1 zeros, trim right
        L = x.shape[1]
        x = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x = F.silu(x)

        y = self._selective_scan(x)
        y = y * F.silu(z)  # gating

        return self.out_proj(y) + residual


class MambaLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        context_size: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._context_size = context_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied embeddings
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # No positional embeddings: causality is inherent in the sequential scan
        x = self.drop(self.tok_emb(idx))  # [B, L, d_model]
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm_f(x))  # [B, L, vocab_size]


def build_model(config: dict, vocab_size: int) -> MambaLM:
    """Factory function — required interface for all model folders."""
    return MambaLM(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        context_size=config.get("context_size", 128),
        d_state=config.get("d_state", 16),
        d_conv=config.get("d_conv", 4),
        expand=config.get("expand", 2),
        dropout=config.get("dropout", 0.0),
    )
