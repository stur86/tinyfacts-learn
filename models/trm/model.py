# models/trm/model.py
"""Tiny Recursion Model (TRM) for tinyfacts-learn.

Reference: "Less is More: Recursive Reasoning with Tiny Networks"
           Jolicoeur-Martineau, 2025 (arXiv:2510.04871)
"""
import torch
import torch.nn as nn


class TRM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        n_embd: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        dropout: float,
        n_recursions: int,
        T: int,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(context_size, n_embd)
        self.drop = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm (GPT-2 style)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(n_embd),
            enable_nested_tensor=False,
        )
        # Output head — tied to tok_emb
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        # Halt probability head — initialised to zero (biased toward continuing)
        self.q_head = nn.Linear(n_embd, 1, bias=False)
        nn.init.zeros_(self.q_head.weight)

        self._context_size = context_size
        self._n_recursions = n_recursions
        self._T = T

        nn.init.normal_(self.tok_emb.weight, std=0.02)

    # ------------------------------------------------------------------
    # Public helpers used by train.py during deep supervision
    # ------------------------------------------------------------------

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        """Token + positional embedding → [B, L, D]."""
        B, L = idx.shape
        positions = torch.arange(L, device=idx.device)
        return self.drop(self.tok_emb(idx) + self.pos_emb(positions))

    def latent_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        n: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recursion cycle: n latent z-updates then 1 answer y-update.

        All three inputs are [B, L, D] in embedding space.
        The same transformer weights are used for every call (shared network).

        Args:
            x: embedded input sequence
            y: current predicted-answer embedding
            z: current latent reasoning state
            n: number of z-update iterations

        Returns:
            (y_new, z_new)
        """
        L = x.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=x.device
        )
        for _ in range(n):
            z = self.transformer(x + y + z, mask=causal_mask, is_causal=True)
        y = self.transformer(y + z, mask=causal_mask, is_causal=True)
        return y, z

    # ------------------------------------------------------------------
    # Standard forward — used by generate.py and inspect command
    # ------------------------------------------------------------------

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Full recursive forward pass for inference.

        Runs T-1 no-grad warm-up cycles then 1 final cycle.
        Returns logits [B, L, vocab_size].
        """
        x = self.embed(idx)
        B, L, D = x.shape
        y = torch.zeros(B, L, D, device=x.device)
        z = torch.zeros(B, L, D, device=x.device)

        with torch.no_grad():
            for _ in range(self._T - 1):
                y, z = self.latent_recursion(x, y, z, self._n_recursions)

        y, z = self.latent_recursion(x, y, z, self._n_recursions)
        return self.head(y)


def build_model(config: dict, vocab_size: int) -> TRM:
    """Factory function — required interface for all model folders."""
    return TRM(
        vocab_size=vocab_size,
        context_size=config["context_size"],
        n_embd=config["n_embd"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        ffn_dim=config.get("ffn_dim", config["n_embd"] * 4),
        dropout=config.get("dropout", 0.1),
        n_recursions=config.get("n_recursions", 6),
        T=config.get("T", 3),
    )
