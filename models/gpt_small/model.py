# models/gpt_small/model.py
"""Small GPT-style transformer for tinyfacts-learn."""
import torch
import torch.nn as nn


class GPTSmall(nn.Module):
    def __init__(self, vocab_size: int, context_size: int, n_embd: int,
                 n_heads: int, n_layers: int, ffn_dim: int, dropout: float):
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
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # Tie embedding and output weights
        self.head.weight = self.tok_emb.weight
        self._context_size = context_size
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
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
