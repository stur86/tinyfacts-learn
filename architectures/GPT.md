# GPT — Generative Pre-trained Transformer

Used by: `gpt_small`, `gpt_tiny`

## Overview

A decoder-only transformer trained autoregressively: at every position the model predicts the next token using only the tokens that came before it. The architecture here follows GPT-2 style with pre-normalization.

## Components

### 1. Embeddings

```
x = dropout(tok_emb(idx) + pos_emb(positions))
```

- **Token embedding** (`tok_emb`): maps each integer token ID to a `d_model`-dimensional vector. Learned.
- **Positional embedding** (`pos_emb`): a second learned matrix of shape `[context_size, d_model]`. Row `t` is added to position `t`. This gives the model knowledge of where in the sequence each token sits.
- Both are summed element-wise, then dropout is applied.

### 2. Transformer layers (×n_layers)

Each layer applies two sub-blocks, both with a **pre-norm** (LayerNorm before the operation, not after). This is the GPT-2 convention and tends to train more stably than post-norm.

#### 2a. Causal self-attention

```
attn_out = MultiHeadAttention(LayerNorm(x), mask=causal_mask)
x = x + attn_out
```

Multi-head attention projects the input into queries Q, keys K, and values V (split across `n_heads` heads). Each head computes:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_head) · V
```

The **causal mask** is an upper-triangular matrix filled with −∞. After softmax, any position can only attend to positions ≤ itself — the model cannot "cheat" by looking at future tokens.

Outputs from all heads are concatenated and projected back to `d_model`.

#### 2b. Feed-forward network (FFN)

```
ffn_out = Linear(GELU(Linear(LayerNorm(x))))
x = x + ffn_out
```

A two-layer MLP that expands to `ffn_dim` then contracts back to `d_model`. GELU (Gaussian Error Linear Unit) is used as the activation — it's smooth and empirically outperforms ReLU on language tasks.

### 3. Final LayerNorm

Applied after all layers before the output head. (`nn.TransformerEncoder` adds this automatically when `norm` is provided.)

### 4. Output head (tied embeddings)

```python
self.head = nn.Linear(d_model, vocab_size, bias=False)
self.head.weight = self.tok_emb.weight  # weight tying
```

The output projection (`head`) shares its weight matrix with the token embedding. This is **weight tying**: it enforces that the representation learned to embed token `t` as input is the same representation used to predict token `t` as output. It cuts parameters significantly (saves `vocab_size × d_model` weights) and consistently improves language model perplexity.

## Training objective

Cross-entropy loss between the model's logits at each position and the next token (the target is the input shifted left by one). This is standard causal language modelling.

## Config keys

| Key | Role |
|-----|------|
| `n_embd` | `d_model` — embedding / hidden dimension |
| `n_heads` | Number of attention heads per layer (`d_head = n_embd / n_heads`) |
| `n_layers` | Number of transformer layers |
| `ffn_dim` | Hidden width of the FFN (typically 2–4× `n_embd`) |
| `context_size` | Maximum sequence length (size of the positional embedding table) |
| `dropout` | Applied after embeddings and inside attention/FFN |

## Complexity

- Time per token: O(L · d_model²) dominated by attention O(L²) for long sequences
- For L=128, attention cost is small; FFN dominates at this scale
