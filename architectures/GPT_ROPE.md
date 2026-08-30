# GPT with RoPE — Rotary Position Embeddings

Used by: `gpt_rope`

Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding", Su et al. (2021), arXiv:2104.09864

Everything not about position is [GPT.md](GPT.md): decoder-only, pre-norm blocks, GeLU feed-forward, tied embeddings, cross-entropy on the next token. This document covers the one part that differs.

## The problem with a learned position table

`gpt_small` adds a learned vector to each token embedding:

```
x_t = tok_emb(idx_t) + pos_emb(t)
```

Two things follow from that. The table has one row per position, so position 127 is only ever trained by tokens that happen to land at index 127, and nothing at all is defined past `context_size`. And the signal is *absolute*: the model has to learn, separately for every pair, that 17→20 and 90→93 are the same three-token gap.

## Rotation instead of addition

RoPE adds nothing to the embedding. It rotates the query and key vectors *inside* each attention head, by an angle proportional to the position they sit at.

Split a head's `d_head`-dimensional vector into `d_head/2` pairs of coordinates. Treat each pair as a point in a plane and turn it by `θ_i · t`, where `t` is the token's position and `θ_i` is that pair's own frequency:

```
freq_i = base ** (-2i / d_head)        i = 0 … d_head/2 - 1
angle  = t · freq_i
```

The first pairs turn roughly once per token; the last turn once per thousands of tokens. Together they write position across every scale the context spans — the same construction as sinusoidal encodings, applied as a rotation rather than an offset.

In code, with the halves convention (pair coordinate `i` with `i + d_head/2`):

```python
rotate_half(x) = cat(-x[d/2:], x[:d/2])
apply_rope(x)  = x * cos(angle) + rotate_half(x) * sin(angle)
```

Applied to Q and K only. **V is left alone** — values carry content, and rotating them would distort what attention actually retrieves.

## Why this makes attention relative

An attention score is a dot product. Rotating both vectors by their own positions gives

```
⟨R(m)·q , R(n)·k⟩ = ⟨q , R(n−m)·k⟩
```

because rotations compose and the rotation matrix is orthogonal. The absolute positions cancel; only `n − m` survives. The model gets relative position **for free in the score**, without a bias table, a second attention term, or any parameters at all.

Two further consequences:

- **Rotation preserves norm.** `‖R·q‖ = ‖q‖`, so RoPE cannot inflate or damp attention logits the way an additive signal competing with token content can.
- **Nothing is learned.** There is no position table to train, so no position is under-trained, and the frequencies are defined for any `t` — extending the context is a matter of the table's length, not of untrained rows.

## Cost against `gpt_small`

| | gpt_small | gpt_rope |
|---|---|---|
| position parameters | `context_size × n_embd` | 0 |
| position signal | absolute, learned, added once at the input | relative, fixed, applied in every layer |
| per-token compute | — | one multiply-add per Q and K element, per layer |

At the current config that is 16,384 parameters removed (932,864 → 916,480) for a small constant of extra arithmetic in attention.

## Implementation notes

`nn.TransformerEncoderLayer` fuses projection and attention, and never exposes Q and K in between, so `models/gpt_rope/model.py` writes the block out by hand. It is a deliberate transcription of what the built-in layer does — same pre-norm order, same GeLU, same dropout placement, same initialisation (xavier on the fused QKV weight, zeroed attention biases, PyTorch's default `nn.Linear` init elsewhere) — so that a `val_loss` difference against `gpt_small` is attributable to the positional scheme and not to an incidental change.

The cos/sin tables are registered as **non-persistent** buffers: they are a pure function of `context_size`, `n_heads`, `n_embd` and `rope_theta`, so they move with `.to(device)` but stay out of the checkpoint. Attention itself is `F.scaled_dot_product_attention(..., is_causal=True)`.

Rotation is computed in float32 and cast back to the activation dtype, so the angles stay exact under mixed precision.

## Config keys

Every key `gpt_small` has, plus:

| Key | Role |
|-----|------|
| `rope_theta` | Base of the frequency geometric series (10000.0, the RoFormer default). Larger stretches the wavelengths, which is the usual knob for extending context |

`rope_theta` is an architectural key: `train.py` refuses to resume a checkpoint across a change to it.
