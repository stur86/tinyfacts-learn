# TRM — Tiny Recursive Model

Used by: `trm`

Reference: "Less is More: Recursive Reasoning with Tiny Networks", Jolicoeur-Martineau (2025), arXiv:2510.04871

## Core idea

Standard transformers get smarter by going *wider* (larger `d_model`) or *deeper* (more layers). TRM instead lets a small network think *longer*: the same weights are applied repeatedly in a latent space, allowing the model to iteratively refine its answer before committing.

This is inspired by the observation that humans don't always know the answer to a hard question instantly — they think about it step by step, updating their belief until confident.

## Three latent sequences

At any point during the recursion, the model maintains three tensors of shape `[B, L, D]`:

| Tensor | Role |
|--------|------|
| `x` | The embedded input — fixed throughout, never updated |
| `y` | The current predicted answer in embedding space |
| `z` | The latent reasoning scratchpad — internal working memory |

Both `y` and `z` start as zeros.

## The recursion

One **cycle** consists of:

1. **n_recursions z-updates** (reasoning): the latent state refines itself by looking at the input, current answer, and its own previous state:
   ```
   z = Transformer(x + y + z)   # repeated n_recursions times
   ```

2. **1 y-update** (answering): write a new answer based on the latest reasoning state:
   ```
   y = Transformer(y + z)
   ```

The critical property: **the same transformer weights are used for every call**, across both the z-updates and the y-update, and across all cycles. The network is tiny in parameters but can execute an arbitrary amount of computation by iterating.

## T cycles and warm-up

A full forward pass runs **T cycles** total:

- **T−1 warm-up cycles** (no gradient): y and z are allowed to converge without incurring memory overhead. These cycles are run under `torch.no_grad()`.
- **1 final cycle** (with gradient): backpropagation flows only through this last cycle.

This mirrors the intuition that most of the reasoning happens in warm-up, and only the final "answer-write" step needs to be differentiable.

## The halt head (`q_head`)

A scalar linear layer on top of `y` outputs a **halt logit**:

```python
q_logit = q_head(y).mean()   # scalar
p_halt  = sigmoid(q_logit)   # probability the current answer is good enough
```

- Initialized to zero weights, so the model starts by always wanting to continue.
- Trained with binary cross-entropy: the target is 1.0 if the current logits are correct on this batch, 0.0 otherwise. This teaches `q_head` to estimate confidence.
- At inference, `sigmoid(q_logit) > 0.5` (i.e., `q_logit > 0`) triggers early stopping.

`q_head` is **inference-only** from the model class's perspective — `forward()` ignores it. Training uses the helpers `embed()`, `latent_recursion()`, `head()`, and `q_head()` directly.

## Deep supervision training loop

To help the network learn to reason iteratively (not just on the final cycle), training applies **n_supervision outer loops** per batch:

```
for sup in range(n_supervision):
    x_emb = embed(x)
    # T-1 warm-up (no grad)
    for _ in range(T-1):
        y, z = latent_recursion(x_emb, y, z, n_recursions)
    # 1 final cycle (with grad)
    y, z = latent_recursion(x_emb, y, z, n_recursions)
    logits = head(y)
    loss = lm_loss + halt_loss
    loss.backward()
    optimizer.step()
    if q_logit > 0:
        break   # early stopping
```

Each supervision step picks up `y` and `z` where the previous one left off, so the latent state accumulates across supervision steps within a single batch.

## EMA (Exponential Moving Average)

With `ema_decay > 0`, a shadow copy of the model weights is maintained:

```
ema_weights = ema_decay * ema_weights + (1 - ema_decay) * model_weights
```

Checkpoints save `ema_weights` instead of the raw weights. EMA smooths out training noise and often gives better inference quality, especially for models that use stochastic training loops like deep supervision.

## Config keys

| Key | Role |
|-----|------|
| `n_embd` | `D` — the shared embedding/hidden dimension for x, y, z |
| `n_heads` | Attention heads in the shared transformer |
| `n_layers` | Layers in the shared transformer |
| `ffn_dim` | FFN width |
| `n_recursions` | Number of z-update iterations per cycle |
| `T` | Total cycles (T-1 warm-up + 1 gradient) |
| `n_supervision` | Outer supervision loops per training step |
| `ema_decay` | EMA coefficient (0 = disabled) |

## Trade-offs vs GPT

| | GPT | TRM |
|-|-----|-----|
| Parameters | proportional to n_layers | same regardless of T or n_recursions |
| Inference cost | fixed | scales with T × n_recursions |
| Reasoning | single pass | iterative refinement |
| Training | standard backprop | deep supervision + EMA |
