# Mamba — Selective State Space Model (S6)

Used by: `mamba`

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", Gu & Dao (2023), arXiv:2312.00752

## Background: State Space Models

A **state space model** (SSM) describes a system that evolves over time through a hidden state. In continuous time:

```
h'(t) = A h(t) + B x(t)       # state update
y(t)  = C h(t) + D x(t)       # output
```

Where:
- `x(t)`: scalar input at time t
- `h(t)`: N-dimensional hidden state (the "memory")
- `y(t)`: scalar output at time t
- `A` [N×N]: how the state evolves on its own
- `B` [N×1]: how the input enters the state
- `C` [1×N]: how the state is read out
- `D` [1]: direct skip connection from input to output

For sequence modelling, inputs and outputs are sequences of vectors (not scalars), so we run `d_inner` independent SSMs in parallel — one per channel — all sharing the same N (called `d_state`).

## Discretization

Neural networks work in discrete time (one token at a time). The continuous system is discretized using a learned **step size** Δ (delta) via the zero-order hold (ZOH) method:

```
Ā = exp(Δ · A)
B̄ = Δ · B          (simplified ZOH; exact form uses matrix inverse)
```

This gives the discrete recurrence:

```
h_t = Ā · h_{t-1} + B̄ · x_t
y_t = C · h_t + D · x_t
```

Larger Δ means "spend more time at this input" — the state integrates the current token more strongly and the previous state decays more.

## S4 → S4D → Mamba: the path to selectivity

**S4** (2021) showed that SSMs work well for sequences if `A` is initialized with HiPPO matrices, which are designed to optimally memorize history. For training efficiency, S4 computes the whole sequence in one shot using convolutions in the frequency domain.

**S4D** simplified this by restricting `A` to be diagonal (each hidden unit evolves independently), making it easier to implement while retaining most of the performance.

**Mamba (S6)** makes the critical leap: instead of fixed `A`, `B`, `C`, `Δ`, it makes **B, C, and Δ input-dependent**:

```
Δ(x) = softplus( dt_proj( x_proj(x)[..., :dt_rank] ) )    # [B, L, d_inner]
B(x) = x_proj(x)[..., dt_rank : dt_rank+d_state]           # [B, L, d_state]
C(x) = x_proj(x)[..., dt_rank+d_state : ]                  # [B, L, d_state]
```

This **selectivity** is the core innovation: the model can dynamically choose what to remember and what to forget based on the content of each token, rather than applying the same fixed dynamics to everything. `A` remains a learned-but-fixed parameter (log-parameterized to stay negative, ensuring stability).

## Mamba block structure

Each block follows a residual + pre-norm design (similar to a transformer layer):

```
residual = x
x = LayerNorm(x)                          # [B, L, d_model]

x, z = split( in_proj(x) )               # each [B, L, d_inner],  d_inner = expand × d_model

x = causal_conv1d(x)                     # local context mixing
x = SiLU(x)

y = selective_ssm(x)                     # the recurrent core
y = y * SiLU(z)                          # gating

output = out_proj(y) + residual          # [B, L, d_model]
```

### Causal conv1d

Before the SSM, a depthwise (grouped) 1D convolution with a short kernel (`d_conv=4`) mixes nearby tokens locally. This gives the model a way to look at the last few tokens in a single step before the recurrent scan processes the sequence. The convolution is made causal by padding `d_conv-1` zeros on the left (and trimming the right).

### Selective scan

The scan runs sequentially over time. For each time step t:

```
dA_t = exp(Δ_t ⊗ A)          # [B, d_inner, d_state] — per-step state decay
dB_t = Δ_t ⊗ B_t             # [B, d_inner, d_state] — per-step input gate

h_t  = dA_t * h_{t-1}  +  dB_t * x_t    # state update
y_t  = sum(h_t * C_t, dim=-1)            # readout: [B, d_inner]
```

Plus the skip connection `y += D * x`.

**Why a Python loop is fine here**: the efficient Mamba paper uses custom CUDA/Triton kernels to run the scan in parallel (via parallel prefix sums). But for `L=128` tokens, the sequential loop is fast enough on a single GPU and requires zero extra dependencies.

### Gating

```
output = SSM_out * SiLU(z)
```

The gate `z` (the other half of `in_proj`) acts as a learned filter: it can suppress parts of the SSM output that are not useful for predicting the next token. This is the same gating pattern used in GLU / SwiGLU variants.

## No positional embeddings

Unlike transformers, Mamba does not need positional embeddings. Causality and position are implicit: the scan is strictly left-to-right, and the step size Δ is input-dependent, so the model naturally learns to encode "how far back" relevant information is. The `context_size` config key is still used by the dataset to define sliding window length, but it is not used by the model itself.

## Parameter count (d_model=128, d_inner=256, d_state=16, dt_rank=8, n_layers=6)

| Component | Params |
|-----------|--------|
| tok_emb (tied with head) | 140,800 |
| Per block: norm | 256 |
| Per block: in_proj | 65,536 |
| Per block: conv1d | 1,280 |
| Per block: x_proj | 10,240 |
| Per block: dt_proj | 2,304 |
| Per block: A_log | 4,096 |
| Per block: D | 256 |
| Per block: out_proj | 32,768 |
| **Per block total** | **116,736** |
| 6 blocks | 700,416 |
| norm_f | 256 |
| **Grand total** | **~841K** |

## Config keys

| Key | Role |
|-----|------|
| `d_model` | Hidden dimension (embedding size) |
| `n_layers` | Number of Mamba blocks |
| `d_state` | SSM state dimension N (size of each channel's hidden state) |
| `d_conv` | Kernel size for the depthwise conv1d |
| `expand` | Channel expansion factor (`d_inner = expand × d_model`) |
| `dropout` | Applied after token embedding |
| `context_size` | Sliding window length for the dataset (not used by the model) |

## Trade-offs vs transformers

| | GPT | Mamba |
|-|-----|-------|
| Complexity per token | O(L) attention + O(d²) FFN | O(d_state × d_inner) recurrence |
| Memory growth with L | O(L²) KV-cache | O(d_inner × d_state) fixed state |
| Positional encoding | Required | Not needed |
| Parallelism during training | Full (matrix ops) | Scan (parallel prefix or loop) |
| Selective memory | No (fixed softmax weights) | Yes (input-dependent Δ, B, C) |
