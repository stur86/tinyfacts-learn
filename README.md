# tinyfacts-learn

Train and run small language models (~1M parameters) on a restricted vocabulary derived from xkcd's [Thing Explainer](https://xkcd.com/thing-explainer/) word list. All text data lives in the `tinyfacts-gen/` submodule.

## Setup

```bash
uv sync
```

This installs all dependencies including PyTorch (CUDA 12.6 build). For CPU-only or a different CUDA version, edit `pyproject.toml` accordingly before syncing.

## Available models

| Model | Description |
|-------|-------------|
| `gpt_small` | GPT-style transformer, ~930K parameters (n_embd=128, 2 layers) |
| `gpt_tiny` | Same architecture as gpt_small, much smaller (n_embd=32, 1 layer) |
| `trm` | Tiny Recursive Model — iterative latent reasoning with a learned halt signal |
| `mamba` | Selective state-space model (S6), ~841K parameters (d_model=128, 6 layers) |

Architecture deep-dives live in [`architectures/`](architectures/).

## Training

```bash
uv run tinyfacts train gpt_small
```

Training writes checkpoints to `models/<name>/checkpoints/` and per-step stats to `models/<name>/runs/run_<timestamp>.jsonl`.

Use `--dry-run` to verify the setup runs without committing a full training:

```bash
uv run tinyfacts train gpt_small --dry-run
```

## Inspect

Show the model config, parameter counts, and a summary of all training runs:

```bash
uv run tinyfacts inspect gpt_small
```

Optionally pass a specific checkpoint to verify it loads:

```bash
uv run tinyfacts inspect gpt_small --checkpoint models/gpt_small/checkpoints/<file>.pt
```

## Inference

Interactive prompt loop (loads the latest checkpoint automatically):

```bash
uv run tinyfacts generate gpt_small
```

Type a prompt and press Enter; Ctrl+C to quit. Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `--tokens` | 100 | Number of tokens to generate |
| `--temperature` | 0.5 | Sampling temperature (0 = greedy) |
| `--top-k` | 10 | Top-k sampling (0 = disabled) |
| `--checkpoint` | latest | Specific checkpoint to load |

Non-interactive single-shot generation:

```bash
uv run tinyfacts generate gpt_small --prompt "a star is"
```

In non-interactive mode, only the generated text is written to stdout (status messages go to stderr), making it easy to pipe the output.

## Training plots

Generate loss/perplexity/accuracy/LR plots from a run stats file:

```bash
uv run tinyfacts report models/gpt_small/runs/run_<timestamp>.jsonl
```

Plots are written to a folder named after the run file, alongside it.

## Tests

```bash
uv run pytest tests/ -v
```

## Adding a new model

1. Create `models/<name>/config.json` with hyperparameters (see `models/gpt_small/config.json` for the full key list).
2. Add either:
   - `models/<name>/model.py` exporting `build_model(config, vocab_size) -> nn.Module`, or
   - `models/<name>/model.source` containing the name of an existing model folder to reuse its architecture.
3. Train: `uv run tinyfacts train <name>`
