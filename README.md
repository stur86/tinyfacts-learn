# tinyfacts-learn

Train and run small language models (~1M parameters) on a restricted vocabulary derived from xkcd's [Thing Explainer](https://xkcd.com/thing-explainer/) word list. The text data lives in the [`Stur86/tinyfacts`](https://huggingface.co/datasets/Stur86/tinyfacts) dataset on the Hugging Face Hub.

## Setup

```bash
uv sync
```

This installs all dependencies including PyTorch (CUDA 12.6 build). For CPU-only or a different CUDA version, edit `pyproject.toml` accordingly before syncing.

The word list itself comes from the `tinyfacts` package, pinned to a commit of
[tinyfacts-gen](https://github.com/stur86/tinyfacts-gen) in `[tool.uv.sources]`.

### Dataset access

The texts are pulled from the Hub the first time you train, and cached after that.
The dataset repository is private, so a token with read rights is needed. Put it in
a `.env` file at the repository root, or in the environment:

```bash
HF_TOKEN=hf_...
```

`TINYFACTS_HF_TOKEN` and `HUGGINGFACE_TOKEN` work too — the same names the
`tinyfacts` CLI uses, so one token serves both. Set `TINYFACTS_HF_REPO` to read
the rows from a different dataset repository.

## Available models

| Model | Description |
|-------|-------------|
| `gpt_small` | GPT-style transformer, ~930K parameters (n_embd=128, 2 layers) |
| `gpt_tiny` | Same architecture as gpt_small, much smaller (n_embd=32, 1 layer) |
| `gpt_rope` | gpt_small with rotary position embeddings (RoPE) in place of the learned position table, ~916K parameters |
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

## Browser playground

Export a checkpoint to ONNX and run it in the browser — no server, no Python:

```bash
uv run tinyfacts export gpt_small          # latest checkpoint
uv run tinyfacts export gpt_small --all    # every checkpoint
```

This writes `<checkpoint>.onnx`, a `<checkpoint>.json` metadata sidecar, and the
shared `tokenizer.json` into `webapp/public/models/`. The web app serves every
`.onnx` it finds there:

```bash
cd webapp
npm install
npm run dev
```

Pick a model from the dropdown, type the start of a sentence and let the model
finish it. Words outside the 1000-word vocabulary are highlighted in red — the
model reads all of them as the same `<UNK>` token.

Publishing to GitHub Pages is a manual step, since the exported models are not
committed and CI would have nothing to deploy:

```bash
./scripts/deploy-webapp.sh --dry-run   # build and show what would be published
./scripts/deploy-webapp.sh             # build and push it to the gh-pages branch
```

See [`webapp/README.md`](webapp/README.md) for details.

## Tests

```bash
uv run pytest tests/ -v          # Python
cd webapp && npm test            # web app
```

## Choosing the training data

`config.json` says which rows a model trains on:

| Key | Meaning |
|-----|---------|
| `sources` | Which runs to use, by `source` name. Empty (or absent) uses every row. |
| `dataset_revision` | A dataset commit sha to pin the run to. `null` uses the latest. |
| `dataset_repo` | A different dataset repository. `null` uses `Stur86/tinyfacts`. |
| `dataset_filters` | Extra row filters: `min_words`, `max_words`, `model`, `tag`, `has_instruction`, and `id`/`title`/`text`/`instruction` regular expressions. |

Every training run writes `models/<name>/runs/run_<timestamp>.meta.json` beside its
stats, naming the dataset revision, sources and row count it actually saw. The same
record goes into each checkpoint under the `dataset` key. The dataset on the Hub grows
as texts are added, so this is what says which version of it a run was trained on.

To repeat an earlier run exactly, copy its `revision` into `dataset_revision`.

## Adding a new model

1. Create `models/<name>/config.json` with hyperparameters (see `models/gpt_small/config.json` for the full key list).
2. Add either:
   - `models/<name>/model.py` exporting `build_model(config, vocab_size) -> nn.Module`, or
   - `models/<name>/model.source` containing the name of an existing model folder to reuse its architecture.
3. Train: `uv run tinyfacts train <name>`
