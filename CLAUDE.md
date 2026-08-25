The goal of this repository is to test training some small scale language models (~1M parameters) on a single user GPU using a very limited synthetic dataset and token vocabulary. The vocabulary uses xkcd's "Thing Explainer" 1000 most common words dataset, plus punctuation and special tokens.

## Tokenizer

`tokenizers.py` contains `LetterTokenizer` (character-level) and `WordTokenizer` (word-level). Always use `WordTokenizer`. It has a public `vocab_size` property. Instantiate with:

```python
from tokenizers import WordTokenizer
tokenizer = WordTokenizer(ignore_case=True, digits=True)
```

## Dataset

`dataset.py` exports `TinyfactsDataset(torch.utils.data.Dataset)` and `TINYFACTS_GEN_DIR`.

- Loads `.txt` files from named subfolders of the `tinyfacts-gen/` submodule
- Validates each file against the Thing Explainer vocabulary using `tinyfacts.check_words`
- `skip_invalid=False` (default) raises `ValueError` on OOV files; `skip_invalid=True` warns and skips
- Returns sliding-window `(input_ids, target_ids)` pairs of length `context_size`
- `vocab_size` property delegates to the tokenizer; `tokens`, `n_tokens` and `n_files` expose the loaded corpus

**`stride`** (default 1) sets the gap between consecutive window starts. At the default,
every token appears in up to `context_size` windows, so one "epoch" over the Dataset is
really `context_size` passes over the corpus. Pass `stride=context_size` for
non-overlapping windows and an epoch count that means one pass.

**`split`** is `"all"` (default), `"train"` or `"val"`. Files are assigned to train or val
by a deterministic SHA-256 hash of their path relative to `tinyfacts-gen/`, controlled by
`val_fraction` (default 0.05) and `split_seed` (default 0). Splitting at the *file* level
keeps both sides representative of every subfolder and guarantees no sliding window
straddles the boundary, so val windows share no tokens with train windows.

Known OOV files in `claude_sonnet_4_5_created/`: `how_music_works.txt` ("blow"), `the_space_story.txt` ("farm", "teach", "man's", "taught"). Use `skip_invalid=True` when loading that subfolder.

## Model convention

Each model lives in its own folder under `models/`. Every model folder must contain:

- `config.json` — hyperparameters and training settings (see keys below)

And must contain either:

- `model.py` — exports `build_model(config: dict, vocab_size: int) -> nn.Module`

OR

- `model.source` — a plain text file containing the name of another model folder under `models/`.
	If present, the CLI loads `model.py` from the referenced folder, but still uses the *current* folder’s
	`config.json`, checkpoints, and runs. This allows multiple configs to share one architecture implementation.

Checkpoints are saved to `models/<name>/checkpoints/<name>_<timestamp>_step<N>.pt` as `{step, model_state_dict, optimizer_state_dict, scheduler_state_dict, config}`.

Training stats are written to `models/<name>/runs/run_<timestamp>.jsonl`, one JSON line per eval interval: `{step, epoch, loss, perplexity, accuracy, val_loss, val_perplexity, val_accuracy, lr, tokens_seen, elapsed_s, timestamp}`.

`epoch` is `tokens_seen / train_split_tokens` — one full pass over the training corpus, *not* a pass over the (heavily overlapping) window count.

## gpt_small model

`models/gpt_small/` — GPT-style transformer built on `nn.TransformerEncoderLayer` + `nn.TransformerEncoder` (pre-norm, `batch_first=True`). Tied input/output embeddings. ~930K parameters at the current 4-layer / 512-FFN config.

`config.json` keys:

| Key | Value | Notes |
|-----|-------|-------|
| `context_size` | 128 | tokens |
| `n_embd` | 128 | |
| `n_heads` | 4 | |
| `n_layers` | 4 | |
| `ffn_dim` | 512 | |
| `dropout` | 0.0 | model underfits at this scale — dropout is pure cost |
| `learning_rate` | 0.001 | AdamW peak LR |
| `min_lr` | 1e-5 | cosine decay floor |
| `warmup_steps` | 1000 | linear warmup before cosine |
| `batch_size` | 256 | |
| `max_steps` | 100000 | |
| `eval_interval` | 1000 | steps between stat flushes / val evals |
| `checkpoint_interval` | 10000 | steps between checkpoint saves |
| `val_fraction` | 0.05 | fraction of *files* held out for validation |
| `val_batches` | 20 | fixed val batches evaluated at each `eval_interval` |
| `split_seed` | 0 | seed for the deterministic file-level split |
| `subfolders` | [...] | list of tinyfacts-gen subfolders |

LR schedule: linear warmup for `warmup_steps`, then cosine decay to `min_lr` over the remaining steps (`SequentialLR` with `LinearLR` + `CosineAnnealingLR`).

## gpt_tiny model

`models/gpt_tiny/` — a half-scale sibling of gpt_small. It carries only a `model.source`
pointing at `gpt_small`, so it reuses that architecture implementation with its own
`config.json`, checkpoints and runs.

Architecture is gpt_small at half width and half depth, keeping the same 4× FFN ratio:

| Key | gpt_tiny | gpt_small |
|-----|----------|-----------|
| `n_embd` | 64 | 128 |
| `n_layers` | 2 | 4 |
| `ffn_dim` | 256 | 512 |
| `n_heads` | 4 | 4 |
| **parameters** | **169,856** | **932,864** |

Every *training* key is identical to gpt_small — `learning_rate`, `min_lr`,
`warmup_steps`, `batch_size`, `max_steps`, `eval_interval`, `checkpoint_interval`,
`dropout`, `subfolders`, and critically `split_seed`, so both models train on the same
files and validate against the same held-out set. The pair therefore isolates model size
as the only variable: any gap in `val_loss` between them is a capacity effect, not a
training-budget or data-split artefact.

## CLI — main.py

`main.py` is the Typer hub for all tools. Run with `uv run python main.py <command>`.

### train
```bash
uv run python main.py train gpt_small
uv run python main.py train gpt_small --dry-run            # 2 steps, no checkpoint saved
uv run python main.py train gpt_small --resume latest      # continue from newest checkpoint
uv run python main.py train gpt_small --resume models/gpt_small/checkpoints/....pt
```
`--resume` restores model, optimizer and LR-scheduler state and continues towards the
*current* config's `max_steps` (raise it first, or the run refuses to start). It aborts if
any architectural config key changed since the checkpoint was written.

Training batches are sampled with replacement straight from a device-resident token
tensor (`TokenSampler` in `train.py`) — no `DataLoader`, no per-item Python indexing, no
host→device copy in the loop. Validation runs in eager mode on the uncompiled module so
`torch.compile` never has to build a second graph for eval.

### inspect
```bash
uv run python main.py inspect gpt_small
uv run python main.py inspect gpt_small --checkpoint models/gpt_small/checkpoints/....pt
```
Shows config, total/trainable parameters, per-component breakdown (tied-weight aware), and a summary of all training runs.

### report
```bash
uv run python main.py report models/gpt_small/runs/run_<timestamp>.jsonl
```
Reads a training run JSONL file and writes a report folder alongside it (named after the run stem) containing: `loss.png`, `perplexity.png`, `accuracy.png`, `lr.png`, and `overview.png` (2×2 grid). Loss, perplexity and accuracy overlay the validation curve as a dashed line when the run recorded one. Each plot has a secondary epoch axis with ticks at round intervals. Core logic is in `report.py` → `generate_report(jsonl_path)`.

### generate
```bash
uv run python main.py generate gpt_small
uv run python main.py generate gpt_small --tokens 200 --temperature 0.8 --top-k 40
```
Loads the latest checkpoint (or `--checkpoint` path), then enters an interactive prompt loop. Type a prompt and press Enter; Ctrl+C to quit. Options: `--tokens` (default 100), `--temperature` (default 1.0; 0 = greedy), `--top-k` (default 0 = disabled).

## Key source files

| File | Purpose |
|------|---------|
| `tokenizers.py` | `WordTokenizer` with `vocab_size` property |
| `dataset.py` | `TinyfactsDataset` — validates, tokenizes, sliding-window Dataset; `stride` + train/val split |
| `train.py` | Core training logic (importable); `TokenSampler`, `evaluate`, resume; JSONL stats; cosine LR scheduler |
| `generate.py` | `generate_tokens(model, tokenizer, prompt, n_tokens, temperature, top_k)` |
| `report.py` | `generate_report(jsonl_path)` → plots folder |
| `main.py` | Typer CLI hub: `train` (`--resume`), `inspect`, `report`, `generate` |
| `models/gpt_small/model.py` | GPT-small transformer + `build_model` factory |
| `models/gpt_small/config.json` | Hyperparameters + training settings |

## Tests

Run with `uv run pytest tests/ -v` (31 tests).

| File | Coverage |
|------|---------|
| `tests/test_dataset.py` | 17 tests — loading, shapes, dtypes, shift, OOV rejection, `stride`, train/val split |
| `tests/test_model.py` | 9 tests — config, forward shape, param count, causal masking, dry-run, JSONL stats |
| `tests/test_generate.py` | 5 tests — output type, token count, greedy determinism, empty-prompt error, top-k |
| `tests/test_report.py` | 5 tests — output dir, individual PNGs, overview, naming, empty-file error |

## Adding a new model

1. Create `models/<name>/` with `config.json` and either:
	- `model.py` (must export `build_model(config, vocab_size)`), or
	- `model.source` containing another model folder name to reuse its `model.py`
2. Add the model name to the `subfolders` list in its config if needed
3. Train: `uv run python main.py train <name>`
