The goal of this repository is to test training some small scale language models (~1M parameters) on a single user GPU using a very limited synthetic dataset and token vocabulary. The vocabulary uses xkcd's "Thing Explainer" 1000 most common words dataset, plus punctuation and special tokens.

All modules live in the `tinyfacts_learn/` package.

## Tokenizer

`tinyfacts_learn/tokenizers.py` contains `LetterTokenizer` (character-level) and `WordTokenizer` (word-level). Always use `WordTokenizer`. It has a public `vocab_size` property. Instantiate with:

```python
from tinyfacts_learn.tokenizers import WordTokenizer
tokenizer = WordTokenizer(ignore_case=True, digits=True)
```

## Dataset

The texts used to live in a `tinyfacts-gen/` submodule as `.txt` files. They now
live in the `Stur86/tinyfacts` dataset on the Hugging Face Hub, as `.jsonl` chunks
of one row per text. The submodule is gone.

`tinyfacts_learn/hub_data.py` does the fetch:

- `load_records(repo_id, revision, token, record_filter) -> (list[DatasetRecord], sha)`
- Pulls `data/tinyfacts-*.jsonl` with `snapshot_download`, reads them with upstream's
  `DatasetStore`, sorts rows by `id` so the token stream is reproducible
- The repo is private. Token comes from `TINYFACTS_HF_TOKEN`, `HF_TOKEN` or
  `HUGGINGFACE_TOKEN`, in the environment or a `.env` file. `TINYFACTS_HF_REPO`
  overrides the repository.
- Raises `HubDataError` naming those variables when the pull is refused

**Do not write a Pydantic model for the rows.** `DatasetRecord`, `DatasetStore` and
`RecordFilter` all come from `tinyfacts.dataset` — the same classes that write the
chunks upstream, so the schema here cannot drift from the Hub's. Import them from
`tinyfacts.dataset`; that path pulls in no heavy dependencies.

Row fields: `id` (`<source>/<name>`), `text`, `title`, `source`, `model`, `provider`,
`instruction`, `instruction_model`, `tags`, `word_count`, `added_at`.

`tinyfacts_learn/dataset.py` exports `TinyfactsDataset(torch.utils.data.Dataset)`:

- Selects rows by `sources` (a list of `source` names; empty means all) and optional
  `filters` passed to `RecordFilter.build`
- Concatenates `record.text` and returns sliding-window `(input_ids, target_ids)` pairs
- Properties: `vocab_size`, `revision`, `sources`, `n_records`, `n_tokens`
- Rows are already filtered against the word list upstream, so no checking is done.
  `validate=True` checks anyway and warns on any row it drops.

As of dataset revision `63f932a`: 10,647 rows, 3.6M tokens, 14 sources, of which
`tinyfacts-llama` is 10,394. Training uses every source by default.

Note: `WordTokenizer.tokenize` discards whitespace, so texts concatenate with no
document boundary token. This is known and deliberate for now — adding one would
change `vocab_size` from 962 and invalidate existing checkpoints.

## Model convention

Each model lives in its own folder under `models/`. Every model folder must contain:

- `config.json` — hyperparameters and training settings (see keys below)

And must contain either:

- `model.py` — exports `build_model(config: dict, vocab_size: int) -> nn.Module`

OR

- `model.source` — a plain text file containing the name of another model folder under `models/`.
	If present, the CLI loads `model.py` from the referenced folder, but still uses the *current* folder’s
	`config.json`, checkpoints, and runs. This allows multiple configs to share one architecture implementation.

Checkpoints are saved to `models/<name>/checkpoints/<name>_<timestamp>_step<N>.pt` as `{step, model_state_dict, optimizer_state_dict, config}`.

Training stats are written to `models/<name>/runs/run_<timestamp>.jsonl`, one JSON line per eval interval: `{step, epoch, loss, perplexity, accuracy, lr, tokens_seen, elapsed_s, timestamp}`.

## gpt_small model

`models/gpt_small/` — GPT-style transformer built on `nn.TransformerEncoderLayer` + `nn.TransformerEncoder` (pre-norm, `batch_first=True`). Tied input/output embeddings. ~930K parameters.

`config.json` keys:

| Key | Value | Notes |
|-----|-------|-------|
| `context_size` | 128 | tokens |
| `n_embd` | 128 | |
| `n_heads` | 4 | |
| `n_layers` | 4 | |
| `ffn_dim` | 512 | |
| `dropout` | 0.1 | |
| `learning_rate` | 0.0003 | AdamW peak LR |
| `min_lr` | 1e-5 | cosine decay floor |
| `warmup_steps` | 200 | linear warmup before cosine |
| `batch_size` | 64 | |
| `max_steps` | 10000 | |
| `eval_interval` | 500 | steps between stat flushes |
| `checkpoint_interval` | 1000 | steps between checkpoint saves |
| `sources` | [] | `source` names to train on; empty means every row |
| `dataset_revision` | null | dataset commit sha to pin to; null means latest |
| `dataset_repo` | null | override the dataset repository |
| `dataset_filters` | null | extra `RecordFilter.build` arguments |

LR schedule: linear warmup for `warmup_steps`, then cosine decay to `min_lr` over the remaining steps (`SequentialLR` with `LinearLR` + `CosineAnnealingLR`).

## CLI — main.py

`main.py` is the Typer hub for all tools. Run with `uv run python main.py <command>`.

### train
```bash
uv run python main.py train gpt_small
uv run python main.py train gpt_small --dry-run   # 2 steps, no checkpoint saved
```

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
Reads a training run JSONL file and writes a report folder alongside it (named after the run stem) containing: `loss.png`, `perplexity.png`, `accuracy.png`, `lr.png`, and `overview.png` (2×2 grid). Each plot has a secondary epoch axis. Core logic is in `report.py` → `generate_report(jsonl_path)`.

### generate
```bash
uv run python main.py generate gpt_small
uv run python main.py generate gpt_small --tokens 200 --temperature 0.8 --top-k 40
```
Loads the latest checkpoint (or `--checkpoint` path), then enters an interactive prompt loop. Type a prompt and press Enter; Ctrl+C to quit. Options: `--tokens` (default 100), `--temperature` (default 1.0; 0 = greedy), `--top-k` (default 0 = disabled).

## Key source files

| File | Purpose |
|------|---------|
| `tinyfacts_learn/tokenizers.py` | `WordTokenizer` with `vocab_size` property |
| `tinyfacts_learn/hub_data.py` | Pulls the dataset chunks from the Hugging Face Hub |
| `tinyfacts_learn/dataset.py` | `TinyfactsDataset` — selects rows, tokenizes, sliding-window Dataset |
| `tinyfacts_learn/train.py` | Core training logic (importable); JSONL stats; cosine LR scheduler |
| `tinyfacts_learn/generate.py` | `generate_tokens(model, tokenizer, prompt, n_tokens, temperature, top_k)` |
| `tinyfacts_learn/report.py` | `generate_report(jsonl_path)` → plots folder |
| `tinyfacts_learn/main.py` | Typer CLI hub: `train`, `inspect`, `generate`, `report` |
| `models/gpt_small/model.py` | GPT-small transformer + `build_model` factory |
| `models/gpt_small/config.json` | Hyperparameters + training settings |

## Tests

Run with `uv run pytest tests/` (47 offline tests).

The dataset tests build fake `.jsonl` chunks in `tmp_path` and monkeypatch
`hub_data.snapshot_download`, so the default run needs no network and no token.

Tests marked `network` are deselected by default (`addopts = "-m 'not network'"`).
They hit the real Hub and need a token. Run them with:

```bash
uv run pytest tests/ -m network
```

| File | Coverage |
|------|---------|
| `tests/test_dataset.py` | 16 offline + 1 network — row selection, filters, shapes, shift, id ordering, revision, validation, error messages |
| `tests/test_model.py` | 9 tests — config, forward shape, param count, causal masking, dry-run (network), JSONL stats (network) |
| `tests/test_trm.py` | TRM model and training dry runs (2 network) |
| `tests/test_model_source.py` | `model.source` indirection |
| `tests/test_generate.py` | 5 tests — output type, token count, greedy determinism, empty-prompt error, top-k |
| `tests/test_report.py` | 5 tests — output dir, individual PNGs, overview, naming, empty-file error |

## Adding a new model

1. Create `models/<name>/` with `config.json` and either:
	- `model.py` (must export `build_model(config, vocab_size)`), or
	- `model.source` containing another model folder name to reuse its `model.py`
2. Set `sources` in its config to narrow the training data, or leave it empty for every row
3. Train: `uv run python main.py train <name>`
