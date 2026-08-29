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

**`stride`** (default 1) sets the gap between consecutive window starts. At the default,
every token appears in up to `context_size` windows, so one "epoch" over the Dataset is
really `context_size` passes over the corpus. Pass `stride=context_size` for
non-overlapping windows and an epoch count that means one pass.

**`split`** is `"all"` (default), `"train"` or `"val"`. Rows are assigned to train or val
by a deterministic SHA-256 hash of their `id`, controlled by `val_fraction` (default 0.05)
and `split_seed` (default 0). Splitting at the *row* level keeps both sides representative
of every source and guarantees no sliding window straddles the boundary, so val windows
share no tokens with train windows.

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
| `val_fraction` | 0.05 | fraction of *rows* held out for validation |
| `val_batches` | 20 | fixed val batches evaluated at each `eval_interval` |
| `split_seed` | 0 | seed for the deterministic row-level split |
| `sources` | [] | `source` names to train on; empty means every row |
| `dataset_revision` | null | dataset commit sha to pin to; null means latest |
| `dataset_repo` | null | override the dataset repository |
| `dataset_filters` | null | extra `RecordFilter.build` arguments |

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
`dropout`, `sources`, and critically `split_seed`, so both models train on the same
rows and validate against the same held-out set. The pair therefore isolates model size
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
Reads a training run JSONL file and writes a report folder alongside it (named after
the run stem). Plots are seaborn, on a light surface with a recessive grid. Core logic
is in `report.py` → `generate_report(jsonl_path)`.

| File | Contents |
|------|----------|
| `loss.png`, `perplexity.png`, `accuracy.png` | train and val on shared axes |
| `lr.png` | the LR schedule, log y |
| `gap.png` | `val_loss - loss` against a zero line — only when the run logged val |
| `throughput.png` | tokens/second between flushes, from `tokens_seen` and `elapsed_s` |
| `overview.png` | 2×2 grid of loss, perplexity, accuracy and LR |

**Colour names the split, not the metric**: train is `#2a78d6` and val `#eb6834` in
every panel, and single-series panels (`lr`, `gap`, `throughput`) take a third hue,
`#4a3aa7`, that neither split owns. The three were checked for colour-blind separation
and contrast against the surface — don't swap them for matplotlib's default cycle,
which encodes the panel rather than the series.

Each panel ends its lines with the final value, staggered apart when train and val
converge, and carries a secondary epoch axis with ticks at round intervals (the same
axis in another unit — never a second y-scale). Panels the run has no data for are
skipped rather than written empty, so an older run without val metrics gets no
`gap.png`. When a `run_<timestamp>.meta.json` sidecar sits beside the JSONL, the
overview's subtitle names the dataset repo, revision and row counts it records.

### generate
```bash
uv run python main.py generate gpt_small
uv run python main.py generate gpt_small --tokens 200 --temperature 0.8 --top-k 40
```
Loads the latest checkpoint (or `--checkpoint` path), then enters an interactive prompt loop. Type a prompt and press Enter; Ctrl+C to quit. Options: `--tokens` (default 100), `--temperature` (default 1.0; 0 = greedy), `--top-k` (default 0 = disabled).

### export
```bash
uv run tinyfacts export gpt_small
uv run tinyfacts export gpt_small --all                 # every checkpoint
uv run tinyfacts export gpt_small --checkpoint <path>
uv run tinyfacts export gpt_small --out-dir some/dir
```
Exports a checkpoint to ONNX for the browser web app. Writes `<checkpoint-stem>.onnx`,
a `<checkpoint-stem>.json` metadata sidecar, and refreshes `tokenizer.json` — all into
`webapp/public/models/` by default. Core logic is in `export_onnx.py`.

The graph is exported with a **fixed** sequence length (`context_size`), single-file
(`external_data=False`), via the dynamo exporter at opset 20. Opset 20 is a floor, not a
preference: the exporter emits an opset-18 `Split` for mamba that an opset-17 graph cannot
express. All models are causal, so callers right-pad a short prompt and read the logits at
the last real position.

## Web app — webapp/

Client-only browser inference with ONNX Runtime Web (Vite + TypeScript, no framework).
See `webapp/README.md`. Points that matter when changing things:

- **The models folder is the source of truth.** `webapp/plugins/models-manifest.ts` scans
	`webapp/public/models` for `.onnx` files and generates `models/manifest.json` — per request
	in dev, as an emitted asset at build time. Adding a model means exporting it, nothing else.
	A model with no `.json` sidecar still loads (`hasMetadata: false`).
- **`webapp/src/tokenizer.ts` is a port of `WordTokenizer`** and must stay in step with it.
	It loads the token table from `tokenizer.json` rather than rebuilding it, so only the
	tokenize/detokenize *logic* is duplicated. After changing either side, regenerate the
	fixtures with `python tests/test_webapp_fixtures.py`; `tests/test_webapp_fixtures.py` fails
	if they are stale and `webapp/test/tokenizer.test.ts` checks the TS output against them.
- **`segment()` returns spans as well as ids**, which is what drives the red highlighting of
	words that tokenize to `<UNK>`.
- **Deployment is manual**: `scripts/deploy-webapp.sh` builds `dist/` and pushes it to the
	`gh-pages` branch (tree mirrors `dist`, one commit per deploy). There is deliberately no
	GitHub Action — exported models are not committed, so CI would have nothing to publish.
	The script refuses to deploy with an empty model folder; `--dry-run` builds and reports
	without touching any branch.

## Key source files

| File | Purpose |
|------|---------|
| `tinyfacts_learn/tokenizers.py` | `WordTokenizer` with `vocab_size` property |
| `tinyfacts_learn/hub_data.py` | Pulls the dataset chunks from the Hugging Face Hub |
| `tinyfacts_learn/dataset.py` | `TinyfactsDataset` — selects rows, tokenizes, sliding-window Dataset; `stride` + train/val split |
| `tinyfacts_learn/train.py` | Core training logic (importable); `TokenSampler`, `evaluate`, resume; JSONL stats; cosine LR scheduler |
| `tinyfacts_learn/generate.py` | `generate_tokens(model, tokenizer, prompt, n_tokens, temperature, top_k)` |
| `tinyfacts_learn/report.py` | `generate_report(jsonl_path)` → seaborn plots folder |
| `tinyfacts_learn/export_onnx.py` | `export_model()` / `export_tokenizer()` → ONNX + vocabulary for the web app |
| `tinyfacts_learn/main.py` | Typer CLI hub: `train` (`--resume`), `inspect`, `report`, `generate`, `export` |
| `models/gpt_small/model.py` | GPT-small transformer + `build_model` factory |
| `models/gpt_small/config.json` | Hyperparameters + training settings |
| `webapp/src/tokenizer.ts` | TypeScript port of `WordTokenizer`, with character spans |
| `webapp/src/inference.ts` | ONNX Runtime Web session, sampling, generation loop |
| `webapp/plugins/models-manifest.ts` | Vite plugin: scans `public/models` → `manifest.json` |
| `scripts/deploy-webapp.sh` | Manual build + publish of the web app to the `gh-pages` branch |

## Tests

Run with `uv run pytest tests/`.

The dataset tests build fake `.jsonl` chunks in `tmp_path` and monkeypatch
`hub_data.snapshot_download`, so the default run needs no network and no token.

Tests marked `network` are deselected by default (`addopts = "-m 'not network'"`).
They hit the real Hub and need a token. Run them with:

```bash
uv run pytest tests/ -m network
```

| File | Coverage |
|------|---------|
| `tests/test_dataset.py` | 24 offline + 1 network — row selection, filters, shapes, shift, id ordering, revision, validation, `stride`, train/val split, error messages |
| `tests/test_model.py` | 9 tests — config, forward shape, param count, causal masking, dry-run (network), JSONL stats (network) |
| `tests/test_trm.py` | TRM model and training dry runs (2 network) |
| `tests/test_model_source.py` | `model.source` indirection |
| `tests/test_generate.py` | 5 tests — output type, token count, greedy determinism, empty-prompt error, top-k |
| `tests/test_report.py` | 10 tests — output dir, individual PNGs, overview, naming, gap/throughput panels, skipped metrics, empty-file error |
| `tests/test_export_onnx.py` | ONNX export — file layout, sidecar, graph signature, parity with PyTorch |
| `tests/test_webapp_fixtures.py` | Web app tokenizer fixtures are current and self-consistent |

Web app tests: `cd webapp && npm test` (vitest — tokenizer parity with Python, manifest scanning).

## Adding a new model

1. Create `models/<name>/` with `config.json` and either:
	- `model.py` (must export `build_model(config, vocab_size)`), or
	- `model.source` containing another model folder name to reuse its `model.py`
2. Set `sources` in its config to narrow the training data, or leave it empty for every row
3. Train: `uv run python main.py train <name>`
