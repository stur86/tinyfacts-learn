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
- `vocab_size` property delegates to the tokenizer

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
| `subfolders` | [...] | list of tinyfacts-gen subfolders |

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
- **Deployment** is `.github/workflows/deploy-webapp.yml` (GitHub Pages). It has no automatic
	trigger while the repo is private. CI cannot export models, so only models committed under
	`webapp/public/models` get deployed.

## Key source files

| File | Purpose |
|------|---------|
| `tokenizers.py` | `WordTokenizer` with `vocab_size` property |
| `dataset.py` | `TinyfactsDataset` — validates, tokenizes, sliding-window Dataset |
| `train.py` | Core training logic (importable); JSONL stats; cosine LR scheduler |
| `generate.py` | `generate_tokens(model, tokenizer, prompt, n_tokens, temperature, top_k)` |
| `report.py` | `generate_report(jsonl_path)` → plots folder |
| `export_onnx.py` | `export_model()` / `export_tokenizer()` → ONNX + vocabulary for the web app |
| `main.py` | Typer CLI hub: `train`, `inspect`, `report`, `generate`, `export` |
| `models/gpt_small/model.py` | GPT-small transformer + `build_model` factory |
| `models/gpt_small/config.json` | Hyperparameters + training settings |
| `webapp/src/tokenizer.ts` | TypeScript port of `WordTokenizer`, with character spans |
| `webapp/src/inference.ts` | ONNX Runtime Web session, sampling, generation loop |
| `webapp/plugins/models-manifest.ts` | Vite plugin: scans `public/models` → `manifest.json` |

## Tests

Run with `uv run pytest tests/ -v` (23 tests).

| File | Coverage |
|------|---------|
| `tests/test_dataset.py` | 9 tests — loading, shapes, dtypes, shift, OOV rejection |
| `tests/test_model.py` | 9 tests — config, forward shape, param count, causal masking, dry-run, JSONL stats |
| `tests/test_generate.py` | 5 tests — output type, token count, greedy determinism, empty-prompt error, top-k |
| `tests/test_report.py` | 5 tests — output dir, individual PNGs, overview, naming, empty-file error |
| `tests/test_export_onnx.py` | ONNX export — file layout, sidecar, graph signature, parity with PyTorch |
| `tests/test_webapp_fixtures.py` | Web app tokenizer fixtures are current and self-consistent |

Web app tests: `cd webapp && npm test` (vitest — tokenizer parity with Python, manifest scanning).

## Adding a new model

1. Create `models/<name>/` with `config.json` and either:
	- `model.py` (must export `build_model(config, vocab_size)`), or
	- `model.source` containing another model folder name to reuse its `model.py`
2. Add the model name to the `subfolders` list in its config if needed
3. Train: `uv run python main.py train <name>`
