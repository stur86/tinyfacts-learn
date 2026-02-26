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

- `model.py` — exports `build_model(config: dict, vocab_size: int) -> nn.Module`
- `config.json` — hyperparameters and training settings (see keys below)

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
| `dataset.py` | `TinyfactsDataset` — validates, tokenizes, sliding-window Dataset |
| `train.py` | Core training logic (importable); JSONL stats; cosine LR scheduler |
| `generate.py` | `generate_tokens(model, tokenizer, prompt, n_tokens, temperature, top_k)` |
| `main.py` | Typer CLI hub: `train`, `inspect`, `generate` |
| `models/gpt_small/model.py` | GPT-small transformer + `build_model` factory |
| `models/gpt_small/config.json` | Hyperparameters + training settings |

## Tests

Run with `uv run pytest tests/ -v` (23 tests).

| File | Coverage |
|------|---------|
| `tests/test_dataset.py` | 9 tests — loading, shapes, dtypes, shift, OOV rejection |
| `tests/test_model.py` | 9 tests — config, forward shape, param count, causal masking, dry-run, JSONL stats |
| `tests/test_generate.py` | 5 tests — output type, token count, greedy determinism, empty-prompt error, top-k |

## Adding a new model

1. Create `models/<name>/` with `model.py` (must export `build_model(config, vocab_size)`) and `config.json`
2. Add the model name to the `subfolders` list in its config if needed
3. Train: `uv run python main.py train <name>`
