# tinyfacts playground

A client-only web app that runs tinyfacts-learn models in the browser with
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). There is no
server component: the page loads an `.onnx` file and the tokenizer vocabulary as
static assets and does everything else in WebAssembly.

## Getting models in

The app serves **every `.onnx` file it finds in `public/models`**. Put one there
with the export command from the repository root:

```bash
uv run python -m tinyfacts_learn.main export gpt_small          # latest checkpoint
uv run python -m tinyfacts_learn.main export gpt_small --all    # every checkpoint
```

That writes three kinds of file into `public/models`:

| File | Purpose |
|------|---------|
| `<checkpoint>.onnx` | the graph, weights included, fixed sequence length |
| `<checkpoint>.json` | metadata sidecar: step, parameter count, context size |
| `tokenizer.json` | the `WordTokenizer` vocabulary, shared by all models |

`plugins/models-manifest.ts` scans the folder and generates
`models/manifest.json` — in dev on every request, at build time as an emitted
asset — so a newly exported model needs no code change to appear in the
dropdown. A model without its `.json` sidecar still loads; it just shows less
detail.

## Running it

```bash
npm install
npm run dev       # http://localhost:5173
npm test          # tokenizer + manifest tests
npm run build     # typecheck, then build to dist/
npm run preview   # serve dist/
```

`npm run build` produces a fully static `dist/` (about 13 MB of that is the ONNX
Runtime WebAssembly binary, which compresses to ~3.5 MB over the wire).

## How it works

**Tokenizing.** `src/tokenizer.ts` is a port of
`tinyfacts_learn/tokenizers.py`'s `WordTokenizer`. It does not rebuild the
vocabulary — it loads the exact token table the model was trained against from
`tokenizer.json`. `segment()` returns the token ids *plus* the span of text each
one came from, which is what drives the red highlighting: any word the
vocabulary does not contain becomes `<UNK>`, and the model cannot tell one
`<UNK>` from another.

The port is kept honest by fixtures: `python tests/test_webapp_fixtures.py`
regenerates `test/fixtures/` from the Python tokenizer, a pytest fails if they
go stale, and `test/tokenizer.test.ts` checks the TypeScript output against them
case by case.

**Inference.** Models are exported with a *fixed* sequence length. Every
architecture in this repo is causal, so `src/inference.ts` right-pads a short
prompt to that length and reads the logits at the last real token — padding
cannot influence anything before it. Sampling (temperature, top-k) mirrors
`generate.py`. Generation yields to the event loop between tokens so the page
stays responsive and the completion streams in.

Threads are disabled (`ort.env.wasm.numThreads = 1`) because GitHub Pages does
not send the cross-origin-isolation headers that `SharedArrayBuffer` requires.

## Deploying

Deployment is manual, by design: exported `.onnx` files are not committed, so a
CI runner would have nothing to publish. `scripts/deploy-webapp.sh` runs from a
machine that has the models:

```bash
uv run tinyfacts export gpt_small        # -> webapp/public/models/
./scripts/deploy-webapp.sh
```

It refuses to publish an app with no models in `public/models`, builds
`dist/`, and pushes it as one commit on the `gh-pages` branch, whose tree
mirrors `dist` exactly.

| Option | Effect |
|--------|--------|
| `--dry-run` | build and list what would be published; touch no branch |
| `--skip-build` | publish the existing `dist/` as-is |
| `--branch` / `--remote` | publish somewhere other than `origin`/`gh-pages` |
| `--allow-no-models` | publish anyway with an empty model folder |

One-time setup on GitHub: Settings → Pages → Build and deployment → Source:
"Deploy from a branch", branch `gh-pages`, folder `/ (root)`. The `gh-pages`
branch holds build output only — never edit it by hand or merge it anywhere.
