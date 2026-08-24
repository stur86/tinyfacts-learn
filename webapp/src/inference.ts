/**
 * ONNX Runtime Web inference for tinyfacts-learn models.
 *
 * Models are exported with a *fixed* sequence length (`contextSize`). Since
 * every architecture in the repo is causal, a shorter prompt is right-padded
 * and the logits are read at the last real position — padding cannot influence
 * anything that came before it.
 */
import * as ort from "onnxruntime-web/wasm";

import type { ModelEntry } from "./manifest";

// GitHub Pages does not serve the cross-origin-isolation headers that
// SharedArrayBuffer (and therefore multi-threaded wasm) requires.
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";

/** Token used to pad the input up to the fixed context window. */
const PAD_ID = 0;

export interface SamplingOptions {
  temperature: number;
  /** 0 disables top-k filtering. */
  topK: number;
}

export class TinyModel {
  private constructor(
    private readonly session: ort.InferenceSession,
    readonly entry: ModelEntry,
    readonly contextSize: number,
  ) {}

  static async load(entry: ModelEntry, url: string): Promise<TinyModel> {
    const session = await ort.InferenceSession.create(url, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new TinyModel(session, entry, contextSizeOf(session, entry));
  }

  /** Logits for the position after `ids` (the last real token's row). */
  async nextLogits(ids: number[]): Promise<Float32Array> {
    const context = ids.slice(-this.contextSize);
    const input = new BigInt64Array(this.contextSize).fill(BigInt(PAD_ID));
    for (let i = 0; i < context.length; i++) {
      input[i] = BigInt(context[i]);
    }

    const tensor = new ort.Tensor("int64", input, [1, this.contextSize]);
    const outputs = await this.session.run({ input_ids: tensor });
    const logits = outputs[this.session.outputNames[0]];
    const data = logits.data as Float32Array;

    const vocabSize = Number(logits.dims[logits.dims.length - 1]);
    const row = context.length - 1;
    return data.slice(row * vocabSize, (row + 1) * vocabSize);
  }

  release(): Promise<void> {
    return this.session.release();
  }
}

/**
 * The sequence length the graph was exported with.
 *
 * Taken from the model's own input shape where possible, so a manifest entry
 * with a missing or stale sidecar cannot silently produce wrong-length inputs.
 */
function contextSizeOf(session: ort.InferenceSession, entry: ModelEntry): number {
  const input = session.inputMetadata?.[0];
  if (input?.isTensor) {
    const dim = input.shape[input.shape.length - 1];
    if (typeof dim === "number" && dim > 0) return dim;
  }
  if (entry.contextSize > 0) return entry.contextSize;
  throw new Error(`Cannot determine the context size of model '${entry.id}'.`);
}

/** Pick one token id from a logits row. Mirrors generate.py's _sample_token. */
export function sampleToken(logits: Float32Array, { temperature, topK }: SamplingOptions): number {
  if (temperature === 0) {
    let best = 0;
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > logits[best]) best = i;
    }
    return best;
  }

  const scaled = Float32Array.from(logits, (value) => value / temperature);

  let cutoff = -Infinity;
  if (topK > 0 && topK < scaled.length) {
    const sorted = Array.from(scaled).sort((a, b) => b - a);
    cutoff = sorted[topK - 1];
  }

  let max = -Infinity;
  for (let i = 0; i < scaled.length; i++) {
    if (scaled[i] >= cutoff && scaled[i] > max) max = scaled[i];
  }

  const probs = new Float64Array(scaled.length);
  let total = 0;
  for (let i = 0; i < scaled.length; i++) {
    if (scaled[i] < cutoff) continue;
    const p = Math.exp(scaled[i] - max);
    probs[i] = p;
    total += p;
  }

  let target = Math.random() * total;
  for (let i = 0; i < probs.length; i++) {
    target -= probs[i];
    if (target <= 0 && probs[i] > 0) return i;
  }
  // Floating-point slack: fall back to the last candidate.
  for (let i = probs.length - 1; i >= 0; i--) {
    if (probs[i] > 0) return i;
  }
  return 0;
}

export interface GenerateOptions extends SamplingOptions {
  nTokens: number;
  /** Called after each token with every id generated so far. */
  onToken?: (ids: number[]) => void;
  /** Return true to stop generating early. */
  shouldStop?: () => boolean;
}

/** Generate `nTokens` continuation tokens for a prompt already tokenized to ids. */
export async function generate(
  model: TinyModel,
  promptIds: number[],
  options: GenerateOptions,
): Promise<number[]> {
  const ids = [...promptIds];
  const generated: number[] = [];

  for (let i = 0; i < options.nTokens; i++) {
    if (options.shouldStop?.()) break;
    const logits = await model.nextLogits(ids);
    const next = sampleToken(logits, options);
    ids.push(next);
    generated.push(next);
    options.onToken?.(generated);
    // Hand the main thread back to the browser so the UI can paint.
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  return generated;
}
