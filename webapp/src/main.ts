/** Wiring for the tinyfacts browser playground. */
import "./style.css";

import { generate, TinyModel } from "./inference";
import {
  describe,
  loadManifest,
  modelUrl,
  TOKENIZER_URL,
  type ModelEntry,
} from "./manifest";
import { WordTokenizer, type Segment } from "./tokenizer";

const els = {
  modelSelect: byId<HTMLSelectElement>("model-select"),
  modelDetails: byId<HTMLParagraphElement>("model-details"),
  prompt: byId<HTMLTextAreaElement>("prompt"),
  highlights: byId<HTMLDivElement>("highlights"),
  vocabNote: byId<HTMLParagraphElement>("vocab-note"),
  complete: byId<HTMLButtonElement>("complete"),
  stop: byId<HTMLButtonElement>("stop"),
  status: byId<HTMLSpanElement>("status"),
  output: byId<HTMLParagraphElement>("output"),
};

let tokenizer: WordTokenizer | null = null;
let entries: ModelEntry[] = [];
let model: TinyModel | null = null;
let loading: Promise<TinyModel> | null = null;
let stopRequested = false;

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
}

function setStatus(message: string, kind: "" | "busy" | "error" = ""): void {
  els.status.textContent = message;
  els.status.dataset.kind = kind;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ── prompt highlighting ───────────────────────────────────────────────────────

function renderHighlights(segments: Segment[], text: string): void {
  const html = segments
    .map((segment) =>
      segment.unknown
        ? `<mark class="unk">${escapeHtml(segment.text)}</mark>`
        : escapeHtml(segment.text),
    )
    .join("");
  // A trailing newline would otherwise not occupy a line in the overlay, so the
  // highlights would drift out of step with the textarea's own layout.
  els.highlights.innerHTML = text.endsWith("\n") ? `${html} ` : html;
  els.highlights.scrollTop = els.prompt.scrollTop;
  els.highlights.scrollLeft = els.prompt.scrollLeft;
}

function refreshPrompt(): void {
  const text = els.prompt.value;
  if (!tokenizer) {
    els.highlights.textContent = text;
    return;
  }

  const segments = tokenizer.segment(text);
  renderHighlights(segments, text);

  const unknown = segments.filter((s) => s.unknown);
  const tokenCount = segments.reduce((total, s) => total + s.ids.length, 0);

  if (unknown.length === 0) {
    els.vocabNote.className = "vocab-note";
    els.vocabNote.textContent = tokenCount
      ? `${tokenCount} token${tokenCount === 1 ? "" : "s"}, all in vocabulary.`
      : "";
  } else {
    const shown = [...new Set(unknown.map((s) => s.text))].slice(0, 6);
    els.vocabNote.className = "vocab-note has-unknown";
    els.vocabNote.textContent =
      `${tokenCount} tokens · ${unknown.length} outside the vocabulary ` +
      `(${shown.join(", ")}${shown.length < unknown.length ? ", …" : ""}) — ` +
      "the model reads each of these as <UNK>.";
  }

  els.complete.disabled = tokenCount === 0 || els.modelSelect.disabled;
}

// ── model loading ─────────────────────────────────────────────────────────────

function describeEntry(entry: ModelEntry): string {
  const bits: string[] = [`architecture: ${entry.model}`];
  if (entry.params !== null) bits.push(`${entry.params.toLocaleString()} parameters`);
  bits.push(`context ${entry.contextSize} tokens`);
  if (entry.vocabSize !== null) bits.push(`vocabulary ${entry.vocabSize}`);
  if (entry.exportedAt) bits.push(`exported ${entry.exportedAt.slice(0, 10)}`);
  if (!entry.hasMetadata) bits.push("no metadata sidecar found");
  return bits.join(" · ");
}

async function selectModel(entry: ModelEntry): Promise<TinyModel> {
  els.modelDetails.textContent = describeEntry(entry);

  if (model?.entry.id === entry.id) return model;

  setStatus(`Loading ${entry.id}…`, "busy");
  const previous = model;
  model = null;

  const pending = TinyModel.load(entry, modelUrl(entry));
  loading = pending;
  try {
    const loaded = await pending;
    if (loading !== pending) return loaded; // superseded by a newer selection
    model = loaded;
    await previous?.release();
    setStatus("Model ready.");
    return loaded;
  } catch (error) {
    setStatus(`Could not load ${entry.file}: ${errorText(error)}`, "error");
    throw error;
  } finally {
    if (loading === pending) loading = null;
  }
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

// ── generation ────────────────────────────────────────────────────────────────

function readNumber(input: HTMLInputElement, fallback: number): number {
  const value = Number(input.value);
  return Number.isFinite(value) ? value : fallback;
}

function renderOutput(prompt: string, continuation: string): void {
  els.output.innerHTML =
    `<span class="prompt-echo">${escapeHtml(prompt)}</span>` +
    (continuation ? ` <span class="generated">${escapeHtml(continuation)}</span>` : "");
}

async function complete(): Promise<void> {
  if (!tokenizer) return;

  const entry = entries.find((e) => e.id === els.modelSelect.value);
  if (!entry) {
    setStatus("No model selected.", "error");
    return;
  }

  const prompt = els.prompt.value.trim();
  const promptIds = tokenizer.tokenize(prompt);
  if (promptIds.length === 0) {
    setStatus("Type something first.", "error");
    return;
  }

  let active: TinyModel;
  try {
    active = await selectModel(entry);
  } catch {
    return;
  }

  stopRequested = false;
  els.complete.disabled = true;
  els.stop.hidden = false;
  setStatus("Generating…", "busy");
  renderOutput(prompt, "");

  const started = performance.now();
  try {
    const ids = await generate(active, promptIds, {
      nTokens: Math.max(1, Math.round(readNumber(byId<HTMLInputElement>("tokens"), 60))),
      temperature: Math.max(0, readNumber(byId<HTMLInputElement>("temperature"), 0.5)),
      topK: Math.max(0, Math.round(readNumber(byId<HTMLInputElement>("top-k"), 10))),
      onToken: (generated) => renderOutput(prompt, tokenizer!.detokenize(generated)),
      shouldStop: () => stopRequested,
    });
    const seconds = (performance.now() - started) / 1000;
    const rate = ids.length / Math.max(seconds, 1e-6);
    setStatus(
      `${ids.length} token${ids.length === 1 ? "" : "s"} in ${seconds.toFixed(1)}s ` +
        `(${rate.toFixed(1)}/s)${stopRequested ? " — stopped" : ""}.`,
    );
  } catch (error) {
    setStatus(`Generation failed: ${errorText(error)}`, "error");
  } finally {
    els.stop.hidden = true;
    els.complete.disabled = false;
  }
}

// ── startup ───────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  els.prompt.addEventListener("input", refreshPrompt);
  els.prompt.addEventListener("scroll", () => {
    els.highlights.scrollTop = els.prompt.scrollTop;
    els.highlights.scrollLeft = els.prompt.scrollLeft;
  });
  els.complete.addEventListener("click", () => void complete());
  els.stop.addEventListener("click", () => {
    stopRequested = true;
    setStatus("Stopping…", "busy");
  });
  els.prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      void complete();
    }
  });

  setStatus("Loading tokenizer…", "busy");
  try {
    tokenizer = await WordTokenizer.load(TOKENIZER_URL);
  } catch (error) {
    setStatus(
      `${errorText(error)} — run "python -m tinyfacts_learn.main export <model>" to generate it.`,
      "error",
    );
    return;
  }

  try {
    const manifest = await loadManifest();
    entries = manifest.models;
  } catch (error) {
    setStatus(errorText(error), "error");
    return;
  }

  if (entries.length === 0) {
    els.modelSelect.innerHTML = "<option>No models found</option>";
    setStatus(
      'No models in public/models — run "python -m tinyfacts_learn.main export <model>" first.',
      "error",
    );
    refreshPrompt();
    return;
  }

  els.modelSelect.innerHTML = "";
  for (const entry of entries) {
    const option = document.createElement("option");
    option.value = entry.id;
    option.textContent = describe(entry);
    els.modelSelect.append(option);
  }
  els.modelSelect.disabled = false;
  els.modelSelect.addEventListener("change", () => {
    const entry = entries.find((e) => e.id === els.modelSelect.value);
    if (entry) void selectModel(entry).catch(() => undefined);
  });

  els.modelDetails.textContent = describeEntry(entries[0]);
  setStatus("Ready. The model loads on first use.");
  refreshPrompt();
}

void main();
