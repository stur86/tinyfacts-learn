/**
 * Vite plugin: build `models/manifest.json` by scanning the models folder.
 *
 * The app serves whatever `.onnx` files happen to be in `public/models`, so the
 * manifest is derived from the folder rather than hand-maintained. Each model
 * may have a `<stem>.json` sidecar (written by `main.py export`) carrying the
 * step count, parameter count and context size; a model without one still
 * shows up, just with less to say about it.
 *
 * In dev the manifest is generated per request, so exporting a new model shows
 * up on reload. In build it is emitted as an asset next to the copied files.
 */
import fs from "node:fs";
import path from "node:path";
import type { Plugin } from "vite";

export interface ModelEntry {
  id: string;
  model: string;
  file: string;
  contextSize: number;
  sizeBytes: number;
  step: number | null;
  params: number | null;
  vocabSize: number | null;
  exportedAt: string | null;
  hasMetadata: boolean;
}

export interface Manifest {
  generatedAt: string;
  models: ModelEntry[];
}

const DEFAULT_CONTEXT_SIZE = 128;

function readSidecar(sidecarPath: string): Record<string, unknown> | null {
  if (!fs.existsSync(sidecarPath)) return null;
  try {
    return JSON.parse(fs.readFileSync(sidecarPath, "utf8")) as Record<string, unknown>;
  } catch (error) {
    console.warn(`[models-manifest] ignoring unreadable sidecar ${sidecarPath}: ${error}`);
    return null;
  }
}

function numberOr(value: unknown, fallback: number | null): number | null {
  return typeof value === "number" ? value : fallback;
}

export function buildManifest(modelsDir: string): Manifest {
  const models: ModelEntry[] = [];

  if (fs.existsSync(modelsDir)) {
    const onnxFiles = fs
      .readdirSync(modelsDir)
      .filter((name) => name.endsWith(".onnx"))
      .sort();

    for (const file of onnxFiles) {
      const stem = file.slice(0, -".onnx".length);
      const meta = readSidecar(path.join(modelsDir, `${stem}.json`));
      models.push({
        id: typeof meta?.id === "string" ? meta.id : stem,
        model: typeof meta?.model === "string" ? meta.model : stem,
        file,
        contextSize: numberOr(meta?.contextSize, DEFAULT_CONTEXT_SIZE) as number,
        sizeBytes: fs.statSync(path.join(modelsDir, file)).size,
        step: numberOr(meta?.step, null),
        params: numberOr(meta?.params, null),
        vocabSize: numberOr(meta?.vocabSize, null),
        exportedAt: typeof meta?.exportedAt === "string" ? meta.exportedAt : null,
        hasMetadata: meta !== null,
      });
    }
  }

  return { generatedAt: new Date().toISOString(), models };
}

export function modelsManifest(modelsDir: string): Plugin {
  const manifestRoute = "/models/manifest.json";

  return {
    name: "tinyfacts-models-manifest",

    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? "").split("?")[0];
        if (!url.endsWith(manifestRoute)) return next();
        res.setHeader("Content-Type", "application/json");
        res.setHeader("Cache-Control", "no-store");
        res.end(JSON.stringify(buildManifest(modelsDir), null, 2));
      });
    },

    generateBundle() {
      const manifest = buildManifest(modelsDir);
      if (manifest.models.length === 0) {
        this.warn(
          `no .onnx models found in ${modelsDir} — the app will build, but its ` +
            "model dropdown will be empty. Run: uv run python -m tinyfacts_learn.main export <model>",
        );
      }
      this.emitFile({
        type: "asset",
        fileName: "models/manifest.json",
        source: JSON.stringify(manifest, null, 2),
      });
    },
  };
}
