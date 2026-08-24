import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

import { modelsManifest } from "./plugins/models-manifest";

const MODELS_DIR = fileURLToPath(new URL("public/models", import.meta.url));

export default defineConfig({
  // Relative asset URLs, so the same build works at a domain root and under a
  // GitHub Pages project path (/<repo>/).
  base: "./",
  plugins: [modelsManifest(MODELS_DIR)],
  build: {
    outDir: "dist",
    // The .onnx files are already packed weights; never inline them.
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 2000,
  },
  server: {
    port: 5173,
  },
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
  },
});
