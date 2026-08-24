import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { buildManifest } from "../plugins/models-manifest";

let dir: string;

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "tinyfacts-models-"));
});

afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

function writeModel(stem: string, meta?: Record<string, unknown>): void {
  fs.writeFileSync(path.join(dir, `${stem}.onnx`), "not-a-real-graph");
  if (meta) {
    fs.writeFileSync(path.join(dir, `${stem}.json`), JSON.stringify(meta));
  }
}

describe("buildManifest", () => {
  it("returns nothing for a missing folder", () => {
    expect(buildManifest(path.join(dir, "nope")).models).toEqual([]);
  });

  it("returns nothing for an empty folder", () => {
    expect(buildManifest(dir).models).toEqual([]);
  });

  it("picks up every .onnx in the folder, sorted", () => {
    writeModel("z_model_step100");
    writeModel("a_model_step100");
    expect(buildManifest(dir).models.map((m) => m.id)).toEqual([
      "a_model_step100",
      "z_model_step100",
    ]);
  });

  it("reads the sidecar metadata when present", () => {
    writeModel("gpt_small_20260101_step2000", {
      id: "gpt_small_20260101_step2000",
      model: "gpt_small",
      contextSize: 64,
      step: 2000,
      params: 930000,
      vocabSize: 962,
      exportedAt: "2026-01-01T00:00:00+00:00",
    });

    const [entry] = buildManifest(dir).models;
    expect(entry).toMatchObject({
      model: "gpt_small",
      file: "gpt_small_20260101_step2000.onnx",
      contextSize: 64,
      step: 2000,
      params: 930000,
      vocabSize: 962,
      hasMetadata: true,
    });
    expect(entry.sizeBytes).toBeGreaterThan(0);
  });

  it("still lists a model whose sidecar is missing or unreadable", () => {
    writeModel("orphan_step10");
    fs.writeFileSync(path.join(dir, "broken_step10.onnx"), "x");
    fs.writeFileSync(path.join(dir, "broken_step10.json"), "{not json");

    const models = buildManifest(dir).models;
    expect(models.map((m) => m.id)).toEqual(["broken_step10", "orphan_step10"]);
    for (const model of models) {
      expect(model.hasMetadata).toBe(false);
      expect(model.step).toBeNull();
      expect(model.contextSize).toBe(128);
    }
  });

  it("ignores files that are not models", () => {
    writeModel("real_step10");
    fs.writeFileSync(path.join(dir, "tokenizer.json"), "{}");
    fs.writeFileSync(path.join(dir, ".gitkeep"), "");
    expect(buildManifest(dir).models.map((m) => m.id)).toEqual(["real_step10"]);
  });
});
