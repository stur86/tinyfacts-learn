# tests/test_export_onnx.py
import json

import numpy as np
import pytest
import torch

from tinyfacts_learn.export_onnx import (
    WEBAPP_MODELS_DIR,
    export_model,
    export_tokenizer,
    latest_checkpoint,
)
from tinyfacts_learn.tokenizers import WordTokenizer
from tinyfacts_learn.train import load_config, load_model_module

# gpt_tiny is the cheapest architecture to export; the others share the code path.
MODEL_NAME = "gpt_tiny"


@pytest.fixture(scope="module")
def tokenizer():
    return WordTokenizer(ignore_case=True, digits=True)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory, tokenizer):
    """An untrained checkpoint — the export path does not care about the weights."""
    config = load_config(MODEL_NAME)
    module = load_model_module(MODEL_NAME)
    model = module.build_model(config, vocab_size=tokenizer.vocab_size)
    path = tmp_path_factory.mktemp("checkpoints") / f"{MODEL_NAME}_20260101_000000_step10.pt"
    torch.save(
        {
            "step": 10,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "config": config,
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def exported(tmp_path_factory, checkpoint):
    out_dir = tmp_path_factory.mktemp("export")
    return export_model(MODEL_NAME, checkpoint=checkpoint, out_dir=out_dir)


def test_export_writes_a_single_self_contained_file(exported):
    onnx_path, _ = exported
    assert onnx_path.exists()
    assert onnx_path.suffix == ".onnx"
    # A sidecar weights file would break the web app's one-fetch-per-model loading.
    assert not onnx_path.with_suffix(".onnx.data").exists()
    assert onnx_path.stat().st_size > 0


def test_export_names_files_after_the_checkpoint(exported, checkpoint):
    onnx_path, meta_path = exported
    assert onnx_path.stem == checkpoint.stem
    assert meta_path.stem == checkpoint.stem


def test_metadata_sidecar_describes_the_model(exported, checkpoint, tokenizer):
    onnx_path, meta_path = exported
    meta = json.loads(meta_path.read_text())

    assert meta["id"] == checkpoint.stem
    assert meta["model"] == MODEL_NAME
    assert meta["file"] == onnx_path.name
    assert meta["checkpoint"] == checkpoint.name
    assert meta["step"] == 10
    assert meta["contextSize"] == load_config(MODEL_NAME)["context_size"]
    assert meta["vocabSize"] == tokenizer.vocab_size
    assert meta["params"] > 0
    assert meta["sizeBytes"] == onnx_path.stat().st_size


def test_exported_graph_has_the_expected_signature(exported, tokenizer):
    ort = pytest.importorskip("onnxruntime")
    onnx_path, _ = exported
    context_size = load_config(MODEL_NAME)["context_size"]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (input_meta,) = session.get_inputs()
    assert input_meta.name == "input_ids"
    assert input_meta.shape == [1, context_size]
    assert input_meta.type == "tensor(int64)"

    (output_meta,) = session.get_outputs()
    assert output_meta.name == "logits"
    assert output_meta.shape == [1, context_size, tokenizer.vocab_size]


def test_exported_graph_matches_pytorch(exported, checkpoint, tokenizer):
    """The browser must see the same logits the CLI would produce."""
    ort = pytest.importorskip("onnxruntime")
    onnx_path, _ = exported
    config = load_config(MODEL_NAME)
    context_size = config["context_size"]

    module = load_model_module(MODEL_NAME)
    model = module.build_model(config, vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"])
    model.eval()

    ids = tokenizer.tokenize("the sun is a big ball of hot air")
    padded = np.zeros((1, context_size), dtype=np.int64)
    padded[0, : len(ids)] = ids

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"input_ids": padded})[0]

    with torch.no_grad():
        torch_logits = model(torch.tensor([ids], dtype=torch.long)).numpy()

    # Right-padding must not disturb the last real position: these models are causal.
    last = len(ids) - 1
    np.testing.assert_allclose(onnx_logits[0, last], torch_logits[0, last], atol=1e-4)


def test_export_rejects_a_missing_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="Checkpoint not found"):
        export_model(MODEL_NAME, checkpoint=tmp_path / "nope.pt", out_dir=tmp_path)


def test_latest_checkpoint_reports_when_there_are_none(monkeypatch, tmp_path):
    monkeypatch.setattr("tinyfacts_learn.export_onnx.MODELS_DIR", tmp_path)
    with pytest.raises(ValueError, match="No checkpoints found"):
        latest_checkpoint(MODEL_NAME)


def test_export_tokenizer_writes_the_vocabulary(tmp_path, tokenizer):
    path = export_tokenizer(tmp_path)
    payload = json.loads(path.read_text())

    assert path.name == "tokenizer.json"
    assert payload["vocabSize"] == tokenizer.vocab_size
    assert payload["tokens"][payload["unkId"]] == "<UNK>"
    assert payload["wordTokens"]["sun"] == tokenizer.tokenize("sun")
    assert payload["formLookup"]["|sun"] == "sun"


def test_default_output_dir_is_the_webapp_models_folder():
    assert WEBAPP_MODELS_DIR.parts[-3:] == ("webapp", "public", "models")
