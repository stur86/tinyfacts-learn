# tests/test_model.py
import json
import torch
import importlib.util
from pathlib import Path

CONFIG_PATH = Path("models/gpt_small/config.json")
MODEL_PATH = Path("models/gpt_small/model.py")
VOCAB_SIZE = 1024  # approximate; exact value doesn't matter for shape tests
BATCH = 2


def _load_model_module():
    spec = importlib.util.spec_from_file_location("gpt_small_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_file_exists():
    assert CONFIG_PATH.exists()


def test_config_has_required_keys():
    config = json.loads(CONFIG_PATH.read_text())
    for key in ("context_size", "n_embd", "n_heads", "n_layers", "dropout"):
        assert key in config, f"Missing key '{key}' in config.json"


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_build_model_callable():
    mod = _load_model_module()
    config = json.loads(CONFIG_PATH.read_text())
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    assert model is not None


def test_forward_output_shape():
    mod = _load_model_module()
    config = json.loads(CONFIG_PATH.read_text())
    context_size = config["context_size"]
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    model.eval()
    x = torch.randint(0, VOCAB_SIZE, (BATCH, context_size))
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (BATCH, context_size, VOCAB_SIZE)


def test_model_param_count_under_2m():
    mod = _load_model_module()
    config = json.loads(CONFIG_PATH.read_text())
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 2_000_000, f"Model too large: {n_params:,} params"
