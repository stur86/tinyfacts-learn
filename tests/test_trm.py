# tests/test_trm.py
import json
import pytest
import torch
import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = _REPO_ROOT / "models/trm/config.json"
MODEL_PATH = _REPO_ROOT / "models/trm/model.py"
VOCAB_SIZE = 512
BATCH = 2


def _load_trm_module():
    spec = importlib.util.spec_from_file_location("trm_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_trm_config():
    return json.loads(CONFIG_PATH.read_text())


def test_config_file_exists():
    assert CONFIG_PATH.exists()


def test_config_has_required_keys():
    config = _load_trm_config()
    for key in ("context_size", "n_embd", "n_heads", "n_layers", "dropout",
                "n_supervision", "n_recursions", "T", "ema_decay"):
        assert key in config, f"Missing key '{key}' in config.json"


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_build_model_callable():
    mod = _load_trm_module()
    config = _load_trm_config()
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    assert model is not None


def test_forward_output_shape():
    mod = _load_trm_module()
    config = _load_trm_config()
    L = config["context_size"]
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    model.eval()
    x = torch.randint(0, VOCAB_SIZE, (BATCH, L))
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (BATCH, L, VOCAB_SIZE)


def test_context_size_attribute():
    mod = _load_trm_module()
    config = _load_trm_config()
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    assert hasattr(model, "_context_size")
    assert model._context_size == config["context_size"]


def test_embed_output_shape():
    mod = _load_trm_module()
    config = _load_trm_config()
    L = config["context_size"]
    D = config["n_embd"]
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    model.eval()
    x = torch.randint(0, VOCAB_SIZE, (BATCH, L))
    with torch.no_grad():
        emb = model.embed(x)
    assert emb.shape == (BATCH, L, D)


def test_latent_recursion_output_shapes():
    mod = _load_trm_module()
    config = _load_trm_config()
    L = config["context_size"]
    D = config["n_embd"]
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    model.eval()
    x = torch.zeros(BATCH, L, D)
    y = torch.zeros(BATCH, L, D)
    z = torch.zeros(BATCH, L, D)
    with torch.no_grad():
        y_new, z_new = model.latent_recursion(x, y, z, n=2)
    assert y_new.shape == (BATCH, L, D)
    assert z_new.shape == (BATCH, L, D)


def test_head_and_q_head_accessible():
    mod = _load_trm_module()
    config = _load_trm_config()
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    assert hasattr(model, "head")
    assert hasattr(model, "q_head")
    L, D = config["context_size"], config["n_embd"]
    h = torch.zeros(BATCH, L, D)
    with torch.no_grad():
        logits = model.head(h)
        q = model.q_head(h)
    assert logits.shape == (BATCH, L, VOCAB_SIZE)
    assert q.shape == (BATCH, L, 1)


@pytest.mark.network
def test_train_dry_run():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "tinyfacts_learn.main", "train", "trm", "--dry-run"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"Dry run failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


@pytest.mark.network
def test_train_dry_run_writes_stats():
    import subprocess, sys, json as _json
    runs_dir = _REPO_ROOT / "models" / "trm" / "runs"
    before = set(runs_dir.glob("run_*.jsonl")) if runs_dir.exists() else set()

    result = subprocess.run(
        [sys.executable, "-m", "tinyfacts_learn.main", "train", "trm", "--dry-run"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"Dry run failed:\n{result.stderr}"

    after = set(runs_dir.glob("run_*.jsonl"))
    new_files = after - before
    assert new_files, "No new JSONL stats file was created"

    stats_file = next(iter(new_files))
    lines = [l for l in stats_file.read_text().splitlines() if l.strip()]
    assert lines, "Stats file is empty"
    entry = _json.loads(lines[0])
    for field in ("loss", "perplexity", "accuracy", "step"):
        assert field in entry, f"Missing field '{field}' in stats"
