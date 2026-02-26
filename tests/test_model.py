# tests/test_model.py
import json
import torch
import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = _REPO_ROOT / "models/gpt_small/config.json"
MODEL_PATH = _REPO_ROOT / "models/gpt_small/model.py"
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


def test_causal_masking():
    """Future tokens must not influence past token logits."""
    mod = _load_model_module()
    config = json.loads(CONFIG_PATH.read_text())
    context_size = config["context_size"]
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    model.eval()

    x = torch.randint(0, VOCAB_SIZE, (1, context_size))
    with torch.no_grad():
        logits_orig = model(x)

    # Perturb all tokens from position 1 onwards
    x_perturbed = x.clone()
    x_perturbed[0, 1:] = torch.randint(0, VOCAB_SIZE, (context_size - 1,))
    with torch.no_grad():
        logits_perturbed = model(x_perturbed)

    # Position 0 logits must be identical (it only attends to itself)
    assert torch.allclose(logits_orig[0, 0], logits_perturbed[0, 0]), \
        "Causal mask violated: position 0 logits changed when future tokens were perturbed"


def test_train_dry_run():
    """Training script must complete a dry run without error."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "train.py", "gpt_small", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"Dry run failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
