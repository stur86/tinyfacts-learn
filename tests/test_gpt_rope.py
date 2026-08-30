import importlib.util
import json
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = _REPO_ROOT / "models/gpt_rope/config.json"
MODEL_PATH = _REPO_ROOT / "models/gpt_rope/model.py"
VOCAB_SIZE = 1024  # approximate; exact value doesn't matter for shape tests
BATCH = 2


def _load_model_module():
    spec = importlib.util.spec_from_file_location("gpt_rope_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_model_module()


@pytest.fixture(scope="module")
def config():
    return json.loads(CONFIG_PATH.read_text())


def test_config_has_required_keys(config):
    for key in ("context_size", "n_embd", "n_heads", "n_layers", "dropout", "rope_theta"):
        assert key in config, f"Missing key '{key}' in config.json"


def test_forward_output_shape(mod, config):
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    model.eval()
    x = torch.randint(0, VOCAB_SIZE, (BATCH, config["context_size"]))
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (BATCH, config["context_size"], VOCAB_SIZE)


def test_forward_accepts_short_sequences(mod, config):
    """RoPE tables are sliced to T, so anything up to context_size runs."""
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randint(0, VOCAB_SIZE, (1, 7)))
    assert logits.shape == (1, 7, VOCAB_SIZE)


def test_forward_rejects_overlong_sequences(mod, config):
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    model.eval()
    too_long = torch.randint(0, VOCAB_SIZE, (1, config["context_size"] + 1))
    with pytest.raises(ValueError, match="exceeds context_size"):
        model(too_long)


def test_model_param_count_under_2m(mod, config):
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 2_000_000, f"Model too large: {n_params:,} params"


def test_no_learned_position_table(mod, config):
    """The whole point: position comes from rotation, not from a parameter."""
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    assert not any("pos_emb" in name for name in model.state_dict())


def test_rope_tables_are_not_checkpointed(mod, config):
    """cos/sin follow from the config, so checkpoints carry weights only."""
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    assert not any("rope" in name for name in model.state_dict())
    # ...but they are still buffers, so .to(device) moves them with the model.
    assert {"rope_cos", "rope_sin"} <= {name for name, _ in model.named_buffers()}


def test_causal_masking(mod, config):
    """Future tokens must not influence past token logits."""
    context_size = config["context_size"]
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    model.eval()

    x = torch.randint(0, VOCAB_SIZE, (1, context_size))
    with torch.no_grad():
        logits_orig = model(x)

    x_perturbed = x.clone()
    x_perturbed[0, 1:] = torch.randint(0, VOCAB_SIZE, (context_size - 1,))
    with torch.no_grad():
        logits_perturbed = model(x_perturbed)

    assert torch.allclose(logits_orig[0, 0], logits_perturbed[0, 0]), \
        "Causal mask violated: position 0 logits changed when future tokens were perturbed"


def test_rope_preserves_vector_norms(mod):
    """A rotation changes direction, never length — so it cannot rescale attention."""
    cos, sin = mod.rope_tables(64, 32, 10000.0)
    x = torch.randn(2, 4, 64, 32)
    rotated = mod.apply_rope(x, cos, sin)
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_is_identity_at_position_zero(mod):
    cos, sin = mod.rope_tables(64, 32, 10000.0)
    x = torch.randn(1, 1, 1, 32)
    assert torch.allclose(mod.apply_rope(x, cos[:1], sin[:1]), x, atol=1e-6)


def test_rope_dot_product_depends_only_on_offset(mod):
    """The defining property: q·k after rotation is a function of (pos_q - pos_k)."""
    cos, sin = mod.rope_tables(128, 32, 10000.0)
    q = torch.randn(1, 1, 1, 32)
    k = torch.randn(1, 1, 1, 32)

    def score(pos_q: int, pos_k: int) -> torch.Tensor:
        qr = mod.apply_rope(q, cos[pos_q:pos_q + 1], sin[pos_q:pos_q + 1])
        kr = mod.apply_rope(k, cos[pos_k:pos_k + 1], sin[pos_k:pos_k + 1])
        return (qr * kr).sum()

    near = score(8, 3)
    far = score(105, 100)  # same offset of 5, a hundred tokens later
    assert torch.allclose(near, far, atol=1e-4)
    # A different offset must actually give a different score, or the test above
    # would pass on an encoding that ignores position altogether.
    assert not torch.allclose(near, score(9, 3), atol=1e-4)


def test_rope_rejects_odd_head_dim(mod):
    with pytest.raises(ValueError, match="even head dimension"):
        mod.rope_tables(16, 7, 10000.0)


def test_build_model_rejects_indivisible_width(mod, config):
    bad = dict(config, n_heads=config["n_embd"] // 2 + 1)
    with pytest.raises(ValueError, match="not divisible"):
        mod.build_model(bad, vocab_size=VOCAB_SIZE)


def test_backward_pass_reaches_every_parameter(mod, config):
    model = mod.build_model(config, vocab_size=VOCAB_SIZE)
    model.train()
    x = torch.randint(0, VOCAB_SIZE, (BATCH, 32))
    logits = model(x)
    torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE), x.reshape(-1)
    ).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"No gradient reached: {missing}"


def test_matches_gpt_small_apart_from_positions(config):
    """gpt_rope is an ablation: every training key must match gpt_small."""
    gpt_small = json.loads((_REPO_ROOT / "models/gpt_small/config.json").read_text())
    for key, value in gpt_small.items():
        assert config[key] == value, f"config key {key!r} drifted from gpt_small"
    assert set(config) - set(gpt_small) == {"rope_theta"}
