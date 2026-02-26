# tests/test_generate.py
import importlib.util
import json
import torch
from pathlib import Path

from tokenizers import WordTokenizer
from generate import generate_tokens

_REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = _REPO_ROOT / "models/gpt_small/config.json"
MODEL_PATH = _REPO_ROOT / "models/gpt_small/model.py"


def _build_model():
    spec = importlib.util.spec_from_file_location("gpt_small_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads(CONFIG_PATH.read_text())
    tokenizer = WordTokenizer(ignore_case=True, digits=True)
    model = module.build_model(config, vocab_size=tokenizer.vocab_size)
    model.eval()
    return model, tokenizer


def test_generate_returns_string():
    model, tokenizer = _build_model()
    text, tokens = generate_tokens(model, tokenizer, "the world", n_tokens=10)
    assert isinstance(text, str)
    assert len(text) > 0


def test_generate_token_count():
    model, tokenizer = _build_model()
    _, tokens = generate_tokens(model, tokenizer, "the world", n_tokens=15)
    assert len(tokens) == 15


def test_generate_greedy_is_deterministic():
    """temperature=0 (greedy) must produce the same output every time."""
    model, tokenizer = _build_model()
    text1, _ = generate_tokens(model, tokenizer, "the world", n_tokens=10, temperature=0.0)
    text2, _ = generate_tokens(model, tokenizer, "the world", n_tokens=10, temperature=0.0)
    assert text1 == text2


def test_generate_empty_prompt_raises():
    model, tokenizer = _build_model()
    # An empty string tokenizes to nothing → should raise
    import pytest
    with pytest.raises(ValueError, match="empty"):
        generate_tokens(model, tokenizer, "", n_tokens=5)


def test_generate_top_k_sampling():
    """top_k > 0 should run without error and return the right number of tokens."""
    model, tokenizer = _build_model()
    _, tokens = generate_tokens(model, tokenizer, "the world", n_tokens=10, temperature=1.0, top_k=10)
    assert len(tokens) == 10
