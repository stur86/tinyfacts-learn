# tests/test_dataset.py
import pytest
import torch
from pathlib import Path
from dataset import TinyfactsDataset, TINYFACTS_GEN_DIR

CONTEXT_SIZE = 8  # small for tests

# Pick a subfolder we know exists and has valid content
VALID_SUBFOLDER = "manually_created"


def test_tinyfacts_gen_dir_exists():
    assert TINYFACTS_GEN_DIR.exists(), f"tinyfacts-gen dir not found at {TINYFACTS_GEN_DIR}"


def test_dataset_loads_valid_subfolder():
    ds = TinyfactsDataset(subfolders=[VALID_SUBFOLDER], context_size=CONTEXT_SIZE)
    assert len(ds) > 0


def test_dataset_item_shapes():
    ds = TinyfactsDataset(subfolders=[VALID_SUBFOLDER], context_size=CONTEXT_SIZE)
    x, y = ds[0]
    assert x.shape == (CONTEXT_SIZE,)
    assert y.shape == (CONTEXT_SIZE,)


def test_dataset_target_is_shifted():
    ds = TinyfactsDataset(subfolders=[VALID_SUBFOLDER], context_size=CONTEXT_SIZE)
    x, y = ds[0]
    # y[i] == x[i+1] for all but last position
    assert torch.equal(x[1:], y[:-1])


def test_dataset_returns_int_tensors():
    ds = TinyfactsDataset(subfolders=[VALID_SUBFOLDER], context_size=CONTEXT_SIZE)
    x, y = ds[0]
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_dataset_rejects_nonexistent_subfolder():
    with pytest.raises(ValueError, match="Subfolder not found"):
        TinyfactsDataset(subfolders=["does_not_exist_xyz"], context_size=CONTEXT_SIZE)


def test_dataset_multiple_subfolders():
    ds = TinyfactsDataset(
        subfolders=["manually_created", "claude_sonnet_4_5_created"],
        context_size=CONTEXT_SIZE,
        skip_invalid=True,
    )
    assert len(ds) > 0


def test_dataset_len_matches_windows():
    ds = TinyfactsDataset(subfolders=[VALID_SUBFOLDER], context_size=CONTEXT_SIZE)
    # Each window is context_size tokens; total windows = total_tokens - context_size
    expected = len(ds._tokens) - CONTEXT_SIZE
    assert len(ds) == expected
