# tests/test_dataset.py
import pytest
import torch
from pathlib import Path
from tinyfacts_learn.dataset import TinyfactsDataset, TINYFACTS_GEN_DIR

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


def test_dataset_raises_on_invalid_file_by_default():
    # claude_sonnet_4_5_created contains files with OOV words
    with pytest.raises(ValueError, match="invalid word"):
        TinyfactsDataset(
            subfolders=["claude_sonnet_4_5_created"],
            context_size=CONTEXT_SIZE,
            # skip_invalid defaults to False
        )


# ── stride ────────────────────────────────────────────────────────────────────

def test_stride_reduces_window_count():
    ds1 = TinyfactsDataset(subfolders=[VALID_SUBFOLDER], context_size=CONTEXT_SIZE, stride=1)
    ds4 = TinyfactsDataset(subfolders=[VALID_SUBFOLDER], context_size=CONTEXT_SIZE, stride=4)
    n_tokens = ds1.n_tokens
    assert len(ds1) == n_tokens - CONTEXT_SIZE
    assert len(ds4) == -(-(n_tokens - CONTEXT_SIZE) // 4)  # ceil division


def test_stride_advances_window_start():
    stride = 3
    ds = TinyfactsDataset(subfolders=[VALID_SUBFOLDER], context_size=CONTEXT_SIZE, stride=stride)
    x0, _ = ds[0]
    x1, _ = ds[1]
    # Window 1 starts `stride` tokens later, so its head overlaps window 0's tail
    assert torch.equal(x1[: CONTEXT_SIZE - stride], x0[stride:])


def test_non_overlapping_stride_gives_one_pass():
    """stride == context_size means an epoch is one pass over the corpus."""
    ds = TinyfactsDataset(
        subfolders=[VALID_SUBFOLDER], context_size=CONTEXT_SIZE, stride=CONTEXT_SIZE
    )
    assert len(ds) == -(-(ds.n_tokens - CONTEXT_SIZE) // CONTEXT_SIZE)


def test_rejects_invalid_stride():
    with pytest.raises(ValueError, match="stride must be"):
        TinyfactsDataset(subfolders=[VALID_SUBFOLDER], context_size=CONTEXT_SIZE, stride=0)


# ── train/val split ───────────────────────────────────────────────────────────

# A subfolder with enough files that a 50/50 split leaves both sides non-empty
SPLIT_SUBFOLDER = "questions_gemini-3-flash-preview_cloud_created"
SPLIT_KWARGS = dict(
    subfolders=[SPLIT_SUBFOLDER],
    context_size=CONTEXT_SIZE,
    skip_invalid=True,
    val_fraction=0.5,
)


def test_split_partitions_all_files():
    all_ds = TinyfactsDataset(split="all", **SPLIT_KWARGS)
    train_ds = TinyfactsDataset(split="train", **SPLIT_KWARGS)
    val_ds = TinyfactsDataset(split="val", **SPLIT_KWARGS)
    assert train_ds.n_files > 0 and val_ds.n_files > 0
    assert train_ds.n_files + val_ds.n_files == all_ds.n_files
    assert train_ds.n_tokens + val_ds.n_tokens == all_ds.n_tokens


def test_split_is_deterministic():
    a = TinyfactsDataset(split="val", **SPLIT_KWARGS)
    b = TinyfactsDataset(split="val", **SPLIT_KWARGS)
    assert torch.equal(a.tokens, b.tokens)


def test_split_seed_changes_partition():
    a = TinyfactsDataset(split="val", split_seed=0, **SPLIT_KWARGS)
    b = TinyfactsDataset(split="val", split_seed=99, **SPLIT_KWARGS)
    assert not torch.equal(a.tokens, b.tokens)


def test_rejects_invalid_split():
    with pytest.raises(ValueError, match="split must be"):
        TinyfactsDataset(subfolders=[VALID_SUBFOLDER], context_size=CONTEXT_SIZE, split="test")
