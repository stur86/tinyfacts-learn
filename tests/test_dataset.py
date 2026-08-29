# tests/test_dataset.py
"""The dataset, built from fake chunks so the tests need no network or token."""

import json

import pytest
import torch
from tinyfacts.dataset import DatasetRecord

from tinyfacts_learn import hub_data
from tinyfacts_learn.dataset import TinyfactsDataset

CONTEXT_SIZE = 8

# Only words from the Thing Explainer list, so `validate=True` is happy.
TEXTS = [
    ("manually/sun", "manually", "The sun is a big hot light in the sky."),
    ("manually/rain", "manually", "Rain is water that falls out of the sky."),
    ("robot/water", "robot", "Water is a thing you drink when you are dry."),
    ("robot/trees", "robot", "Trees are big green things that grow out of the ground."),
]

FAKE_SHA = "0" * 40


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """A snapshot folder laid out the way the Hub cache lays one out."""
    snapshot = tmp_path / "snapshots" / FAKE_SHA
    data_dir = snapshot / hub_data.DATA_DIR
    data_dir.mkdir(parents=True)

    chunk = data_dir / f"{hub_data.CHUNK_PREFIX}-0000.jsonl"
    lines = []
    for record_id, source, text in TEXTS:
        record = DatasetRecord.build(
            id=record_id,
            text=text,
            source=source,
            title=record_id.split("/")[-1],
        )
        lines.append(record.to_json_line())
    chunk.write_text("\n".join(lines) + "\n")

    monkeypatch.setattr(hub_data, "snapshot_download", lambda **kwargs: str(snapshot))
    return snapshot


def test_loads_every_row(fake_hub):
    ds = TinyfactsDataset(context_size=CONTEXT_SIZE)
    assert ds.n_records == len(TEXTS)
    assert ds.sources == ["manually", "robot"]


def test_filters_by_source(fake_hub):
    ds = TinyfactsDataset(sources=["robot"], context_size=CONTEXT_SIZE)
    assert ds.n_records == 2
    assert ds.sources == ["robot"]


def test_item_shapes_and_dtype(fake_hub):
    ds = TinyfactsDataset(context_size=CONTEXT_SIZE)
    x, y = ds[0]
    assert x.shape == (CONTEXT_SIZE,)
    assert y.shape == (CONTEXT_SIZE,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_target_is_shifted(fake_hub):
    ds = TinyfactsDataset(context_size=CONTEXT_SIZE)
    x, y = ds[0]
    assert torch.equal(x[1:], y[:-1])


def test_len_matches_windows(fake_hub):
    ds = TinyfactsDataset(context_size=CONTEXT_SIZE)
    assert len(ds) == ds.n_tokens - CONTEXT_SIZE


def test_rows_are_read_in_id_order(fake_hub):
    """The token stream must not depend on the order rows came down in."""
    first = TinyfactsDataset(context_size=CONTEXT_SIZE)
    second = TinyfactsDataset(context_size=CONTEXT_SIZE)
    assert torch.equal(first[0][0], second[0][0])


def test_unknown_source_lists_the_real_ones(fake_hub):
    with pytest.raises(ValueError, match="manually, robot"):
        TinyfactsDataset(sources=["does_not_exist"], context_size=CONTEXT_SIZE)


def test_extra_filters_are_passed_through(fake_hub):
    ds = TinyfactsDataset(context_size=CONTEXT_SIZE, filters={"title": "^sun$"})
    assert ds.n_records == 1


def test_revision_comes_from_the_snapshot_path(fake_hub):
    ds = TinyfactsDataset(context_size=CONTEXT_SIZE)
    assert ds.revision == FAKE_SHA


def test_validate_keeps_valid_rows(fake_hub):
    ds = TinyfactsDataset(context_size=CONTEXT_SIZE, validate=True)
    assert ds.n_records == len(TEXTS)


def test_validate_drops_out_of_vocabulary_rows(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshots" / FAKE_SHA
    data_dir = snapshot / hub_data.DATA_DIR
    data_dir.mkdir(parents=True)
    rows = [
        DatasetRecord.build(id="a/good", text=TEXTS[0][2], source="a"),
        # "photosynthesis" is not in the Thing Explainer list.
        DatasetRecord.build(
            id="a/bad", text="The tree does photosynthesis in the sun.", source="a"
        ),
    ]
    (data_dir / f"{hub_data.CHUNK_PREFIX}-0000.jsonl").write_text(
        "\n".join(r.to_json_line() for r in rows) + "\n"
    )
    monkeypatch.setattr(hub_data, "snapshot_download", lambda **kwargs: str(snapshot))

    with pytest.warns(UserWarning, match="outside the list"):
        ds = TinyfactsDataset(context_size=CONTEXT_SIZE, validate=True)
    assert ds.n_records == 1


def test_context_larger_than_the_data_is_refused(fake_hub):
    with pytest.raises(ValueError, match="Not enough tokens"):
        TinyfactsDataset(context_size=100_000)


def test_empty_dataset_is_reported(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshots" / FAKE_SHA
    (snapshot / hub_data.DATA_DIR).mkdir(parents=True)
    monkeypatch.setattr(hub_data, "snapshot_download", lambda **kwargs: str(snapshot))
    with pytest.raises(hub_data.HubDataError, match="No rows found"):
        TinyfactsDataset(context_size=CONTEXT_SIZE)


def test_missing_repo_names_the_token_variables(monkeypatch):
    from huggingface_hub.errors import RepositoryNotFoundError

    def refuse(**kwargs):
        raise RepositoryNotFoundError("nope")

    monkeypatch.setattr(hub_data, "snapshot_download", refuse)
    with pytest.raises(hub_data.HubDataError, match="HF_TOKEN"):
        hub_data.load_records()


def test_repo_id_resolution(monkeypatch):
    monkeypatch.delenv(hub_data.REPO_ENV_VAR, raising=False)
    assert hub_data.resolve_repo_id() == hub_data.DEFAULT_REPO_ID
    assert hub_data.resolve_repo_id("me/other") == "me/other"
    monkeypatch.setenv(hub_data.REPO_ENV_VAR, "env/repo")
    assert hub_data.resolve_repo_id() == "env/repo"
    assert hub_data.resolve_repo_id("me/other") == "me/other"


def test_snapshot_revision_ignores_a_non_sha_folder(tmp_path):
    assert hub_data.snapshot_revision(tmp_path / FAKE_SHA) == FAKE_SHA
    assert hub_data.snapshot_revision(tmp_path / "main") is None


@pytest.mark.network
def test_real_hub_pull():
    """The actual dataset. Needs a token; deselected unless -m network is given."""
    records, revision = hub_data.load_records()
    assert records
    assert revision is None or len(revision) == 40
