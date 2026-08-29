# tests/test_report.py
import json
import math
import pytest
from pathlib import Path

from tinyfacts_learn.report import generate_report


@pytest.fixture()
def synthetic_run(tmp_path: Path) -> Path:
    """Write a small synthetic run JSONL file and return its path."""
    entries = []
    for step in range(100, 600, 100):
        lr = 3e-4 * (1 + math.cos(math.pi * step / 500)) / 2
        entries.append({
            "step": step,
            "epoch": round(step / 447, 3),
            "loss": round(6.9 - step / 1000, 6),
            "perplexity": round(math.exp(6.9 - step / 1000), 4),
            "accuracy": round(step / 5000, 6),
            "lr": lr,
            "tokens_seen": step * 64 * 128,
            "elapsed_s": step * 0.1,
            "timestamp": "2026-02-26T12:00:00",
        })
    run_file = tmp_path / "run_test.jsonl"
    run_file.write_text("\n".join(json.dumps(e) for e in entries))
    return run_file


@pytest.fixture()
def synthetic_run_with_val(tmp_path: Path) -> Path:
    """A run that logged validation metrics, with the sidecar `train` writes."""
    entries = []
    for step in range(100, 600, 100):
        loss = 6.9 - step / 1000
        val_loss = loss + 0.05 + step / 20000
        entries.append({
            "step": step,
            "epoch": round(step / 447, 3),
            "loss": round(loss, 6),
            "perplexity": round(math.exp(loss), 4),
            "accuracy": round(step / 5000, 6),
            "val_loss": round(val_loss, 6),
            "val_perplexity": round(math.exp(val_loss), 4),
            "val_accuracy": round(step / 5600, 6),
            "lr": 3e-4 * (1 + math.cos(math.pi * step / 500)) / 2,
            "tokens_seen": step * 64 * 128,
            "elapsed_s": step * 0.1,
            "timestamp": "2026-02-26T12:00:00",
        })
    run_file = tmp_path / "run_val.jsonl"
    run_file.write_text("\n".join(json.dumps(e) for e in entries))
    run_file.with_suffix(".meta.json").write_text(json.dumps({
        "repo_id": "Stur86/tinyfacts",
        "revision": "0" * 40,
        "n_records": 100,
        "n_tokens": 40000,
        "n_val_records": 5,
    }))
    return run_file


def test_report_creates_output_dir(synthetic_run):
    out_dir = generate_report(synthetic_run)
    assert out_dir.is_dir()


def test_report_individual_plots_exist(synthetic_run):
    out_dir = generate_report(synthetic_run)
    for name in ("loss.png", "perplexity.png", "accuracy.png", "lr.png"):
        assert (out_dir / name).exists(), f"Missing plot: {name}"


def test_report_overview_exists(synthetic_run):
    out_dir = generate_report(synthetic_run)
    assert (out_dir / "overview.png").exists()


def test_report_output_dir_named_after_run(synthetic_run):
    out_dir = generate_report(synthetic_run)
    assert out_dir.name == synthetic_run.stem


def test_report_empty_file_raises(tmp_path):
    empty = tmp_path / "run_empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="No data"):
        generate_report(empty)


def test_report_gap_plot_needs_validation(synthetic_run_with_val):
    """The generalization gap is val minus train, so it needs both."""
    assert (generate_report(synthetic_run_with_val) / "gap.png").exists()


def test_report_skips_gap_without_validation(synthetic_run):
    assert not (generate_report(synthetic_run) / "gap.png").exists()


def test_report_throughput_plot(synthetic_run):
    assert (generate_report(synthetic_run) / "throughput.png").exists()


def test_report_plots_validation_run(synthetic_run_with_val):
    out_dir = generate_report(synthetic_run_with_val)
    for name in ("loss.png", "perplexity.png", "accuracy.png", "lr.png", "overview.png"):
        assert (out_dir / name).exists(), f"Missing plot: {name}"


def test_report_skips_metrics_the_run_never_logged(tmp_path):
    """A run holding only a loss column gets a loss plot and nothing invented."""
    run_file = tmp_path / "run_sparse.jsonl"
    run_file.write_text("\n".join(
        json.dumps({"step": step, "loss": 6.9 - step / 1000})
        for step in range(100, 600, 100)
    ))
    out_dir = generate_report(run_file)
    assert (out_dir / "loss.png").exists()
    assert not (out_dir / "lr.png").exists()
    assert not (out_dir / "throughput.png").exists()
