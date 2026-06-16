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
