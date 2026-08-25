# report.py
"""Generate plots from a tinyfacts-learn training run JSONL file."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _load_run(jsonl_path: Path) -> dict[str, list]:
    """Parse a run JSONL file into parallel lists keyed by metric name."""
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No data found in {jsonl_path}")
    keys = rows[0].keys()
    return {k: [r[k] for r in rows if k in r] for k in keys}


def _nice_epoch_step(max_epoch: float, target_ticks: int = 8) -> float:
    """Pick a round epoch interval giving roughly `target_ticks` labels."""
    if max_epoch <= 0:
        return 1.0
    raw = max_epoch / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        if magnitude * mult >= raw:
            return magnitude * mult
    return magnitude * 10


def _epoch_ticks(ax, steps: list[float], epochs: list[float]):
    """Add a secondary x-axis showing epoch numbers.

    An epoch here is one full pass over the training corpus. With overlapping
    windows and a large step budget that can reach the hundreds, so ticks are
    placed at round intervals rather than at every integer boundary.
    """
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    max_epoch = max(epochs) if epochs else 0
    interval = _nice_epoch_step(max_epoch)

    epoch_steps: list[float] = []
    labels: list[str] = []
    e = interval
    while e <= max_epoch:
        # Find the first logged point at or past epoch e
        for i, ep in enumerate(epochs):
            if ep >= e:
                epoch_steps.append(steps[i])
                labels.append(f"{e:g}")
                break
        e += interval

    ax2.set_xticks(epoch_steps)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_xlabel("Epoch", fontsize=8)


def _plot_metric(
    ax,
    steps: list,
    values: list,
    label: str,
    color: str,
    ylabel: str,
    log_scale: bool = False,
    val_values: list | None = None,
):
    has_val = bool(val_values)
    ax.plot(steps, values, color=color, linewidth=1.5,
            label="train" if has_val else label)
    if has_val:
        # Truncate to the shorter series in case a run was interrupted mid-flush
        n = min(len(steps), len(val_values))
        ax.plot(steps[:n], val_values[:n], color=color, linewidth=1.5,
                linestyle="--", alpha=0.8, label="val")
        ax.legend(fontsize=8)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(label)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


def generate_report(jsonl_path: Path) -> Path:
    """Generate plots from a training run JSONL file.

    Creates a directory alongside the JSONL file named after the run stem,
    writes one PNG per metric plus a 2×2 overview, and returns the output dir.
    """
    data = _load_run(jsonl_path)
    steps = data["step"]
    epochs = data.get("epoch", [i / max(len(steps), 1) for i in range(len(steps))])

    out_dir = jsonl_path.parent / jsonl_path.stem
    out_dir.mkdir(exist_ok=True)

    metrics = [
        ("loss",        data.get("loss", []),        "Loss",            "cross-entropy loss", "tab:blue",   False, data.get("val_loss", [])),
        ("perplexity",  data.get("perplexity", []),  "Perplexity",      "perplexity",         "tab:orange", False, data.get("val_perplexity", [])),
        ("accuracy",    data.get("accuracy", []),    "Top-1 Accuracy",  "accuracy",           "tab:green",  False, data.get("val_accuracy", [])),
        ("lr",          data.get("lr", []),          "Learning Rate",   "learning rate",      "tab:red",    True,  []),
    ]

    # ── Individual plots ──────────────────────────────────────────────────────
    saved = []
    for stem, values, title, ylabel, color, log_scale, val_values in metrics:
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        _plot_metric(ax, steps, values, title, color, ylabel, log_scale, val_values)
        _epoch_ticks(ax, steps, epochs)
        fig.tight_layout()
        path = out_dir / f"{stem}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # ── 2×2 overview ─────────────────────────────────────────────────────────
    available = [m for m in metrics if m[1]]
    if len(available) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Training run: {jsonl_path.stem}", fontsize=11)
        for idx, ax in enumerate(axes.flat):
            if idx < len(available):
                stem, values, title, ylabel, color, log_scale, val_values = available[idx]
                _plot_metric(ax, steps, values, title, color, ylabel, log_scale, val_values)
            else:
                ax.set_visible(False)
        fig.tight_layout()
        overview_path = out_dir / "overview.png"
        fig.savefig(overview_path, dpi=150)
        plt.close(fig)
        saved.append(overview_path)

    return out_dir
