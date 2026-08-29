# report.py
"""Generate plots from a tinyfacts-learn training run JSONL file."""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker

# The colour names the split, not the metric: train is the same blue in every
# panel it appears in, val the same orange, so a reader who has learned one
# panel has learned them all. Single-series panels (lr, and anything derived)
# take a third hue that neither split owns.
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
DERIVED_COLOR = "#4a3aa7"
SPLIT_PALETTE = {"train": TRAIN_COLOR, "val": VAL_COLOR}

# Ink and surface. Labels wear these, never a series colour — the line beside
# them already carries the identity.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"
AXIS_LINE = "#d8d7d2"
GRID_LINE = "#e7e6e1"


def _apply_theme() -> None:
    """Seaborn theme with recessive grid and axes, so the data reads first."""
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS_LINE,
            "axes.labelcolor": INK_MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "medium",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID_LINE,
            "grid.linewidth": 0.8,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
        }
    )


@dataclass(frozen=True)
class Metric:
    """One panel: where its numbers come from and how they are written."""

    stem: str
    title: str
    ylabel: str
    column: str
    val_column: str | None = None
    log_scale: bool = False
    fmt: Callable[[float], str] = lambda v: f"{v:,.3f}"


def _fmt_ratio(value: float) -> str:
    return f"{value:.1%}"


def _fmt_lr(value: float) -> str:
    return f"{value:.1e}"


def _fmt_count(value: float) -> str:
    return f"{value:,.0f}"


METRICS = (
    Metric("loss", "Loss", "cross-entropy loss", "loss", "val_loss"),
    Metric(
        "perplexity",
        "Perplexity",
        "perplexity",
        "perplexity",
        "val_perplexity",
        log_scale=True,
        fmt=lambda v: f"{v:,.1f}",
    ),
    Metric(
        "accuracy",
        "Top-1 accuracy",
        "accuracy",
        "accuracy",
        "val_accuracy",
        fmt=_fmt_ratio,
    ),
    Metric("lr", "Learning rate", "learning rate", "lr", log_scale=True, fmt=_fmt_lr),
)


def _load_run(jsonl_path: Path) -> pd.DataFrame:
    """Parse a run JSONL file into a frame, one row per eval interval.

    A run that started logging validation part-way through leaves gaps rather
    than ragged columns, and the plots skip the gaps.
    """
    rows = [
        json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"No data found in {jsonl_path}")

    frame = pd.DataFrame(rows)
    if "step" not in frame:
        frame["step"] = range(1, len(frame) + 1)
    if "epoch" not in frame:
        frame["epoch"] = [i / max(len(frame), 1) for i in range(len(frame))]
    return frame.sort_values("step").reset_index(drop=True)


def _load_meta(jsonl_path: Path) -> dict | None:
    """The `.meta.json` sidecar `train` writes beside the stats, when there is one."""
    meta_path = jsonl_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None


def _describe_dataset(meta: dict | None) -> str:
    """A one-line subtitle saying which rows the run actually saw."""
    if not meta:
        return ""
    parts = []
    repo, revision = meta.get("repo_id"), meta.get("revision")
    if repo:
        parts.append(f"{repo}@{revision[:7]}" if revision else repo)
    if meta.get("n_records") is not None:
        parts.append(f"{meta['n_records']:,} rows / {meta.get('n_tokens', 0):,} tokens")
    if meta.get("n_val_records") is not None:
        parts.append(f"val {meta['n_val_records']:,} rows")
    return "  ·  ".join(parts)


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


def _epoch_ticks(ax, steps, epochs) -> None:
    """Add a secondary x-axis showing epoch numbers.

    An epoch here is one full pass over the training corpus. With overlapping
    windows and a large step budget that can reach the hundreds, so ticks are
    placed at round intervals rather than at every integer boundary. This is the
    same axis in another unit, not a second scale.
    """
    steps = list(steps)
    epochs = list(epochs)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

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
    ax2.set_xticklabels(labels, fontsize=7, color=INK_MUTED)
    ax2.set_xlabel("Epoch", fontsize=8, color=INK_MUTED)
    ax2.tick_params(length=0)


def _label_end(ax, x, y, text: str, color: str, dy: float = 0.0) -> None:
    """Write the final value at the end of a line.

    One label per series rather than a number on every point: the end of a
    training curve is the number anyone reading it came for.
    """
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(5, dy),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=INK_MUTED,
        annotation_clip=False,
    )
    ax.plot([x], [y], marker="o", markersize=4, color=color, zorder=5)


def _label_ends(ax, points: list[tuple[float, float, str, str]]) -> None:
    """Label the end of every line, pushing the labels apart when they collide.

    Train and val converge, which is exactly when their two end labels land on
    top of each other, so the closer they get the more they are staggered.
    """

    def axis_fraction(value: float) -> float:
        lo, hi = ax.get_ylim()
        if ax.get_yscale() == "log":
            lo, hi, value = (
                math.log10(lo),
                math.log10(hi),
                math.log10(max(value, 1e-12)),
            )
        return (value - lo) / (hi - lo) if hi != lo else 0.0

    offsets = [0.0] * len(points)
    if len(points) == 2:
        separation = abs(axis_fraction(points[0][1]) - axis_fraction(points[1][1]))
        if separation < 0.05:
            higher = 0 if points[0][1] >= points[1][1] else 1
            offsets[higher] = 6.0
            offsets[1 - higher] = -6.0

    for (x, y, text, color), dy in zip(points, offsets):
        _label_end(ax, x, y, text, color, dy=dy)


def _finish_axes(
    ax, xlabel: str, ylabel: str, title: str, log_scale: bool, thousands_y: bool = False
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    if log_scale:
        ax.set_yscale("log")
    elif thousands_y:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))
    # Room on the right for the end-of-line labels
    ax.margins(x=0.1)


def _plot_metric(ax, frame: pd.DataFrame, metric: Metric) -> bool:
    """Draw one metric, with its validation counterpart when the run logged one.

    Returns False when the run holds no data for it, so the caller can skip the
    panel entirely rather than write an empty one.
    """
    if metric.column not in frame or frame[metric.column].dropna().empty:
        return False

    columns = {"train": metric.column}
    if metric.val_column and metric.val_column in frame:
        if not frame[metric.val_column].dropna().empty:
            columns["val"] = metric.val_column

    long = (
        frame[["step", *columns.values()]]
        .rename(columns={v: k for k, v in columns.items()})
        .melt(id_vars="step", var_name="split", value_name="value")
        .dropna(subset=["value"])
    )

    has_val = "val" in columns
    sns.lineplot(
        data=long,
        x="step",
        y="value",
        hue="split",
        hue_order=[k for k in ("train", "val") if k in columns],
        palette=SPLIT_PALETTE,
        legend=has_val,
        ax=ax,
    )
    if has_val:
        ax.legend(title=None, fontsize=8, loc="best")

    _finish_axes(ax, "Step", metric.ylabel, metric.title, metric.log_scale)

    ends = []
    for split in columns:
        last = long[long["split"] == split].iloc[-1]
        ends.append(
            (
                last["step"],
                last["value"],
                metric.fmt(last["value"]),
                SPLIT_PALETTE[split],
            )
        )
    _label_ends(ax, ends)
    return True


def _plot_single(
    ax,
    x,
    y,
    title: str,
    ylabel: str,
    color: str,
    fmt: Callable[[float], str],
    log_scale: bool = False,
    baseline: float | None = None,
    thousands_y: bool = False,
) -> None:
    """A one-series panel. No legend — the title names the series."""
    x, y = list(x), list(y)
    sns.lineplot(x=x, y=y, color=color, ax=ax)
    if baseline is not None:
        ax.axhline(baseline, color=INK_MUTED, linewidth=0.8, linestyle=":", alpha=0.7)
    _finish_axes(ax, "Step", ylabel, title, log_scale, thousands_y=thousands_y)
    if x:
        _label_end(ax, x[-1], y[-1], fmt(y[-1]), color)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _generalization_gap(frame: pd.DataFrame) -> pd.DataFrame | None:
    """val_loss − loss per step: how far the model has drifted from the val set.

    Kept as its own panel because on the shared loss axes the two curves sit
    close enough that the gap — the thing that says whether the model has
    started memorising — is the hardest thing to read off them.
    """
    if "val_loss" not in frame or "loss" not in frame:
        return None
    gap = frame[["step", "loss", "val_loss"]].dropna()
    if gap.empty:
        return None
    return pd.DataFrame({"step": gap["step"], "gap": gap["val_loss"] - gap["loss"]})


def _throughput(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Tokens per second between flushes — what the run cost, as it ran."""
    if "tokens_seen" not in frame or "elapsed_s" not in frame:
        return None
    run = frame[["step", "tokens_seen", "elapsed_s"]].dropna()
    if len(run) < 2:
        return None
    d_tokens = run["tokens_seen"].diff()
    d_seconds = run["elapsed_s"].diff()
    rate = (d_tokens / d_seconds).where(d_seconds > 0)
    out = pd.DataFrame({"step": run["step"], "tokens_per_s": rate}).dropna()
    return out if not out.empty else None


def generate_report(jsonl_path: Path) -> Path:
    """Generate plots from a training run JSONL file.

    Creates a directory alongside the JSONL file named after the run stem,
    writes one PNG per metric plus a 2×2 overview, and returns the output dir.
    Panels the run holds no data for are skipped.
    """
    jsonl_path = Path(jsonl_path)
    frame = _load_run(jsonl_path)
    meta = _load_meta(jsonl_path)
    subtitle = _describe_dataset(meta)

    out_dir = jsonl_path.parent / jsonl_path.stem
    out_dir.mkdir(exist_ok=True)

    _apply_theme()

    # ── Individual plots ──────────────────────────────────────────────────────
    drawn: list[Metric] = []
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        if not _plot_metric(ax, frame, metric):
            plt.close(fig)
            continue
        _epoch_ticks(ax, frame["step"], frame["epoch"])
        _save(fig, out_dir / f"{metric.stem}.png")
        drawn.append(metric)

    # ── Derived plots ────────────────────────────────────────────────────────
    gap = _generalization_gap(frame)
    if gap is not None:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        _plot_single(
            ax,
            gap["step"],
            gap["gap"],
            "Generalization gap",
            "val loss − train loss",
            DERIVED_COLOR,
            fmt=lambda v: f"{v:+.3f}",
            baseline=0.0,
        )
        _epoch_ticks(ax, frame["step"], frame["epoch"])
        _save(fig, out_dir / "gap.png")

    throughput = _throughput(frame)
    if throughput is not None:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        _plot_single(
            ax,
            throughput["step"],
            throughput["tokens_per_s"],
            "Throughput",
            "tokens / second",
            DERIVED_COLOR,
            fmt=_fmt_count,
            thousands_y=True,
        )
        _save(fig, out_dir / "throughput.png")

    # ── 2×2 overview ─────────────────────────────────────────────────────────
    if len(drawn) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
        title = f"Training run: {jsonl_path.stem}"
        fig.suptitle(title, fontsize=13, color=INK, y=0.985)
        if subtitle:
            fig.text(0.5, 0.952, subtitle, ha="center", fontsize=8.5, color=INK_MUTED)
        for ax, metric in zip(axes.flat, drawn):
            _plot_metric(ax, frame, metric)
        for ax in axes.flat[len(drawn) :]:
            ax.set_visible(False)
        fig.tight_layout(rect=(0, 0, 1, 0.94 if subtitle else 0.96))
        fig.savefig(out_dir / "overview.png", dpi=150)
        plt.close(fig)

    return out_dir
