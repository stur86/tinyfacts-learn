#!/usr/bin/env python
"""main.py — tinyfacts-learn CLI hub."""
import json
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer

from tokenizers import WordTokenizer
from train import MODELS_DIR, load_config, load_model_module

app = typer.Typer(help="tinyfacts-learn: train and inspect small language models.")


# ── train ──────────────────────────────────────────────────────────────────────

@app.command()
def train(
    model_name: Annotated[str, typer.Argument(help="Model folder name under models/")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Run 2 steps and exit")] = False,
):
    """Train a model."""
    from train import train as run_train
    run_train(model_name, dry_run=dry_run)


# ── inspect ────────────────────────────────────────────────────────────────────

@app.command()
def inspect(
    model_name: Annotated[str, typer.Argument(help="Model folder name under models/")],
    checkpoint: Annotated[Optional[Path], typer.Option(help="Path to a checkpoint .pt file")] = None,
):
    """Inspect a model: architecture, parameter counts, training history."""
    config = load_config(model_name)
    module = load_model_module(model_name)

    tokenizer = WordTokenizer(ignore_case=True, digits=True)
    vocab_size = tokenizer.vocab_size
    model = module.build_model(config, vocab_size=vocab_size)

    # ── Config ──
    typer.echo(f"\nModel: {model_name}")
    typer.echo("─" * 40)
    typer.echo("Config:")
    for k, v in config.items():
        typer.echo(f"  {k}: {v}")

    # ── Parameters ──
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    typer.echo(f"\nTotal parameters:     {total:>12,}")
    typer.echo(f"Trainable parameters: {trainable:>12,}")

    # Per-component breakdown, handling tied weights correctly
    typer.echo("\nComponent breakdown:")
    seen: set[int] = set()
    for name, child in model.named_children():
        child_params = list(child.parameters())
        unique = [p for p in child_params if p.data_ptr() not in seen]
        seen.update(p.data_ptr() for p in child_params)
        n = sum(p.numel() for p in unique)
        note = "  (tied)" if child_params and n == 0 else ""
        typer.echo(f"  {name:<20} {n:>10,}{note}")

    # ── Checkpoint ──
    if checkpoint is not None:
        if not checkpoint.exists():
            typer.echo(f"\nCheckpoint not found: {checkpoint}", err=True)
            raise typer.Exit(1)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        typer.echo(f"\nCheckpoint: {checkpoint}")
        typer.echo(f"  Trained to step: {state.get('step', '?'):,}")

    # ── Training history ──
    runs_dir = MODELS_DIR / model_name / "runs"
    jsonl_files = sorted(runs_dir.glob("run_*.jsonl")) if runs_dir.exists() else []
    if jsonl_files:
        typer.echo(f"\nTraining runs ({len(jsonl_files)} found):")
        for run_file in jsonl_files:
            lines = [json.loads(l) for l in run_file.read_text().splitlines() if l.strip()]
            if not lines:
                continue
            last = lines[-1]
            typer.echo(
                f"  {run_file.name}  "
                f"steps={last['step']}  "
                f"loss={last['loss']:.4f}  "
                f"ppl={last['perplexity']:.2f}  "
                f"acc={last['accuracy']:.3f}"
            )
    typer.echo("")


if __name__ == "__main__":
    app()
