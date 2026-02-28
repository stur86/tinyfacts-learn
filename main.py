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
            lines = [json.loads(ln) for ln in run_file.read_text().splitlines() if ln.strip()]
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


# ── report ─────────────────────────────────────────────────────────────────────

@app.command()
def report(
    jsonl_file: Annotated[Path, typer.Argument(help="Path to a run_*.jsonl stats file")],
):
    """Generate a report folder with plots from a training run JSONL file."""
    from report import generate_report

    if not jsonl_file.exists():
        typer.echo(f"File not found: {jsonl_file}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Reading {jsonl_file} ...")
    out_dir = generate_report(jsonl_file)
    plots = sorted(out_dir.glob("*.png"))
    typer.echo(f"Report written to {out_dir}/")
    for p in plots:
        typer.echo(f"  {p.name}")


# ── generate ───────────────────────────────────────────────────────────────────

@app.command()
def generate(
    model_name: Annotated[str, typer.Argument(help="Model folder name under models/")],
    checkpoint: Annotated[Optional[Path], typer.Option(help="Checkpoint .pt file (default: latest)")] = None,
    tokens: Annotated[int, typer.Option(help="Number of tokens to generate")] = 100,
    temperature: Annotated[float, typer.Option(help="Sampling temperature (0 = greedy)")] = 0.5,
    top_k: Annotated[int, typer.Option(help="Top-k sampling (0 = disabled)")] = 10,
):
    """Interactively generate text — loads a checkpoint then prompts for input."""
    from generate import generate_tokens

    # Resolve checkpoint — default to latest
    if checkpoint is None:
        checkpoint_dir = MODELS_DIR / model_name / "checkpoints"
        candidates = sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
        if not candidates:
            typer.echo(f"No checkpoints found in {checkpoint_dir}", err=True)
            raise typer.Exit(1)
        checkpoint = candidates[-1]
        typer.echo(f"Using checkpoint: {checkpoint.name}")

    if not checkpoint.exists():
        typer.echo(f"Checkpoint not found: {checkpoint}", err=True)
        raise typer.Exit(1)

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    step = state.get("step", "?")
    typer.echo(f"Trained to step: {step:,}" if isinstance(step, int) else f"Trained to step: {step}")

    module = load_model_module(model_name)
    tokenizer = WordTokenizer(ignore_case=True, digits=True)
    model = module.build_model(config, vocab_size=tokenizer.vocab_size)
    model.load_state_dict(state["model_state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    typer.echo(f"\nModel ready. Generating {tokens} tokens per prompt.")
    typer.echo("Enter a prompt and press Enter. Ctrl+C to quit.\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (KeyboardInterrupt, EOFError):
            typer.echo("\nGoodbye!")
            break

        if not prompt:
            continue

        try:
            generated, _ = generate_tokens(
                model, tokenizer, prompt,
                n_tokens=tokens, temperature=temperature, top_k=top_k, device=device,
            )
            typer.echo(f"\n{prompt} {generated}\n")
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)


if __name__ == "__main__":
    app()
