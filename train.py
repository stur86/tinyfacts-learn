# train.py
"""Training script for tinyfacts-learn models.

Usage:
    uv run python train.py gpt_small
    uv run python train.py gpt_small --dry-run
"""
import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TinyfactsDataset
from tokenizers import WordTokenizer

MODELS_DIR = Path(__file__).parent / "models"


def load_model_module(model_name: str):
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")
    model_file = model_dir / "model.py"
    if not model_file.exists():
        raise ValueError(f"model.py not found in {model_dir}")
    spec = importlib.util.spec_from_file_location(f"{model_name}_model", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(model_name: str) -> dict:
    config_file = MODELS_DIR / model_name / "config.json"
    if not config_file.exists():
        raise ValueError(f"config.json not found in {MODELS_DIR / model_name}")
    return json.loads(config_file.read_text())


def train(model_name: str, dry_run: bool = False):
    config = load_config(model_name)
    module = load_model_module(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = WordTokenizer(ignore_case=True, digits=True)
    subfolders = config.get("subfolders", [])
    if not subfolders:
        print("ERROR: 'subfolders' not set in config.json", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from: {subfolders}")
    dataset = TinyfactsDataset(
        subfolders=subfolders,
        context_size=config["context_size"],
        tokenizer=tokenizer,
        skip_invalid=True,
    )
    vocab_size = dataset.vocab_size
    print(f"Vocab size: {vocab_size} | Dataset size: {len(dataset)} windows")

    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 64),
        shuffle=True,
        drop_last=True,
    )

    model = module.build_model(config, vocab_size=vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 3e-4),
    )

    max_steps = 2 if dry_run else config.get("max_steps", 10000)
    eval_interval = config.get("eval_interval", 500)
    checkpoint_interval = config.get("checkpoint_interval", 1000)

    checkpoint_dir = MODELS_DIR / model_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    data_iter = iter(dataloader)

    print(f"Training for {max_steps} steps{'  [DRY RUN]' if dry_run else ''}...")

    while step < max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)  # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1

        if step % eval_interval == 0 or step == max_steps:
            print(f"step {step}/{max_steps} | loss {loss.item():.4f}")

        if not dry_run and step % checkpoint_interval == 0:
            _save_checkpoint(model, optimizer, config, step, checkpoint_dir, model_name)

    if not dry_run:
        _save_checkpoint(model, optimizer, config, step, checkpoint_dir, model_name)
        print("Training complete.")
    else:
        print("Dry run complete.")


def _save_checkpoint(model, optimizer, config, step, checkpoint_dir, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = checkpoint_dir / f"{model_name}_{timestamp}_step{step}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        filename,
    )
    print(f"Checkpoint saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Train a tinyfacts-learn model")
    parser.add_argument("model_name", help="Name of the model folder under models/")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 2 training steps and exit (for testing)",
    )
    args = parser.parse_args()
    train(args.model_name, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
