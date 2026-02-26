# train.py
"""Core training logic for tinyfacts-learn models."""
import importlib.util
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load model module from {model_file}")
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
    batch_size = config.get("batch_size", 64)
    print(f"Vocab size: {vocab_size} | Dataset size: {len(dataset)} windows")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    steps_per_epoch = len(dataloader)

    model = module.build_model(config, vocab_size=vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 3e-4))

    max_steps = 2 if dry_run else config.get("max_steps", 10000)
    min_lr = config.get("min_lr", 1e-5)
    warmup_steps = 0 if dry_run else config.get("warmup_steps", 0)

    if warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=min_lr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=min_lr)
    eval_interval = config.get("eval_interval", 500)
    checkpoint_interval = config.get("checkpoint_interval", 1000)

    checkpoint_dir = MODELS_DIR / model_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = MODELS_DIR / model_name / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stats_file = runs_dir / f"run_{run_timestamp}.jsonl"

    print(f"Stats log: {stats_file}")
    print(f"Training for {max_steps} steps{'  [DRY RUN]' if dry_run else ''}...")

    model.train()
    step = 0
    data_iter = iter(dataloader)
    train_start = time.time()

    # Running accumulators — reset after each flush
    acc_loss = 0.0
    acc_correct = 0
    acc_tokens = 0
    acc_steps = 0

    def flush_stats():
        avg_loss = acc_loss / acc_steps
        accuracy = acc_correct / acc_tokens
        perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow
        epoch = step / steps_per_epoch
        tokens_seen = step * batch_size * config["context_size"]
        elapsed = time.time() - train_start
        lr = scheduler.get_last_lr()[0]
        entry = {
            "step": step,
            "epoch": round(epoch, 3),
            "loss": round(avg_loss, 6),
            "perplexity": round(perplexity, 4),
            "accuracy": round(accuracy, 6),
            "lr": lr,
            "tokens_seen": tokens_seen,
            "elapsed_s": round(elapsed, 2),
            "timestamp": datetime.now().isoformat(),
        }
        with open(stats_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(
            f"step {step:>6}/{max_steps} | epoch {epoch:.2f} | "
            f"loss {avg_loss:.4f} | ppl {perplexity:.2f} | acc {accuracy:.3f} | "
            f"lr {lr:.2e}"
        )
        return entry

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
        scheduler.step()

        # Accumulate stats (detached — no grad overhead)
        with torch.no_grad():
            preds = logits.detach().argmax(dim=-1)
            acc_correct += (preds == y).sum().item()
            acc_tokens += y.numel()
        acc_loss += loss.item()
        acc_steps += 1
        step += 1

        if step % eval_interval == 0:
            flush_stats()
            acc_loss = acc_correct = acc_tokens = acc_steps = 0

        if not dry_run and step % checkpoint_interval == 0:
            _save_checkpoint(model, optimizer, config, step, checkpoint_dir, model_name)

    # Always flush remaining accumulated stats at the end
    if acc_steps > 0:
        flush_stats()

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
