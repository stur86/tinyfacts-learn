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

from .dataset import TinyfactsDataset
from .tokenizers import WordTokenizer

MODELS_DIR = Path(__file__).parent.parent / "models"


def load_model_module(model_name: str):
    def _validate_model_ref(ref: str) -> str:
        ref = ref.strip()
        if not ref:
            raise ValueError("model.source is empty")
        # Treat model names as folder names under models/ (no path traversal)
        if any(sep in ref for sep in ("/", "\\")) or ref in (".", "..") or ".." in ref:
            raise ValueError(f"Invalid model.source reference: {ref!r}")
        return ref

    def _resolve_model_dir(start_name: str) -> tuple[str, Path]:
        chain: list[str] = []
        current = start_name

        while True:
            if current in chain:
                loop = " -> ".join(chain + [current])
                raise ValueError(f"model.source cycle detected: {loop}")
            chain.append(current)

            current_dir = MODELS_DIR / current
            if not current_dir.exists():
                raise ValueError(f"Model directory not found: {current_dir}")

            source_file = current_dir / "model.source"
            if source_file.exists():
                target = _validate_model_ref(source_file.read_text())
                current = target
                continue

            return current, current_dir

    resolved_name, model_dir = _resolve_model_dir(model_name)
    model_file = model_dir / "model.py"
    if not model_file.exists():
        raise ValueError(
            f"model.py not found in {model_dir} (resolved from {model_name!r} -> {resolved_name!r})"
        )

    spec_name = f"{model_name}_model_from_{resolved_name}"
    spec = importlib.util.spec_from_file_location(spec_name, model_file)
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

    if not dry_run:
        if device == "cuda" and torch.cuda.get_device_capability()[0] < 7:
            compile_backend = "cudagraphs"
        else:
            compile_backend = "inductor"
        print(f"Compiling model with torch.compile (backend={compile_backend}, first step will be slow)...")
        model = torch.compile(model, backend=compile_backend)

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
    n_supervision = config.get("n_supervision", 0)
    n_recursions  = config.get("n_recursions", 6)
    T_cycles      = config.get("T", 3)
    ema_decay     = config.get("ema_decay", 0.0)
    ema_state = None
    if ema_decay > 0:
        ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

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

        if n_supervision > 0 and hasattr(model, "latent_recursion"):
            # ── Deep supervision (TRM) ──────────────────────────────────
            n_embd = config["n_embd"]
            y_lat = torch.zeros(x.shape[0], x.shape[1], n_embd, device=device)
            z_lat = torch.zeros_like(y_lat)

            total_loss = 0.0
            last_logits = None
            steps_taken = 0

            for _sup in range(n_supervision):
                optimizer.zero_grad()

                # Re-embed with latest weights each supervision step
                x_emb = model.embed(x)

                # T-1 warm-up cycles — improve y_lat/z_lat without grad
                with torch.no_grad():
                    for _ in range(T_cycles - 1):
                        y_lat, z_lat = model.latent_recursion(
                            x_emb.detach(), y_lat, z_lat, n_recursions
                        )

                # Final cycle — full gradient through n+1 transformer calls
                y_new, z_new = model.latent_recursion(x_emb, y_lat, z_lat, n_recursions)
                logits = model.head(y_new)          # [B, L, vocab_size]
                # Average over [B, L, 1] → scalar halt signal; q_head init=0 so
                # strict >0 early-stop never fires on the very first training step.
                q_logit = model.q_head(y_new).mean()

                # Language-model loss
                lm_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                # Halt loss: teach q to predict whether the current answer is correct
                with torch.no_grad():
                    correct = (logits.detach().argmax(-1) == y).float().mean()
                halt_loss = F.binary_cross_entropy_with_logits(q_logit, correct)
                loss = lm_loss + halt_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                last_logits = logits.detach()
                steps_taken += 1
                y_lat = y_new.detach()
                z_lat = z_new.detach()

                # Early stopping: q_logit > 0 ↔ sigmoid > 0.5
                if q_logit.item() > 0:
                    break

            # EMA update after all supervision steps
            if ema_state is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema_state[k].mul_(ema_decay).add_(v, alpha=1.0 - ema_decay)

            scheduler.step()
            logits    = last_logits
            loss_item = total_loss / steps_taken

        else:
            # ── Standard single-step (gpt_small etc.) ──────────────────
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            logits    = logits.detach()
            loss_item = loss.item()

        # Accumulate stats (detached — no grad overhead)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc_correct += (preds == y).sum().item()
            acc_tokens  += y.numel()
        acc_loss  += loss_item
        acc_steps += 1
        step += 1

        if step % eval_interval == 0:
            flush_stats()
            acc_loss = acc_correct = acc_tokens = acc_steps = 0

        if not dry_run and step % checkpoint_interval == 0:
            _save_checkpoint(model, optimizer, config, step, checkpoint_dir, model_name,
                             ema_state=ema_state)

    # Always flush remaining accumulated stats at the end
    if acc_steps > 0:
        flush_stats()

    if not dry_run:
        _save_checkpoint(model, optimizer, config, step, checkpoint_dir, model_name,
                         ema_state=ema_state)
        print("Training complete.")
    else:
        print("Dry run complete.")


def _save_checkpoint(model, optimizer, config, step, checkpoint_dir, model_name,
                     ema_state=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = checkpoint_dir / f"{model_name}_{timestamp}_step{step}.pt"
    state_dict = ema_state if ema_state is not None else model.state_dict()
    torch.save(
        {
            "step": step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        filename,
    )
    print(f"Checkpoint saved: {filename}")
