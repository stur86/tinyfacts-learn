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

from .dataset import TinyfactsDataset
from .hub_data import DEFAULT_REPO_ID
from .tokenizers import WordTokenizer

MODELS_DIR = Path(__file__).parent.parent / "models"

# Keys that must match between a checkpoint's config and the current config.json
# for a resume to be meaningful.
_ARCH_KEYS = (
    "context_size", "n_embd", "n_heads", "n_layers", "ffn_dim",
    "d_model", "d_state", "d_conv", "expand",
    "n_supervision", "n_recursions", "T",
)


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


class TokenSampler:
    """Samples random training windows straight from a device-resident token stream.

    The whole corpus is a few tens of MB as int64, so it lives on the GPU and a
    batch is one gather — no DataLoader, no per-item Python indexing, no host to
    device copy in the training loop. Sampling is with replacement, which is the
    usual choice for LM pretraining and makes "epoch" a function of tokens seen
    rather than of the (heavily overlapping) window count.
    """

    def __init__(self, tokens: torch.Tensor, context_size: int, batch_size: int,
                 device: str, seed: int | None = None):
        self.tokens = tokens.to(device)
        self.context_size = context_size
        self.batch_size = batch_size
        self.device = device
        # A window needs context_size + 1 tokens (inputs plus the shifted target)
        self.n_starts = len(self.tokens) - context_size
        if self.n_starts < 1:
            raise ValueError(
                f"Corpus of {len(self.tokens)} tokens is too small for context_size={context_size}"
            )
        self._offsets = torch.arange(context_size + 1, device=device)
        self._gen = torch.Generator(device=device)
        if seed is not None:
            self._gen.manual_seed(seed)

    def batch(self, batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        n = batch_size or self.batch_size
        starts = torch.randint(0, self.n_starts, (n,), device=self.device, generator=self._gen)
        chunks = self.tokens[starts[:, None] + self._offsets[None, :]]
        # Slicing off one column leaves a non-contiguous view (stride is still
        # context_size + 1), which .view(-1) later rejects. Materialise both so
        # callers get the same contiguous tensors a DataLoader would have handed them.
        return chunks[:, :-1].contiguous(), chunks[:, 1:].contiguous()


@torch.no_grad()
def evaluate(model, batches, vocab_size: int) -> tuple[float, float]:
    """Mean loss and top-1 accuracy over a fixed list of (x, y) batches."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    n_tokens = 0
    for x, y in batches:
        logits = model(x)
        total_loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        n_tokens += y.numel()
    if was_training:
        model.train()
    return total_loss / len(batches), correct / n_tokens


def train(model_name: str, dry_run: bool = False, resume: Path | None = None):
    config = load_config(model_name)
    module = load_model_module(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = WordTokenizer(ignore_case=True, digits=True)
    if "subfolders" in config:
        print(
            "ERROR: 'subfolders' is no longer used. The texts now come from the "
            "dataset on the Hugging Face Hub, not the tinyfacts-gen submodule.\n"
            "       Replace it with 'sources', dropping the '_created' suffix from "
            "each name (e.g. 'manually_created' -> 'manually').",
            file=sys.stderr,
        )
        sys.exit(1)

    sources = config.get("sources", [])
    context_size = config["context_size"]
    batch_size = config.get("batch_size", 64)
    val_fraction = config.get("val_fraction", 0.05)
    val_batches_n = config.get("val_batches", 20)
    split_seed = config.get("split_seed", 0)

    print(f"Loading dataset from sources: {sources or 'all'}")
    split_kwargs = dict(
        sources=sources,
        context_size=context_size,
        tokenizer=tokenizer,
        repo_id=config.get("dataset_repo"),
        filters=config.get("dataset_filters"),
        val_fraction=val_fraction,
        split_seed=split_seed,
    )
    train_ds = TinyfactsDataset(
        split="train", revision=config.get("dataset_revision"), **split_kwargs
    )
    # Pin the val half to whatever commit the train half resolved to, so a run
    # started as the dataset changes cannot end up with mismatched halves.
    val_ds = TinyfactsDataset(split="val", revision=train_ds.revision, **split_kwargs)
    vocab_size = train_ds.vocab_size
    train_tokens = train_ds.n_tokens
    print(f"Vocab size: {vocab_size}")
    print(f"Train split: {train_ds.n_records:,} rows | {train_tokens:,} tokens")
    print(f"Val split:   {val_ds.n_records:,} rows | {val_ds.n_tokens:,} tokens")
    print(f"Dataset revision: {train_ds.revision}")

    sampler = TokenSampler(train_ds.tokens, context_size, batch_size, device)
    # Fixed seed → the same val windows every eval, every run, so the curves are comparable
    val_sampler = TokenSampler(val_ds.tokens, context_size, batch_size, device, seed=1234)
    n_val = 2 if dry_run else val_batches_n
    val_set = [val_sampler.batch() for _ in range(n_val)]

    model = module.build_model(config, vocab_size=vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    max_steps = 2 if dry_run else config.get("max_steps", 10000)
    min_lr = config.get("min_lr", 1e-5)
    warmup_steps = 0 if dry_run else config.get("warmup_steps", 0)

    # ── Resume ────────────────────────────────────────────────────────────────
    ckpt = None
    start_step = 0
    if resume is not None:
        resume = Path(resume)
        if not resume.exists():
            raise ValueError(f"Checkpoint not found: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        old_config = ckpt.get("config", {})
        drift = [
            k for k in _ARCH_KEYS
            if k in old_config and k in config and old_config[k] != config[k]
        ]
        if drift:
            details = ", ".join(f"{k}: {old_config[k]} -> {config[k]}" for k in drift)
            raise ValueError(
                f"Cannot resume from {resume.name}: architecture changed since it was "
                f"saved ({details}). Train from scratch instead."
            )
        state_dict = ckpt["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        start_step = int(ckpt.get("step", 0))
        if start_step >= max_steps:
            raise ValueError(
                f"Checkpoint is already at step {start_step:,} but max_steps is "
                f"{max_steps:,}. Raise max_steps in config.json to train further."
            )
        print(f"Resuming from {resume.name} at step {start_step:,}")

    if not dry_run:
        if device == "cuda" and torch.cuda.get_device_capability()[0] < 7:
            compile_backend = "cudagraphs"
        else:
            compile_backend = "inductor"
        print(f"Compiling model with torch.compile (backend={compile_backend}, first step will be slow)...")
        model = torch.compile(model, backend=compile_backend)

    # Validation runs in eager on the uncompiled module: toggling train/eval on a
    # compiled model forces a second graph and repeated guard churn, and 20 eager
    # batches every eval_interval costs nothing by comparison.
    eval_model = getattr(model, "_orig_mod", model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 3e-4))

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

    if ckpt is not None:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            # Older checkpoints predate scheduler state — replay the schedule instead
            for _ in range(start_step):
                scheduler.step()

    eval_interval = config.get("eval_interval", 500)
    checkpoint_interval = config.get("checkpoint_interval", 1000)

    checkpoint_dir = MODELS_DIR / model_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = MODELS_DIR / model_name / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stats_file = runs_dir / f"run_{run_timestamp}.jsonl"

    # Which rows this run saw, kept beside the stats. The dataset on the Hub
    # changes as texts are added, so the stats alone do not say what was trained
    # on. This is a separate file because every line of the .jsonl is an eval
    # entry, which is what `report` expects to read.
    dataset_meta = {
        "repo_id": config.get("dataset_repo") or DEFAULT_REPO_ID,
        "revision": train_ds.revision,
        "requested_revision": config.get("dataset_revision"),
        "sources": train_ds.sources,
        "filters": config.get("dataset_filters"),
        "val_fraction": val_fraction,
        "split_seed": split_seed,
        "n_records": train_ds.n_records,
        "n_tokens": train_ds.n_tokens,
        "n_val_records": val_ds.n_records,
        "n_val_tokens": val_ds.n_tokens,
        "vocab_size": vocab_size,
    }
    meta_file = runs_dir / f"run_{run_timestamp}.meta.json"
    meta_file.write_text(json.dumps(dataset_meta, indent=2) + "\n")

    print(f"Stats log: {stats_file}")
    print(
        f"Training to step {max_steps}"
        f"{f' (from {start_step})' if start_step else ''}"
        f"{'  [DRY RUN]' if dry_run else ''}..."
    )

    model.train()
    step = start_step
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
        tokens_seen = step * batch_size * context_size
        # One epoch == one pass over every token of the train split
        epoch = tokens_seen / train_tokens
        elapsed = time.time() - train_start
        lr = scheduler.get_last_lr()[0]
        val_loss, val_accuracy = evaluate(eval_model, val_set, vocab_size)
        val_perplexity = math.exp(min(val_loss, 20))
        entry = {
            "step": step,
            "epoch": round(epoch, 3),
            "loss": round(avg_loss, 6),
            "perplexity": round(perplexity, 4),
            "accuracy": round(accuracy, 6),
            "val_loss": round(val_loss, 6),
            "val_perplexity": round(val_perplexity, 4),
            "val_accuracy": round(val_accuracy, 6),
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
            f"val_loss {val_loss:.4f} | val_ppl {val_perplexity:.2f} | val_acc {val_accuracy:.3f} | "
            f"lr {lr:.2e}"
        )
        return entry

    while step < max_steps:
        x, y = sampler.batch()

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
            _save_checkpoint(model, optimizer, scheduler, config, step, checkpoint_dir,
                             model_name, ema_state=ema_state, dataset_meta=dataset_meta)

    # Always flush remaining accumulated stats at the end
    if acc_steps > 0:
        flush_stats()

    if not dry_run:
        _save_checkpoint(model, optimizer, scheduler, config, step, checkpoint_dir,
                         model_name, ema_state=ema_state, dataset_meta=dataset_meta)
        print("Training complete.")
    else:
        print("Dry run complete.")


def _save_checkpoint(model, optimizer, scheduler, config, step, checkpoint_dir, model_name,
                     ema_state=None, dataset_meta=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = checkpoint_dir / f"{model_name}_{timestamp}_step{step}.pt"
    raw_model = getattr(model, "_orig_mod", model)  # unwrap torch.compile if present
    state_dict = ema_state if ema_state is not None else raw_model.state_dict()
    torch.save(
        {
            "step": step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "dataset": dataset_meta,
        },
        filename,
    )
    print(f"Checkpoint saved: {filename}")
