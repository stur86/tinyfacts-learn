# export_onnx.py
"""Export trained tinyfacts-learn models to ONNX for browser inference.

Two artefacts are produced per checkpoint, both dropped into the web app's model
folder (``webapp/public/models`` by default):

* ``<checkpoint-stem>.onnx`` - the graph itself, exported with a *fixed*
  sequence length equal to the model's ``context_size``. Every architecture in
  this repo is causal, so a caller can right-pad a shorter prompt and read the
  logits at the last real position. A fixed length also keeps the export valid
  for models whose forward pass loops over the sequence (mamba's selective scan,
  TRM's recursion).
* ``<checkpoint-stem>.json`` - a metadata sidecar the web app reads to build its
  model dropdown.

``export_tokenizer`` writes ``tokenizer.json``, a self-contained dump of the
:class:`~tinyfacts_learn.tokenizers.WordTokenizer` vocabulary so the browser can
tokenize and detokenize without any Python.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from .tokenizers import WordTokenizer
from .train import MODELS_DIR, load_config, load_model_module

# Where the web app looks for the models it should serve.
WEBAPP_MODELS_DIR = Path(__file__).parent.parent / "webapp" / "public" / "models"

DEFAULT_OPSET = 20

# Separator used in the detokenize lookup keys ("<TAG>|base"). Neither tags nor
# base words can contain it, so keys stay unambiguous.
_KEY_SEP = "|"


def latest_checkpoint(model_name: str) -> Path:
    """Return the most recent checkpoint for a model, or raise."""
    checkpoint_dir = MODELS_DIR / model_name / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
    if not candidates:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    return candidates[-1]


def tokenizer_payload(tokenizer: WordTokenizer) -> dict:
    """Serialise everything the JS tokenizer needs to mirror the Python one."""
    word_map = tokenizer._word_forms_dict._word_map

    word_tokens: dict[str, list[int]] = {
        form: [tokenizer._token_to_id[t] for t in tokenizer._word_forms_dict.get_tokens(form)]
        for form in word_map
    }

    # Detokenize lookup: (tag, base) -> surface form. Mirrors WordTokenizer's
    # _base_tag_to_id, including its "last one wins" behaviour on collisions.
    form_lookup: dict[str, str] = {
        f"{tagged.tag or ''}{_KEY_SEP}{tagged.base}": form
        for form, tagged in word_map.items()
    }

    return {
        "tokens": tokenizer._tokens,
        "wordTokens": word_tokens,
        "formLookup": form_lookup,
        "tags": tokenizer._tags,
        "specials": tokenizer._allowed_special,
        "digits": tokenizer._digits,
        "unkId": tokenizer._token_to_id["<UNK>"],
        "upcId": tokenizer._token_to_id.get("<UPC>"),
        "ignoreCase": tokenizer._ignore_case,
        "vocabSize": tokenizer.vocab_size,
    }


def export_tokenizer(out_dir: Path = WEBAPP_MODELS_DIR) -> Path:
    """Write ``tokenizer.json`` next to the exported models."""
    tokenizer = WordTokenizer(ignore_case=True, digits=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tokenizer.json"
    out_path.write_text(json.dumps(tokenizer_payload(tokenizer)))
    return out_path


def export_model(
    model_name: str,
    checkpoint: Optional[Path] = None,
    out_dir: Path = WEBAPP_MODELS_DIR,
    opset: int = DEFAULT_OPSET,
) -> tuple[Path, Path]:
    """Export one checkpoint to ONNX plus its metadata sidecar.

    Returns:
        (onnx_path, metadata_path)
    """
    if checkpoint is None:
        checkpoint = latest_checkpoint(model_name)
    if not checkpoint.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state.get("config") or load_config(model_name)
    context_size = int(config["context_size"])

    tokenizer = WordTokenizer(ignore_case=True, digits=True)
    module = load_model_module(model_name)
    model = module.build_model(config, vocab_size=tokenizer.vocab_size)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint.stem
    onnx_path = out_dir / f"{stem}.onnx"
    meta_path = out_dir / f"{stem}.json"

    dummy = torch.zeros(1, context_size, dtype=torch.long)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            input_names=["input_ids"],
            output_names=["logits"],
            opset_version=opset,
            dynamo=True,
            # Keep the graph and its weights in a single file: the web app
            # fetches one URL per model, with no sidecar .onnx.data to track.
            external_data=False,
        )

    meta = {
        "id": stem,
        "model": model_name,
        "file": onnx_path.name,
        "checkpoint": checkpoint.name,
        "step": state.get("step"),
        "contextSize": context_size,
        "vocabSize": tokenizer.vocab_size,
        "params": sum(p.numel() for p in model.parameters()),
        "sizeBytes": onnx_path.stat().st_size,
        "exportedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    return onnx_path, meta_path
