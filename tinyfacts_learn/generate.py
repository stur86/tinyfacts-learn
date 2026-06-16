# generate.py
"""Autoregressive text generation for tinyfacts-learn models."""
import torch
import torch.nn as nn

from .tokenizers import WordTokenizer


def _sample_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    """Pick the next token ID from a logits vector."""
    if temperature == 0.0:
        return int(logits.argmax().item())
    logits = logits / temperature
    if top_k > 0:
        cutoff, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < cutoff[..., -1:]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_tokens(
    model: nn.Module,
    tokenizer: WordTokenizer,
    prompt: str,
    n_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    device: str = "cpu",
) -> tuple[str, list[str]]:
    """Generate tokens autoregressively from a prompt.

    Args:
        model: A trained GPT-style model with a ``_context_size`` attribute.
        tokenizer: WordTokenizer used to encode/decode text.
        prompt: Input text to condition generation on.
        n_tokens: Number of new tokens to generate.
        temperature: Softmax temperature (0 = greedy, >1 = more random).
        top_k: If > 0, restrict sampling to the top-k logits.
        device: Torch device string.

    Returns:
        (generated_text, generated_token_strings)
    """
    model.eval()
    context_size: int = model._context_size

    prompt_ids = tokenizer.tokenize(prompt)
    if not prompt_ids:
        raise ValueError("Prompt tokenized to an empty sequence.")

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    new_ids: list[int] = []

    with torch.no_grad():
        for _ in range(n_tokens):
            context = ids[:, -context_size:]          # crop to window
            logits = model(context)                    # (1, T, vocab_size)
            next_id = _sample_token(logits[0, -1, :], temperature, top_k)
            new_ids.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    generated_text = tokenizer.detokenize(new_ids)
    generated_token_strings = [tokenizer._tokens[i] for i in new_ids]
    return generated_text, generated_token_strings
