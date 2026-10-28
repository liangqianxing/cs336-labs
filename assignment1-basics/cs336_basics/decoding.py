from __future__ import annotations

import torch

from cs336_basics.activations import softmax
from cs336_basics.transformer import TransformerLM


@torch.no_grad()
def top_p_filtering(probs: torch.Tensor, top_p: float = 1.0) -> torch.Tensor:
    if top_p >= 1.0:
        return probs
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff_mask = (cumulative_probs - sorted_probs) >= top_p
    sorted_probs = sorted_probs.masked_fill(cutoff_mask, 0.0)

    filtered = torch.zeros_like(probs)
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
    normalizer = filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return filtered / normalizer


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    was_training = model.training
    model.eval()

    squeeze_batch = False
    if prompt_ids.ndim == 1:
        seq = prompt_ids.unsqueeze(0).clone()
        squeeze_batch = True
    else:
        seq = prompt_ids.clone()

    for _ in range(max_new_tokens):
        cond = seq[:, -model.context_length :]
        logits = model(cond)[:, -1, :]

        if temperature <= 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            scaled_logits = logits / temperature
            probs = softmax(scaled_logits, dim=-1)
            probs = top_p_filtering(probs, top_p=top_p)
            next_token = torch.multinomial(probs, num_samples=1)

        seq = torch.cat([seq, next_token], dim=-1)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

    if was_training:
        model.train()

    return seq.squeeze(0) if squeeze_batch else seq
