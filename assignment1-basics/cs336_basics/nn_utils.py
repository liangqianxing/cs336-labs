from collections.abc import Iterable

import torch


def cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    max_logits = torch.amax(inputs, dim=-1, keepdim=True)
    shifted = inputs - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    target_logits = torch.gather(
        shifted,
        dim=-1,
        index=targets.long().unsqueeze(-1),
    ).squeeze(-1)
    loss = log_sum_exp - target_logits
    return loss.mean()


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    total_norm_sq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for g in grads:
        total_norm_sq = total_norm_sq + torch.sum(g.detach().to(torch.float32) ** 2)
    total_norm = torch.sqrt(total_norm_sq)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.detach().mul_(scale.to(g.dtype))

    return total_norm
