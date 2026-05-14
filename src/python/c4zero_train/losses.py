from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: float
    policy: float
    value: float
    l2_regularization: float = 0.0
    paper_total_loss: float = 0.0
    optimized_total: float = 0.0


def alpha_zero_loss(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
) -> tuple[torch.Tensor, LossBreakdown]:
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(values, target_value)
    total = policy_loss + value_loss
    total_value = float(total.detach().cpu())
    return total, LossBreakdown(
        total=total_value,
        policy=float(policy_loss.detach().cpu()),
        value=float(value_loss.detach().cpu()),
        paper_total_loss=total_value,
        optimized_total=total_value,
    )
