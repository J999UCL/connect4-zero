from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: float
    policy: float
    value: float


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
    return total, LossBreakdown(
        total=float(total.detach().cpu()),
        policy=float(policy_loss.detach().cpu()),
        value=float(value_loss.detach().cpu()),
    )
