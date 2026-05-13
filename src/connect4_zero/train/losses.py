"""Loss functions for AlphaZero policy/value training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AlphaZeroLossOutput:
    """Named loss components for logging."""

    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor


class AlphaZeroLoss(nn.Module):
    """Policy cross-entropy plus scalar value MSE."""

    def __init__(self, policy_weight: float = 1.0, value_weight: float = 1.0) -> None:
        super().__init__()
        if policy_weight <= 0:
            raise ValueError("policy_weight must be positive")
        if value_weight <= 0:
            raise ValueError("value_weight must be positive")
        self.policy_weight = float(policy_weight)
        self.value_weight = float(value_weight)

    def forward(
        self,
        policy_logits: torch.Tensor,
        values: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> AlphaZeroLossOutput:
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.to(dtype=torch.bool), -1e9)

        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(target_policy.to(dtype=torch.float32) * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(values.flatten(), target_value.to(dtype=torch.float32).flatten())
        total = self.policy_weight * policy_loss + self.value_weight * value_loss
        return AlphaZeroLossOutput(total=total, policy=policy_loss, value=value_loss)
