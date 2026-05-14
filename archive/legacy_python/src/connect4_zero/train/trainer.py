"""Minimal train-step utilities for the ResNet policy/value model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.optim import Optimizer

from connect4_zero.train.losses import AlphaZeroLoss


@dataclass(frozen=True)
class TrainerConfig:
    """Defaults for the first ResNet training script."""

    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    num_workers: int = 2
    amp: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


def train_step(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    optimizer: Optimizer,
    loss_fn: AlphaZeroLoss,
    device: torch.device,
    config: TrainerConfig | None = None,
) -> dict[str, float]:
    """Run one supervised AlphaZero update and return scalar metrics."""
    cfg = config if config is not None else TrainerConfig()
    model.train()
    inputs = batch["input"].to(device=device, dtype=torch.float32)
    target_policy = batch["policy"].to(device=device, dtype=torch.float32)
    target_value = batch["value"].to(device=device, dtype=torch.float32)
    legal_mask = batch.get("legal_mask")
    if legal_mask is not None:
        legal_mask = legal_mask.to(device=device, dtype=torch.bool)

    optimizer.zero_grad(set_to_none=True)
    policy_logits, values = model(inputs)
    losses = loss_fn(policy_logits, values, target_policy, target_value, legal_mask=legal_mask)
    losses.total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
    optimizer.step()

    return {
        "loss": float(losses.total.detach().cpu().item()),
        "policy_loss": float(losses.policy.detach().cpu().item()),
        "value_loss": float(losses.value.detach().cpu().item()),
        "grad_norm": float(grad_norm.detach().cpu().item()),
    }
