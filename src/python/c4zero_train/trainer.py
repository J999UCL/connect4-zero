from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch

from c4zero_train.encoding import encode_samples
from c4zero_train.losses import LossBreakdown, alpha_zero_loss
from c4zero_train.model import AlphaZeroNet
from c4zero_train.replay import ReplayBuffer


@dataclass(frozen=True, slots=True)
class TrainConfig:
    batch_size: int
    learning_rate: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 1e-4
    seed: int = 1
    policy_weight: float = 1.0
    value_weight: float = 1.0
    augment_symmetries: bool = False


def make_optimizer(model: AlphaZeroNet, config: TrainConfig) -> torch.optim.SGD:
    return torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def make_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.MultiStepLR:
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100_000, 300_000, 500_000], gamma=0.1)


def batch_targets(samples, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    policy = torch.as_tensor(np.stack([sample.policy for sample in samples]), dtype=torch.float32, device=device)
    value = torch.as_tensor(np.array([sample.value for sample in samples], dtype=np.float32), dtype=torch.float32, device=device)
    return policy, value


def l2_regularization_value(model: AlphaZeroNet, optimizer: torch.optim.Optimizer) -> float:
    weight_decay = max(float(group.get("weight_decay", 0.0)) for group in optimizer.param_groups)
    if weight_decay == 0.0:
        return 0.0
    total = torch.zeros((), dtype=torch.float32, device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.requires_grad:
            total = total + parameter.detach().pow(2).sum()
    return float((weight_decay * total).cpu())


def train_step(
    model: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    samples,
    device: torch.device | str = "cpu",
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
) -> LossBreakdown:
    model.train()
    inputs = encode_samples(samples, device=device)
    target_policy, target_value = batch_targets(samples, device=device)
    optimizer.zero_grad(set_to_none=True)
    policy_logits, values = model(inputs)
    loss, breakdown = alpha_zero_loss(
        policy_logits,
        values,
        target_policy,
        target_value,
        policy_weight=policy_weight,
        value_weight=value_weight,
    )
    l2_value = l2_regularization_value(model, optimizer)
    loss.backward()
    optimizer.step()
    return LossBreakdown(
        total=breakdown.total,
        policy=breakdown.policy,
        value=breakdown.value,
        l2_regularization=l2_value,
        paper_total_loss=breakdown.total + l2_value,
        optimized_total=breakdown.total,
    )


def train_steps(
    model: AlphaZeroNet,
    replay: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: TrainConfig,
    steps: int,
    device: torch.device | str = "cpu",
) -> list[LossBreakdown]:
    rng = random.Random(config.seed)
    losses: list[LossBreakdown] = []
    for _ in range(steps):
        samples = replay.sample_batch(config.batch_size, rng, augment_symmetries=config.augment_symmetries)
        losses.append(
            train_step(
                model,
                optimizer,
                samples,
                device=device,
                policy_weight=config.policy_weight,
                value_weight=config.value_weight,
            )
        )
        if scheduler is not None:
            scheduler.step()
    return losses
