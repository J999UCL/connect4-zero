from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from c4az.network import AlphaZeroNet, NetworkConfig


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0


class AlphaZeroLoss(nn.Module):
    def forward(
        self,
        policy_logits: torch.Tensor,
        values: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        log_probs = torch.log_softmax(policy_logits, dim=1)
        policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
        value_loss = torch.mean((values - target_value) ** 2)
        loss = policy_loss + value_loss
        return loss, {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
        }


def train_step(
    model: AlphaZeroNet,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    loss_fn: AlphaZeroLoss,
    *,
    device: torch.device | str = "cpu",
    grad_clip_norm: float = 5.0,
) -> dict[str, float]:
    model.train()
    inputs = batch["input"].to(device)
    target_policy = batch["policy"].to(device)
    target_value = batch["value"].to(device)
    optimizer.zero_grad(set_to_none=True)
    logits, values = model(inputs)
    loss, metrics = loss_fn(logits, values, target_policy, target_value)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    metrics["grad_norm"] = float(grad_norm.detach().cpu())
    return metrics


def save_checkpoint(
    path: Path,
    model: AlphaZeroNet,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    step: int = 0,
    epoch: int = 0,
    metrics: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "epoch": epoch,
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: AlphaZeroNet | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: torch.device | str = "cpu",
) -> tuple[AlphaZeroNet, dict]:
    payload = torch.load(path, map_location=map_location)
    config = NetworkConfig(**payload["model_config"])
    loaded_model = model or AlphaZeroNet(config)
    loaded_model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return loaded_model, payload
