"""Checkpoint helpers for policy/value networks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from connect4_zero.model.resnet3d import Connect4ResNet3D, ResNet3DConfig


@dataclass(frozen=True)
class CheckpointState:
    """Loaded checkpoint metadata and model state."""

    model: Connect4ResNet3D
    step: int
    epoch: int
    metrics: dict[str, float]


def save_checkpoint(
    path: Path | str,
    model: Connect4ResNet3D,
    optimizer: Optional[Optimizer] = None,
    step: int = 0,
    epoch: int = 0,
    metrics: Optional[dict[str, float]] = None,
) -> None:
    """Save model config, weights, optional optimizer state, and metadata."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "connect4_zero.resnet3d.v1",
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, target)


def load_checkpoint(
    path: Path | str,
    model: Optional[Connect4ResNet3D] = None,
    optimizer: Optional[Optimizer] = None,
    map_location: str | torch.device | None = None,
) -> CheckpointState:
    """Load a checkpoint and optionally restore an existing model/optimizer."""
    payload = torch.load(Path(path), map_location=map_location)
    if payload.get("format") != "connect4_zero.resnet3d.v1":
        raise ValueError(f"unsupported checkpoint format: {payload.get('format')}")

    config = ResNet3DConfig.from_dict(payload["model_config"])
    loaded_model = model if model is not None else Connect4ResNet3D(config)
    if loaded_model.config != config:
        raise ValueError("checkpoint model config does not match provided model")
    loaded_model.load_state_dict(payload["model_state"])

    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    return CheckpointState(
        model=loaded_model,
        step=int(payload.get("step", 0)),
        epoch=int(payload.get("epoch", 0)),
        metrics={key: float(value) for key, value in payload.get("metrics", {}).items()},
    )
