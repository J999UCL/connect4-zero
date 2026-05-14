from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch

from c4zero_tools.version import current_version_info
from c4zero_train.model import AlphaZeroNet, ModelConfig


def save_checkpoint(
    directory: str | Path,
    model: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    epoch: int,
    replay_manifests: list[str],
    metrics: dict[str, Any] | None = None,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "epoch": epoch,
        "config": asdict(model.config),
    }
    torch.save(payload, directory / "model_state.pt")
    metadata = {
        "version": current_version_info(),
        "model_config": asdict(model.config),
        "model_config_hash": model.config.stable_hash(),
        "step": step,
        "epoch": epoch,
        "replay_manifests": replay_manifests,
        "metrics": metrics or {},
        "export_schema": {
            "input": "float32[B,2,4,4,4]",
            "policy_logits": "float32[B,16]",
            "value": "float32[B]",
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def load_checkpoint(directory: str | Path, device: torch.device | str = "cpu") -> tuple[AlphaZeroNet, dict[str, Any]]:
    directory = Path(directory)
    payload = torch.load(directory / "model_state.pt", map_location=device)
    config = ModelConfig(**payload["config"])
    model = AlphaZeroNet(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
