from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch

from c4zero_tools.version import current_version_info
from c4zero_train.model import AlphaZeroNet, ModelConfig


VERSION_KEYS_TO_VALIDATE = (
    "checkpoint_schema_version",
    "model_config_version",
    "encoder_version",
    "game_rules_version",
    "action_mapping_version",
)


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


def validate_checkpoint_metadata(directory: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"checkpoint is missing metadata.json: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    current_version = current_version_info()
    checkpoint_version = metadata.get("version", {})
    for key in VERSION_KEYS_TO_VALIDATE:
        if checkpoint_version.get(key) != current_version[key]:
            raise ValueError(f"{key} mismatch: {checkpoint_version.get(key)} != {current_version[key]}")
    config = ModelConfig(**payload["config"])
    if metadata.get("model_config_hash") != config.stable_hash():
        raise ValueError("checkpoint model config hash mismatch")
    if metadata.get("model_config") != payload["config"]:
        raise ValueError("checkpoint model config metadata does not match payload")
    return metadata


def load_checkpoint(directory: str | Path, device: torch.device | str = "cpu") -> tuple[AlphaZeroNet, dict[str, Any]]:
    directory = Path(directory)
    payload = torch.load(directory / "model_state.pt", map_location=device)
    metadata = validate_checkpoint_metadata(directory, payload)
    config = ModelConfig(**payload["config"])
    model = AlphaZeroNet(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    payload["metadata"] = metadata
    return model, payload


def restore_optimizer_and_scheduler(
    payload: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> None:
    optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
