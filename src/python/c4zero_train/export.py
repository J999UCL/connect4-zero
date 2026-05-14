from __future__ import annotations

from pathlib import Path

import torch

from c4zero_train.checkpoint import load_checkpoint
from c4zero_train.model import AlphaZeroNet


def export_torchscript_model(
    model: AlphaZeroNet,
    output_path: str | Path,
    device: torch.device | str = "cpu",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    model.eval()
    example = torch.zeros((1, 2, 4, 4, 4), dtype=torch.float32, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(model, example, strict=True)
        traced.save(str(output_path))


def export_checkpoint(
    checkpoint_dir: str | Path,
    output_path: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    model, _payload = load_checkpoint(checkpoint_dir, device=device)
    path = Path(output_path) if output_path is not None else checkpoint_dir / "inference.ts"
    export_torchscript_model(model, path, device=device)
    return path
