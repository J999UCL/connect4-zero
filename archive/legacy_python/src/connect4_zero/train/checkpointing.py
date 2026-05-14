"""Training checkpoint naming helpers."""

from __future__ import annotations

from pathlib import Path


def default_checkpoint_path(output_dir: Path | str, step: int) -> Path:
    """Return the standard checkpoint path for a training step."""
    return Path(output_dir) / "checkpoints" / f"checkpoint-step-{int(step):08d}.pt"
