"""Typed result objects for game-engine operations."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StepResult:
    """Post-step status for each game in a batch.

    ``legal``, ``won``, and ``draw`` describe the attempted move. ``done`` and
    ``outcome`` are snapshots of the engine state after the step has completed.
    ``outcome == 1`` means the player who just moved won; ``outcome == 0``
    means the game is unfinished or drawn.
    """

    legal: torch.Tensor
    won: torch.Tensor
    draw: torch.Tensor
    done: torch.Tensor
    outcome: torch.Tensor
