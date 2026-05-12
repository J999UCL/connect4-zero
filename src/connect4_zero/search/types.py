"""Public dataclasses and protocols for MCTS search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Union

import torch

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE, BOARD_CELLS

DeviceLike = Optional[Union[str, torch.device]]


class Evaluator(Protocol):
    """Evaluate a non-terminal state from the player-to-move perspective."""

    def evaluate(self, state: Connect4x4x4Batch) -> float:
        """Return a scalar value in ``[-1, 1]``."""


@dataclass(frozen=True)
class MCTSConfig:
    """Configuration for single-root MCTS."""

    num_simulations: int = 100
    exploration_constant: float = 1.4
    rollout_batch_size: int = 128
    rollout_device: DeviceLike = None
    seed: Optional[int] = None
    max_rollout_steps: int = BOARD_CELLS

    def __post_init__(self) -> None:
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if self.exploration_constant < 0:
            raise ValueError("exploration_constant must be non-negative")
        if self.rollout_batch_size <= 0:
            raise ValueError("rollout_batch_size must be positive")
        if self.max_rollout_steps <= 0:
            raise ValueError("max_rollout_steps must be positive")


@dataclass(frozen=True)
class SearchResult:
    """Search statistics projected onto the 16-action policy space."""

    visit_counts: torch.Tensor
    policy: torch.Tensor
    q_values: torch.Tensor
    root_value: float


@dataclass(frozen=True)
class BatchedRootActionConfig:
    """Configuration for batched root/action search.

    Each root action evaluation is one MCTS-style leaf evaluation whose value is
    estimated by ``rollouts_per_leaf`` random continuations.
    """

    num_selection_waves: int = 8
    leaves_per_root: int = 4
    rollouts_per_leaf: int = 64
    exploration_constant: float = 1.4
    policy_temperature: float = 1.0
    rollout_device: DeviceLike = None
    seed: Optional[int] = None
    max_rollout_steps: int = BOARD_CELLS
    max_rollouts_per_chunk: int = 65536
    evaluate_all_actions_first: bool = True

    def __post_init__(self) -> None:
        if self.num_selection_waves < 0:
            raise ValueError("num_selection_waves must be non-negative")
        if self.leaves_per_root <= 0:
            raise ValueError("leaves_per_root must be positive")
        if self.leaves_per_root > ACTION_SIZE:
            raise ValueError(f"leaves_per_root must be <= {ACTION_SIZE}")
        if self.rollouts_per_leaf <= 0:
            raise ValueError("rollouts_per_leaf must be positive")
        if self.exploration_constant < 0:
            raise ValueError("exploration_constant must be non-negative")
        if self.policy_temperature <= 0:
            raise ValueError("policy_temperature must be positive")
        if self.max_rollout_steps <= 0:
            raise ValueError("max_rollout_steps must be positive")
        if self.max_rollouts_per_chunk <= 0:
            raise ValueError("max_rollouts_per_chunk must be positive")


@dataclass(frozen=True)
class BatchedSearchResult:
    """Search statistics for a batch of root states."""

    visit_counts: torch.Tensor
    policy: torch.Tensor
    q_values: torch.Tensor
    root_values: torch.Tensor
