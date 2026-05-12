"""Shared data structures for self-play training samples."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from connect4_zero.game.constants import ACTION_SIZE, BOARD_SIZE


@dataclass(frozen=True)
class SelfPlaySamples:
    """A contiguous batch of AlphaZero-style training samples."""

    boards: torch.Tensor
    policies: torch.Tensor
    values: torch.Tensor
    visit_counts: torch.Tensor
    q_values: torch.Tensor
    legal_masks: torch.Tensor
    actions: torch.Tensor
    plies: torch.Tensor

    @property
    def num_samples(self) -> int:
        return int(self.boards.shape[0])

    def validate(self) -> None:
        n = self.num_samples
        expected_board_shape = (n, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE)
        if tuple(self.boards.shape) != expected_board_shape:
            raise ValueError(f"boards must have shape {expected_board_shape}, got {tuple(self.boards.shape)}")
        for name, tensor in (
            ("policies", self.policies),
            ("visit_counts", self.visit_counts),
            ("q_values", self.q_values),
            ("legal_masks", self.legal_masks),
        ):
            expected = (n, ACTION_SIZE)
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        for name, tensor in (("values", self.values), ("actions", self.actions), ("plies", self.plies)):
            expected = (n,)
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")

    def slice(self, start: int, end: int) -> "SelfPlaySamples":
        return SelfPlaySamples(
            boards=self.boards[start:end],
            policies=self.policies[start:end],
            values=self.values[start:end],
            visit_counts=self.visit_counts[start:end],
            q_values=self.q_values[start:end],
            legal_masks=self.legal_masks[start:end],
            actions=self.actions[start:end],
            plies=self.plies[start:end],
        )
