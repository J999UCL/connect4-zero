"""Batched random rollout evaluation for classical MCTS."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import BOARD_CELLS

DeviceLike = Optional[Union[str, torch.device]]


class RandomRolloutEvaluator:
    """Estimate state value by playing random games to termination in batch."""

    def __init__(
        self,
        rollout_batch_size: int = 128,
        device: DeviceLike = None,
        seed: Optional[int] = None,
        max_steps: int = BOARD_CELLS,
    ) -> None:
        if rollout_batch_size <= 0:
            raise ValueError("rollout_batch_size must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self.rollout_batch_size = int(rollout_batch_size)
        self.device = torch.device(device) if device is not None else None
        self.seed = seed
        self.max_steps = int(max_steps)
        self._generators: Dict[Tuple[str, int], torch.Generator] = {}

    def evaluate(self, state: Connect4x4x4Batch) -> float:
        """Return rollout value from ``state``'s player-to-move perspective."""
        self._validate_state(state)
        if bool(state.done[0].item()):
            return self._terminal_value(state)

        batch = self._make_rollout_batch(state)
        values = torch.zeros(self.rollout_batch_size, dtype=torch.float32, device=batch.device)
        player_sign = torch.ones(self.rollout_batch_size, dtype=torch.float32, device=batch.device)

        for _ in range(self.max_steps):
            if bool(batch.done.all().item()):
                return float(values.mean().item())

            legal_mask = batch.legal_mask()
            no_legal_actions = (~batch.done) & ~legal_mask.any(dim=1)
            if bool(no_legal_actions.any().item()):
                batch.done[no_legal_actions] = True
                values[no_legal_actions] = 0.0

            actions = self._sample_actions(legal_mask, device=batch.device)
            result = batch.step(actions)

            won = result.won
            draw = result.draw
            values[won] = player_sign[won]
            values[draw] = 0.0

            still_playing_after_move = result.legal & ~result.done
            player_sign[still_playing_after_move] *= -1.0

            if bool(batch.done.all().item()):
                return float(values.mean().item())

        raise RuntimeError("random rollouts exceeded max_steps before all games terminated")

    def _validate_state(self, state: Connect4x4x4Batch) -> None:
        if state.batch_size != 1:
            raise ValueError(f"rollout evaluator requires batch_size=1, got {state.batch_size}")

    def _terminal_value(self, state: Connect4x4x4Batch) -> float:
        if int(state.outcome[0].item()) == 1:
            return -1.0
        return 0.0

    def _make_rollout_batch(self, state: Connect4x4x4Batch) -> Connect4x4x4Batch:
        device = self.device if self.device is not None else state.device
        base = state.to(device)
        batch = Connect4x4x4Batch(self.rollout_batch_size, device=device)
        batch.board = base.board.expand(self.rollout_batch_size, -1, -1, -1).clone()
        batch.heights = base.heights.expand(self.rollout_batch_size, -1, -1).clone()
        batch.done = base.done.expand(self.rollout_batch_size).clone()
        batch.outcome = base.outcome.expand(self.rollout_batch_size).clone()
        return batch

    def _sample_actions(self, legal_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        generator = self._generator_for(device)
        scores = torch.rand(legal_mask.shape, device=device, generator=generator)
        scores = scores.masked_fill(~legal_mask, -1.0)
        return scores.argmax(dim=1).to(dtype=torch.long)

    def _generator_for(self, device: torch.device) -> Optional[torch.Generator]:
        if self.seed is None:
            return None
        if device.type not in ("cpu", "cuda"):
            return None

        key = (device.type, device.index if device.index is not None else -1)
        if key not in self._generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            self._generators[key] = generator
        return self._generators[key]


class BatchedRandomRolloutEvaluator:
    """Estimate values for many leaf states with chunked random rollouts."""

    def __init__(
        self,
        rollouts_per_state: int = 64,
        device: DeviceLike = None,
        seed: Optional[int] = None,
        max_steps: int = BOARD_CELLS,
        max_rollouts_per_chunk: int = 65536,
    ) -> None:
        if rollouts_per_state <= 0:
            raise ValueError("rollouts_per_state must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if max_rollouts_per_chunk <= 0:
            raise ValueError("max_rollouts_per_chunk must be positive")

        self.rollouts_per_state = int(rollouts_per_state)
        self.device = torch.device(device) if device is not None else None
        self.seed = seed
        self.max_steps = int(max_steps)
        self.max_rollouts_per_chunk = int(max_rollouts_per_chunk)
        self._generators: Dict[Tuple[str, int], torch.Generator] = {}

    def evaluate_batch(self, states: Connect4x4x4Batch) -> torch.Tensor:
        """Return one value per input state from its player-to-move perspective."""
        device = self.device if self.device is not None else states.device
        search_states = states.to(device)
        values = torch.empty(search_states.batch_size, dtype=torch.float32, device=device)

        max_states_per_chunk = max(1, self.max_rollouts_per_chunk // self.rollouts_per_state)
        for start in range(0, search_states.batch_size, max_states_per_chunk):
            end = min(start + max_states_per_chunk, search_states.batch_size)
            chunk = self._slice_states(search_states, start, end)
            values[start:end] = self._evaluate_chunk(chunk)

        return values

    def _evaluate_chunk(self, states: Connect4x4x4Batch) -> torch.Tensor:
        rollout_batch = self._make_rollout_batch(states)
        values = torch.zeros(rollout_batch.batch_size, dtype=torch.float32, device=rollout_batch.device)
        player_sign = torch.ones(rollout_batch.batch_size, dtype=torch.float32, device=rollout_batch.device)

        initial_previous_player_won = rollout_batch.done & rollout_batch.outcome.eq(1)
        values[initial_previous_player_won] = -1.0

        for _ in range(self.max_steps):
            if bool(rollout_batch.done.all().item()):
                return self._mean_values(values, states.batch_size)

            legal_mask = rollout_batch.legal_mask()
            no_legal_actions = (~rollout_batch.done) & ~legal_mask.any(dim=1)
            if bool(no_legal_actions.any().item()):
                rollout_batch.done[no_legal_actions] = True
                values[no_legal_actions] = 0.0

            actions = self._sample_actions(legal_mask, device=rollout_batch.device)
            result = rollout_batch.step(actions)

            won = result.won
            draw = result.draw
            values[won] = player_sign[won]
            values[draw] = 0.0

            still_playing_after_move = result.legal & ~result.done
            player_sign[still_playing_after_move] *= -1.0

        if bool(rollout_batch.done.all().item()):
            return self._mean_values(values, states.batch_size)

        raise RuntimeError("random rollouts exceeded max_steps before all games terminated")

    def _make_rollout_batch(self, states: Connect4x4x4Batch) -> Connect4x4x4Batch:
        repeats = self.rollouts_per_state
        batch = Connect4x4x4Batch(states.batch_size * repeats, device=states.device)
        batch.board = states.board.repeat_interleave(repeats, dim=0).clone()
        batch.heights = states.heights.repeat_interleave(repeats, dim=0).clone()
        batch.done = states.done.repeat_interleave(repeats, dim=0).clone()
        batch.outcome = states.outcome.repeat_interleave(repeats, dim=0).clone()
        return batch

    def _mean_values(self, rollout_values: torch.Tensor, state_count: int) -> torch.Tensor:
        return rollout_values.reshape(state_count, self.rollouts_per_state).mean(dim=1)

    def _sample_actions(self, legal_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        generator = self._generator_for(device)
        scores = torch.rand(legal_mask.shape, device=device, generator=generator)
        scores = scores.masked_fill(~legal_mask, -1.0)
        return scores.argmax(dim=1).to(dtype=torch.long)

    def _generator_for(self, device: torch.device) -> Optional[torch.Generator]:
        if self.seed is None:
            return None
        if device.type not in ("cpu", "cuda"):
            return None

        key = (device.type, device.index if device.index is not None else -1)
        if key not in self._generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            self._generators[key] = generator
        return self._generators[key]

    def _slice_states(
        self,
        states: Connect4x4x4Batch,
        start: int,
        end: int,
    ) -> Connect4x4x4Batch:
        sliced = Connect4x4x4Batch(end - start, device=states.device)
        sliced.board = states.board[start:end].clone()
        sliced.heights = states.heights[start:end].clone()
        sliced.done = states.done[start:end].clone()
        sliced.outcome = states.outcome[start:end].clone()
        return sliced
