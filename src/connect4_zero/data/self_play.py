"""Batched self-play sample generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from connect4_zero.data.types import SelfPlaySamples
from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE, BOARD_CELLS, CURRENT_PLAYER, OPPONENT_PLAYER
from connect4_zero.search import BatchedRootActionMCTS
from connect4_zero.search.types import BatchedSearchResult, DeviceLike


@dataclass(frozen=True)
class SelfPlayConfig:
    """Configuration for batched self-play data generation."""

    batch_size: int = 128
    device: DeviceLike = None
    action_temperature: float = 1.0
    max_plies: int = BOARD_CELLS
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.action_temperature < 0:
            raise ValueError("action_temperature must be non-negative")
        if self.max_plies <= 0:
            raise ValueError("max_plies must be positive")


class SelfPlayGenerator:
    """Generate AlphaZero-style samples from batched self-play games."""

    def __init__(
        self,
        search: BatchedRootActionMCTS,
        config: Optional[SelfPlayConfig] = None,
    ) -> None:
        self.search = search
        self.config = config if config is not None else SelfPlayConfig()
        self.device = torch.device("cpu" if self.config.device is None else self.config.device)
        self._generators: Dict[Tuple[str, int], torch.Generator] = {}

    def generate(self, num_games: int) -> SelfPlaySamples:
        """Generate samples from ``num_games`` complete self-play games."""
        if num_games <= 0:
            raise ValueError("num_games must be positive")

        batches: List[SelfPlaySamples] = []
        remaining = int(num_games)
        while remaining > 0:
            batch_games = min(remaining, self.config.batch_size)
            batches.append(self._generate_batch(batch_games))
            remaining -= batch_games
        return self._concat_samples(batches)

    def _generate_batch(self, batch_games: int) -> SelfPlaySamples:
        game = Connect4x4x4Batch(batch_games, device=self.device)
        player_to_move = torch.full(
            (batch_games,),
            CURRENT_PLAYER,
            dtype=torch.int8,
            device=self.device,
        )
        winner_by_game = torch.zeros(batch_games, dtype=torch.int8, device=self.device)

        boards: List[torch.Tensor] = []
        policies: List[torch.Tensor] = []
        visit_counts: List[torch.Tensor] = []
        q_values: List[torch.Tensor] = []
        legal_masks: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        plies: List[torch.Tensor] = []
        record_games: List[torch.Tensor] = []
        record_players: List[torch.Tensor] = []

        for ply in range(self.config.max_plies):
            active = ~game.done
            if bool(active.logical_not().all().item()):
                return self._finalize_samples(
                    boards,
                    policies,
                    visit_counts,
                    q_values,
                    legal_masks,
                    actions,
                    plies,
                    record_games,
                    record_players,
                    winner_by_game,
                )

            active_indices = active.nonzero(as_tuple=False).flatten()
            roots = self._index_batch(game, active_indices)
            result = self.search.search_batch(roots)
            root_legal_mask = roots.legal_mask()
            chosen_actions = self._select_actions(result, root_legal_mask)

            boards.append(roots.board.detach().cpu().clone())
            policies.append(result.policy.detach().cpu().clone())
            visit_counts.append(result.visit_counts.detach().cpu().round().to(dtype=torch.int32))
            q_values.append(result.q_values.detach().cpu().clone())
            legal_masks.append(root_legal_mask.detach().cpu().clone())
            actions.append(chosen_actions.detach().cpu().to(dtype=torch.uint8))
            plies.append(torch.full((active_indices.numel(),), ply, dtype=torch.int16))
            record_games.append(active_indices.detach().cpu().to(dtype=torch.long))
            current_players = player_to_move[active_indices]
            record_players.append(current_players.detach().cpu().clone())

            action_batch = torch.zeros(batch_games, dtype=torch.long, device=self.device)
            action_batch[active_indices] = chosen_actions.to(device=self.device, dtype=torch.long)
            step = game.step(action_batch)

            active_won = step.won[active_indices]
            active_draw = step.draw[active_indices]
            active_done = step.done[active_indices]
            if bool(active_won.any().item()):
                winner_by_game[active_indices[active_won]] = current_players[active_won]
            if bool(active_draw.any().item()):
                winner_by_game[active_indices[active_draw]] = 0

            still_playing = ~active_done
            if bool(still_playing.any().item()):
                player_to_move[active_indices[still_playing]] *= OPPONENT_PLAYER

        if bool(game.done.all().item()):
            return self._finalize_samples(
                boards,
                policies,
                visit_counts,
                q_values,
                legal_masks,
                actions,
                plies,
                record_games,
                record_players,
                winner_by_game,
            )

        raise RuntimeError("self-play exceeded max_plies before all games terminated")

    def _select_actions(
        self,
        result: BatchedSearchResult,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        policy = result.policy.to(device=legal_mask.device)
        legal_policy = policy.masked_fill(~legal_mask, 0.0)
        fallback = legal_mask.to(dtype=torch.float32)
        bad_rows = legal_policy.sum(dim=1).eq(0)
        if bool(bad_rows.any().item()):
            legal_policy[bad_rows] = fallback[bad_rows]

        if self.config.action_temperature == 0:
            return legal_policy.argmax(dim=1).to(dtype=torch.long)

        weights = legal_policy.pow(1.0 / self.config.action_temperature)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        generator = self._generator_for(weights.device)
        return torch.multinomial(weights, num_samples=1, generator=generator).flatten().to(dtype=torch.long)

    def _finalize_samples(
        self,
        boards: List[torch.Tensor],
        policies: List[torch.Tensor],
        visit_counts: List[torch.Tensor],
        q_values: List[torch.Tensor],
        legal_masks: List[torch.Tensor],
        actions: List[torch.Tensor],
        plies: List[torch.Tensor],
        record_games: List[torch.Tensor],
        record_players: List[torch.Tensor],
        winner_by_game: torch.Tensor,
    ) -> SelfPlaySamples:
        record_game_tensor = torch.cat(record_games)
        record_player_tensor = torch.cat(record_players).to(dtype=torch.int8)
        winners = winner_by_game.detach().cpu()[record_game_tensor]
        values = torch.zeros(record_game_tensor.shape[0], dtype=torch.int8)
        decided = winners.ne(0)
        values[decided] = torch.where(
            record_player_tensor[decided].eq(winners[decided]),
            torch.ones_like(values[decided]),
            -torch.ones_like(values[decided]),
        )

        samples = SelfPlaySamples(
            boards=torch.cat(boards).to(dtype=torch.int8),
            policies=torch.cat(policies).to(dtype=torch.float32),
            values=values,
            visit_counts=torch.cat(visit_counts).to(dtype=torch.int32),
            q_values=torch.cat(q_values).to(dtype=torch.float32),
            legal_masks=torch.cat(legal_masks).to(dtype=torch.bool),
            actions=torch.cat(actions).to(dtype=torch.uint8),
            plies=torch.cat(plies).to(dtype=torch.int16),
        )
        samples.validate()
        return samples

    def _concat_samples(self, batches: List[SelfPlaySamples]) -> SelfPlaySamples:
        samples = SelfPlaySamples(
            boards=torch.cat([batch.boards for batch in batches]),
            policies=torch.cat([batch.policies for batch in batches]),
            values=torch.cat([batch.values for batch in batches]),
            visit_counts=torch.cat([batch.visit_counts for batch in batches]),
            q_values=torch.cat([batch.q_values for batch in batches]),
            legal_masks=torch.cat([batch.legal_masks for batch in batches]),
            actions=torch.cat([batch.actions for batch in batches]),
            plies=torch.cat([batch.plies for batch in batches]),
        )
        samples.validate()
        return samples

    def _index_batch(
        self,
        batch: Connect4x4x4Batch,
        indices: torch.Tensor,
    ) -> Connect4x4x4Batch:
        selected = Connect4x4x4Batch(indices.numel(), device=batch.device)
        selected.board = batch.board[indices].clone()
        selected.heights = batch.heights[indices].clone()
        selected.done = batch.done[indices].clone()
        selected.outcome = batch.outcome[indices].clone()
        return selected

    def _generator_for(self, device: torch.device) -> Optional[torch.Generator]:
        if self.config.seed is None:
            return None
        if device.type not in ("cpu", "cuda"):
            return None

        key = (device.type, device.index if device.index is not None else -1)
        if key not in self._generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.config.seed)
            self._generators[key] = generator
        return self._generators[key]
