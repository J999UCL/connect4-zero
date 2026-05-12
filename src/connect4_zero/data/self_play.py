"""Batched self-play sample generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from connect4_zero.data.types import SelfPlaySamples
from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE, BOARD_CELLS, CURRENT_PLAYER, OPPONENT_PLAYER
from connect4_zero.search.types import BatchedSearch, BatchedSearchResult, DeviceLike

ProgressCallback = Callable[[str, Dict[str, float | int | str]], None]


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
        search: BatchedSearch,
        config: Optional[SelfPlayConfig] = None,
    ) -> None:
        self.search = search
        self.config = config if config is not None else SelfPlayConfig()
        self.device = torch.device("cpu" if self.config.device is None else self.config.device)
        self._generators: Dict[Tuple[str, int], torch.Generator] = {}

    def generate(self, num_games: int) -> SelfPlaySamples:
        """Generate samples from ``num_games`` complete self-play games."""
        return self.generate_with_progress(num_games=num_games, progress_callback=None)

    def generate_with_progress(
        self,
        num_games: int,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SelfPlaySamples:
        """Generate samples and emit optional structured progress events."""
        if num_games <= 0:
            raise ValueError("num_games must be positive")

        batches: List[SelfPlaySamples] = []
        remaining = int(num_games)
        batch_index = 0
        games_completed = 0
        while remaining > 0:
            batch_games = min(remaining, self.config.batch_size)
            self._emit(
                progress_callback,
                "batch_start",
                {
                    "batch_index": batch_index,
                    "batch_games": batch_games,
                    "games_completed": games_completed,
                    "games_total": num_games,
                },
            )
            batch = self._generate_batch(batch_games, progress_callback=progress_callback)
            batches.append(batch)
            games_completed += batch_games
            remaining -= batch_games
            self._emit(
                progress_callback,
                "batch_end",
                {
                    "batch_index": batch_index,
                    "batch_games": batch_games,
                    "batch_samples": batch.num_samples,
                    "games_completed": games_completed,
                    "games_total": num_games,
                },
            )
            batch_index += 1
        return self._concat_samples(batches)

    def _generate_batch(
        self,
        batch_games: int,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SelfPlaySamples:
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
        trees_by_game: Optional[List[Any]] = [None for _ in range(batch_games)] if self._supports_tree_reuse() else None

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
            root_trees = self._active_trees(trees_by_game, active_indices)
            active_count = int(active_indices.numel())
            self._emit(
                progress_callback,
                "ply_search_start",
                {
                    "ply": ply,
                    "active_games": active_count,
                    "finished_games": batch_games - active_count,
                },
            )
            result = self._search_roots(roots, root_trees)
            self._emit(
                progress_callback,
                "ply_search_end",
                {
                    "ply": ply,
                    "active_games": active_count,
                    "mean_root_value": float(result.root_values.mean().detach().cpu().item()),
                    "mean_policy_entropy": self._policy_entropy(result.policy),
                    "total_visits": int(result.visit_counts.sum().detach().cpu().item()),
                    "leaf_evaluations": int(getattr(self.search, "last_leaf_evaluations", -1)),
                    "terminal_evaluations": int(getattr(self.search, "last_terminal_evaluations", -1)),
                    "leaf_batches": len(getattr(self.search, "last_leaf_batch_sizes", [])),
                    "max_leaf_batch": self._max_search_leaf_batch(),
                    "tree_reuse_hits": int(getattr(self.search, "last_reused_roots", -1)),
                    "tree_fresh_roots": int(getattr(self.search, "last_fresh_roots", -1)),
                    "expanded_children": int(getattr(self.search, "last_expanded_children", -1)),
                    "expansion_batches": len(getattr(self.search, "last_expansion_batch_sizes", [])),
                    "max_expansion_batch": self._max_search_expansion_batch(),
                    "timing_prepare": self._search_timing("prepare_trees"),
                    "timing_select": self._search_timing("select"),
                    "timing_expand": self._search_timing("expand"),
                    "timing_rollout": self._search_timing("rollout_eval"),
                    "timing_backprop": self._search_timing("backprop"),
                    "timing_build": self._search_timing("build_result"),
                },
            )
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
            self._advance_trees(
                trees_by_game=trees_by_game,
                active_indices=active_indices,
                chosen_actions=chosen_actions,
                active_done=active_done,
            )
            if bool(still_playing.any().item()):
                player_to_move[active_indices[still_playing]] *= OPPONENT_PLAYER
            self._emit(
                progress_callback,
                "ply_end",
                {
                    "ply": ply,
                    "active_games": active_count,
                    "wins": int(active_won.sum().detach().cpu().item()),
                    "draws": int(active_draw.sum().detach().cpu().item()),
                    "newly_done": int(active_done.sum().detach().cpu().item()),
                    "still_active": int((~game.done).sum().detach().cpu().item()),
                    "samples_so_far": sum(tensor.shape[0] for tensor in boards),
                },
            )

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

    def _supports_tree_reuse(self) -> bool:
        return hasattr(self.search, "search_batch_with_trees") and hasattr(self.search, "advance_tree")

    def _active_trees(
        self,
        trees_by_game: Optional[List[Any]],
        active_indices: torch.Tensor,
    ) -> Optional[List[Any]]:
        if trees_by_game is None:
            return None
        return [trees_by_game[int(index)] for index in active_indices.detach().cpu().tolist()]

    def _search_roots(
        self,
        roots: Connect4x4x4Batch,
        root_trees: Optional[List[Any]],
    ) -> BatchedSearchResult:
        search_with_trees = getattr(self.search, "search_batch_with_trees", None)
        if root_trees is not None and search_with_trees is not None:
            return search_with_trees(roots, root_trees)
        return self.search.search_batch(roots)

    def _advance_trees(
        self,
        trees_by_game: Optional[List[Any]],
        active_indices: torch.Tensor,
        chosen_actions: torch.Tensor,
        active_done: torch.Tensor,
    ) -> None:
        if trees_by_game is None:
            return

        advance_tree = getattr(self.search, "advance_tree", None)
        searched_trees = getattr(self.search, "last_trees", [])
        if advance_tree is None or len(searched_trees) == 0:
            return

        game_indices = active_indices.detach().cpu().tolist()
        actions = chosen_actions.detach().cpu().tolist()
        done_flags = active_done.detach().cpu().tolist()
        for local_index, game_index in enumerate(game_indices):
            if bool(done_flags[local_index]) or local_index >= len(searched_trees):
                trees_by_game[int(game_index)] = None
                continue
            trees_by_game[int(game_index)] = advance_tree(
                searched_trees[local_index],
                int(actions[local_index]),
            )

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

    def _emit(
        self,
        progress_callback: Optional[ProgressCallback],
        event: str,
        payload: Dict[str, float | int | str],
    ) -> None:
        if progress_callback is not None:
            progress_callback(event, payload)

    def _policy_entropy(self, policy: torch.Tensor) -> float:
        safe_policy = policy.clamp_min(1e-12)
        entropy = -(safe_policy * safe_policy.log()).sum(dim=1)
        return float(entropy.mean().detach().cpu().item())

    def _max_search_leaf_batch(self) -> int:
        batch_sizes = getattr(self.search, "last_leaf_batch_sizes", [])
        return int(max(batch_sizes)) if batch_sizes else 0

    def _max_search_expansion_batch(self) -> int:
        batch_sizes = getattr(self.search, "last_expansion_batch_sizes", [])
        return int(max(batch_sizes)) if batch_sizes else 0

    def _search_timing(self, key: str) -> float:
        timings = getattr(self.search, "last_timing_seconds", {})
        return float(timings.get(key, -1.0))
