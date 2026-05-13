"""Arena evaluation between two PUCT agents."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE
from connect4_zero.model import load_checkpoint
from connect4_zero.search import BatchedPUCTMCTS, NeuralPolicyValueEvaluator, PUCTMCTSConfig, PUCTSearchTree
from connect4_zero.scripts._common import RateTracker

_CANDIDATE = 1
_BASELINE = -1


@dataclass(frozen=True)
class ArenaConfig:
    """Configuration for checkpoint-vs-checkpoint evaluation."""

    candidate_checkpoint: Path
    baseline_checkpoint: Path
    games: int = 128
    batch_size: int = 32
    device: str | torch.device = "cpu"
    simulations_per_root: int = 64
    max_leaf_batch_size: int = 128
    c_puct: float = 1.5
    policy_temperature: float = 1.0
    inference_batch_size: int = 4096
    max_plies: int = 64
    seed: Optional[int] = None
    alternate_starts: bool = True
    opening_plies: int = 0
    paired_openings: bool = False
    add_root_noise: bool = False
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    action_temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.games <= 0:
            raise ValueError("games must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.simulations_per_root <= 0:
            raise ValueError("simulations_per_root must be positive")
        if self.max_leaf_batch_size <= 0:
            raise ValueError("max_leaf_batch_size must be positive")
        if self.c_puct < 0:
            raise ValueError("c_puct must be non-negative")
        if self.policy_temperature <= 0:
            raise ValueError("policy_temperature must be positive")
        if self.inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        if self.max_plies <= 0:
            raise ValueError("max_plies must be positive")
        if self.opening_plies < 0:
            raise ValueError("opening_plies must be non-negative")
        if self.opening_plies > 6:
            raise ValueError("opening_plies must be <= 6 to avoid terminal random openings")
        if self.paired_openings and self.games % 2 != 0:
            raise ValueError("paired_openings requires an even number of games")
        if self.paired_openings and self.batch_size % 2 != 0:
            raise ValueError("paired_openings requires an even batch_size")
        if self.root_dirichlet_alpha <= 0:
            raise ValueError("root_dirichlet_alpha must be positive")
        if not 0 <= self.root_exploration_fraction <= 1:
            raise ValueError("root_exploration_fraction must be in [0, 1]")
        if self.action_temperature < 0:
            raise ValueError("action_temperature must be non-negative")


@dataclass(frozen=True)
class ArenaSummary:
    """Aggregate arena result, from candidate checkpoint perspective."""

    candidate_checkpoint: str
    baseline_checkpoint: str
    games: int
    candidate_wins: int
    baseline_wins: int
    draws: int
    candidate_score_rate: float
    baseline_score_rate: float
    avg_plies: float
    elapsed_seconds: float
    games_per_second: float
    opening_plies: int = 0
    paired_openings: bool = False
    unique_openings: int = 1
    add_root_noise: bool = False
    action_temperature: float = 0.0
    candidate_first: dict[str, int] = field(default_factory=dict)
    baseline_first: dict[str, int] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_arena(config: ArenaConfig, logger=None) -> ArenaSummary:
    """Play an arena and return aggregate results."""
    if config.seed is not None:
        torch.manual_seed(config.seed)
    rng = torch.Generator(device="cpu")
    if config.seed is not None:
        rng.manual_seed(config.seed)

    device = torch.device(config.device)
    candidate_search = _build_search(config.candidate_checkpoint, config, device, seed_offset=17)
    baseline_search = _build_search(config.baseline_checkpoint, config, device, seed_offset=29)

    candidate_wins = 0
    baseline_wins = 0
    draws = 0
    total_plies = 0
    candidate_first = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    baseline_first = {"games": 0, "wins": 0, "losses": 0, "draws": 0}

    started_at = time.perf_counter()
    tracker = RateTracker()
    completed = 0
    next_game_index = 0

    while next_game_index < config.games:
        batch_games = min(config.batch_size, config.games - next_game_index)
        batch_result = _play_batch(
            start_index=next_game_index,
            batch_games=batch_games,
            config=config,
            candidate_search=candidate_search,
            baseline_search=baseline_search,
            rng=rng,
        )
        completed += batch_games
        next_game_index += batch_games
        candidate_wins += batch_result["candidate_wins"]
        baseline_wins += batch_result["baseline_wins"]
        draws += batch_result["draws"]
        total_plies += batch_result["total_plies"]
        _add_breakdown(candidate_first, batch_result["candidate_first"])
        _add_breakdown(baseline_first, batch_result["baseline_first"])

        if logger is not None:
            elapsed, _ = tracker.mark()
            logger.info(
                "arena_progress completed=%s/%s candidate_wins=%s baseline_wins=%s draws=%s games_per_sec=%.3f",
                completed,
                config.games,
                candidate_wins,
                baseline_wins,
                draws,
                completed / elapsed if elapsed > 0 else 0.0,
            )

    elapsed_seconds = time.perf_counter() - started_at
    candidate_score = (candidate_wins + 0.5 * draws) / config.games
    baseline_score = (baseline_wins + 0.5 * draws) / config.games
    return ArenaSummary(
        candidate_checkpoint=str(config.candidate_checkpoint),
        baseline_checkpoint=str(config.baseline_checkpoint),
        games=config.games,
        candidate_wins=candidate_wins,
        baseline_wins=baseline_wins,
        draws=draws,
        candidate_score_rate=candidate_score,
        baseline_score_rate=baseline_score,
        avg_plies=total_plies / config.games,
        elapsed_seconds=elapsed_seconds,
        games_per_second=config.games / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        opening_plies=config.opening_plies,
        paired_openings=config.paired_openings,
        unique_openings=_unique_opening_count(config),
        add_root_noise=config.add_root_noise,
        action_temperature=config.action_temperature,
        candidate_first=candidate_first,
        baseline_first=baseline_first,
    )


def _build_search(
    checkpoint: Path,
    config: ArenaConfig,
    device: torch.device,
    *,
    seed_offset: int,
) -> BatchedPUCTMCTS:
    loaded = load_checkpoint(checkpoint, map_location=device)
    evaluator = NeuralPolicyValueEvaluator(
        model=loaded.model,
        device=device,
        inference_batch_size=config.inference_batch_size,
    )
    search_config = PUCTMCTSConfig(
        simulations_per_root=config.simulations_per_root,
        max_leaf_batch_size=config.max_leaf_batch_size,
        c_puct=config.c_puct,
        policy_temperature=config.policy_temperature,
        root_dirichlet_alpha=config.root_dirichlet_alpha,
        root_exploration_fraction=config.root_exploration_fraction,
        add_root_noise=config.add_root_noise,
        max_selection_depth=config.max_plies,
        seed=None if config.seed is None else config.seed + seed_offset,
    )
    return BatchedPUCTMCTS(evaluator=evaluator, config=search_config)


def _play_batch(
    *,
    start_index: int,
    batch_games: int,
    config: ArenaConfig,
    candidate_search: BatchedPUCTMCTS,
    baseline_search: BatchedPUCTMCTS,
    rng: torch.Generator,
) -> dict[str, object]:
    games = Connect4x4x4Batch(batch_games, device="cpu")
    _apply_openings(games, start_index=start_index, config=config, rng=rng)
    active = torch.ones(batch_games, dtype=torch.bool)
    owners = _initial_owners(start_index, batch_games, alternate=config.alternate_starts)
    starters = owners.clone()
    candidate_trees: list[Optional[PUCTSearchTree]] = [None for _ in range(batch_games)]
    baseline_trees: list[Optional[PUCTSearchTree]] = [None for _ in range(batch_games)]
    plies = torch.full((batch_games,), config.opening_plies, dtype=torch.long)

    candidate_wins = 0
    baseline_wins = 0
    draws = 0
    candidate_first = {"games": int((starters == _CANDIDATE).sum().item()), "wins": 0, "losses": 0, "draws": 0}
    baseline_first = {"games": int((starters == _BASELINE).sum().item()), "wins": 0, "losses": 0, "draws": 0}

    for _ply in range(config.max_plies):
        if not bool(active.any().item()):
            break

        active_indices = active.nonzero(as_tuple=False).flatten()
        actions = torch.full((batch_games,), -1, dtype=torch.long)
        next_candidate_trees: dict[int, Optional[PUCTSearchTree]] = {}
        next_baseline_trees: dict[int, Optional[PUCTSearchTree]] = {}

        for side, search, trees, next_trees in (
            (_CANDIDATE, candidate_search, candidate_trees, next_candidate_trees),
            (_BASELINE, baseline_search, baseline_trees, next_baseline_trees),
        ):
            side_indices = active_indices[owners[active_indices] == side]
            if side_indices.numel() == 0:
                continue
            roots = _slice_batch(games, side_indices.tolist())
            root_trees = [trees[int(index)] for index in side_indices.tolist()]
            result = search.search_batch_with_trees(roots, root_trees)
            chosen = _choose_actions(
                result.visit_counts,
                roots.legal_mask(),
                temperature=config.action_temperature,
                rng=rng,
            )
            for local_index, game_index in enumerate(side_indices.tolist()):
                action = int(chosen[local_index].item())
                actions[game_index] = action
                next_trees[game_index] = search.advance_tree(search.last_trees[local_index], action)

        owner_before = owners.clone()
        for game_index in active_indices.tolist():
            action = int(actions[game_index].item())
            if action < 0 or action >= ACTION_SIZE:
                raise RuntimeError(f"arena failed to select an action for game {game_index}")

            if game_index in next_candidate_trees:
                candidate_trees[game_index] = next_candidate_trees[game_index]
            else:
                candidate_trees[game_index] = (
                    candidate_search.advance_tree(candidate_trees[game_index], action)
                    if candidate_trees[game_index] is not None
                    else None
                )

            if game_index in next_baseline_trees:
                baseline_trees[game_index] = next_baseline_trees[game_index]
            else:
                baseline_trees[game_index] = (
                    baseline_search.advance_tree(baseline_trees[game_index], action)
                    if baseline_trees[game_index] is not None
                    else None
                )

        step_result = games.step(actions)
        if not bool(step_result.legal[active_indices].all().item()):
            raise RuntimeError("arena selected an illegal move")

        plies[active_indices] += 1
        newly_done = active & step_result.done
        for game_index in newly_done.nonzero(as_tuple=False).flatten().tolist():
            starter = int(starters[game_index].item())
            mover = int(owner_before[game_index].item())
            if bool(step_result.won[game_index].item()):
                if mover == _CANDIDATE:
                    candidate_wins += 1
                    _record_breakdown(candidate_first, baseline_first, starter, candidate_won=True, draw=False)
                else:
                    baseline_wins += 1
                    _record_breakdown(candidate_first, baseline_first, starter, candidate_won=False, draw=False)
            else:
                draws += 1
                _record_breakdown(candidate_first, baseline_first, starter, candidate_won=False, draw=True)

        still_active = active & ~step_result.done
        owners[still_active] *= -1
        active = still_active

    if bool(active.any().item()):
        for game_index in active.nonzero(as_tuple=False).flatten().tolist():
            draws += 1
            starter = int(starters[game_index].item())
            _record_breakdown(candidate_first, baseline_first, starter, candidate_won=False, draw=True)

    return {
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "draws": draws,
        "total_plies": int(plies.sum().item()),
        "candidate_first": candidate_first,
        "baseline_first": baseline_first,
    }


def _apply_openings(
    games: Connect4x4x4Batch,
    *,
    start_index: int,
    config: ArenaConfig,
    rng: torch.Generator,
) -> None:
    if config.opening_plies == 0:
        return

    if config.paired_openings:
        for offset in range(0, games.batch_size, 2):
            state = _random_opening(config.opening_plies, rng)
            _copy_single_state(state, games, offset)
            _copy_single_state(state, games, offset + 1)
        return

    for offset in range(games.batch_size):
        _copy_single_state(_random_opening(config.opening_plies, rng), games, offset)


def _random_opening(opening_plies: int, rng: torch.Generator) -> Connect4x4x4Batch:
    state = Connect4x4x4Batch(1, device="cpu")
    for _ in range(opening_plies):
        legal_actions = state.legal_mask()[0].nonzero(as_tuple=False).flatten()
        choice_index = int(torch.randint(len(legal_actions), (1,), generator=rng).item())
        action = legal_actions[choice_index].to(dtype=torch.long).view(1)
        result = state.step(action)
        if bool(result.done[0].item()):
            raise RuntimeError("random opening became terminal; lower opening_plies")
    return state


def _copy_single_state(source: Connect4x4x4Batch, target: Connect4x4x4Batch, index: int) -> None:
    target.board[index : index + 1] = source.board
    target.heights[index : index + 1] = source.heights
    target.done[index : index + 1] = source.done
    target.outcome[index : index + 1] = source.outcome


def _initial_owners(start_index: int, batch_games: int, alternate: bool) -> torch.Tensor:
    owners = torch.full((batch_games,), _CANDIDATE, dtype=torch.int8)
    if alternate:
        for offset in range(batch_games):
            if (start_index + offset) % 2 == 1:
                owners[offset] = _BASELINE
    return owners


def _slice_batch(games: Connect4x4x4Batch, indices: list[int]) -> Connect4x4x4Batch:
    index_tensor = torch.tensor(indices, dtype=torch.long, device=games.device)
    roots = Connect4x4x4Batch(len(indices), device=games.device)
    roots.board = games.board[index_tensor].clone()
    roots.heights = games.heights[index_tensor].clone()
    roots.done = games.done[index_tensor].clone()
    roots.outcome = games.outcome[index_tensor].clone()
    return roots


def _choose_actions(
    visit_counts: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    temperature: float,
    rng: torch.Generator,
) -> torch.Tensor:
    if temperature > 0:
        weights = visit_counts.to(dtype=torch.float32).pow(1.0 / temperature)
        weights = weights.masked_fill(~legal_mask, 0.0)
        fallback = legal_mask.to(dtype=torch.float32)
        fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
        totals = weights.sum(dim=1, keepdim=True)
        weights = torch.where(totals > 0, weights / totals.clamp_min(1e-12), fallback)
        return torch.multinomial(weights.cpu(), num_samples=1, generator=rng).flatten().to(dtype=torch.long)

    scores = visit_counts.to(dtype=torch.float32).clone()
    scores = scores.masked_fill(~legal_mask, -1.0)
    chosen = scores.argmax(dim=1)
    no_visits = visit_counts.sum(dim=1) <= 0
    if bool(no_visits.any().item()):
        fallback = legal_mask.to(dtype=torch.float32).argmax(dim=1)
        chosen = torch.where(no_visits, fallback, chosen)
    return chosen.to(dtype=torch.long)


def _unique_opening_count(config: ArenaConfig) -> int:
    if config.opening_plies == 0:
        return 1
    if config.paired_openings:
        return config.games // 2
    return config.games


def _record_breakdown(
    candidate_first: dict[str, int],
    baseline_first: dict[str, int],
    starter: int,
    *,
    candidate_won: bool,
    draw: bool,
) -> None:
    bucket = candidate_first if starter == _CANDIDATE else baseline_first
    if draw:
        bucket["draws"] += 1
    elif candidate_won:
        if starter == _CANDIDATE:
            bucket["wins"] += 1
        else:
            bucket["losses"] += 1
    else:
        if starter == _CANDIDATE:
            bucket["losses"] += 1
        else:
            bucket["wins"] += 1


def _add_breakdown(target: dict[str, int], source: object) -> None:
    source_dict = source if isinstance(source, dict) else {}
    for key in ("games", "wins", "losses", "draws"):
        target[key] += int(source_dict.get(key, 0))
