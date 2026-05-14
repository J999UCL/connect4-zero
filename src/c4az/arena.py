from __future__ import annotations

from dataclasses import dataclass

from c4az.game import initial_position
from c4az.mcts import Evaluator, MCTSConfig, PUCTMCTS


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    games: int = 2
    simulations_per_move: int = 64
    c_puct: float = 1.5
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class ArenaResult:
    wins: int
    losses: int
    draws: int
    games: int

    @property
    def score_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(1, self.games)


def evaluate_arena(candidate: Evaluator, baseline: Evaluator, config: ArenaConfig) -> ArenaResult:
    wins = losses = draws = 0
    for game_index in range(config.games):
        candidate_starts = game_index % 2 == 0
        result = _play_game(candidate, baseline, config, candidate_starts=candidate_starts, seed=None if config.seed is None else config.seed + game_index)
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1
    return ArenaResult(wins=wins, losses=losses, draws=draws, games=config.games)


def _play_game(
    candidate: Evaluator,
    baseline: Evaluator,
    config: ArenaConfig,
    *,
    candidate_starts: bool,
    seed: int | None,
) -> int:
    position = initial_position()
    trees = [None, None]
    mcts = [
        PUCTMCTS(candidate, MCTSConfig(simulations_per_move=config.simulations_per_move, c_puct=config.c_puct, seed=seed)),
        PUCTMCTS(baseline, MCTSConfig(simulations_per_move=config.simulations_per_move, c_puct=config.c_puct, seed=seed)),
    ]
    while not position.is_terminal:
        side_index = position.ply % 2
        candidate_to_move = candidate_starts if side_index == 0 else not candidate_starts
        player_index = 0 if candidate_to_move else 1
        result = mcts[player_index].search(position, root=trees[player_index], add_root_noise=False, temperature=0.0)
        action = int(result.policy.argmax())
        trees[player_index] = mcts[player_index].advance_tree(result.root, action)
        other = 1 - player_index
        trees[other] = mcts[other].advance_tree(trees[other], action) if trees[other] is not None else None
        position = position.play(action)

    assert position.terminal_value is not None
    if position.terminal_value == 0:
        return 0
    candidate_to_move_at_terminal = candidate_starts if position.ply % 2 == 0 else not candidate_starts
    value_for_candidate = position.terminal_value if candidate_to_move_at_terminal else -position.terminal_value
    return 1 if value_for_candidate > 0 else -1
