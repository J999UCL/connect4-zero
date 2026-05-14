from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from c4az.data import SelfPlaySample
from c4az.game import Position, initial_position
from c4az.mcts import Evaluator, MCTSConfig, PUCTMCTS, choose_action, visit_counts_to_policy


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    games: int = 1
    simulations_per_move: int = 128
    c_puct: float = 1.5
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    temperature_cutoff_ply: int = 12
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class SelfPlayGame:
    samples: list[SelfPlaySample]
    terminal_value: float
    plies: int


def generate_self_play_games(evaluator: Evaluator, config: SelfPlayConfig) -> list[SelfPlayGame]:
    rng = random.Random(config.seed)
    games = []
    for game_id in range(config.games):
        game_seed = None if config.seed is None else config.seed + game_id
        games.append(_play_one_game(evaluator, config, game_id=game_id, seed=game_seed, rng=rng))
    return games


def _play_one_game(
    evaluator: Evaluator,
    config: SelfPlayConfig,
    *,
    game_id: int,
    seed: int | None,
    rng: random.Random,
) -> SelfPlayGame:
    position = initial_position()
    tree = None
    pending: list[tuple[Position, np.ndarray, np.ndarray, int]] = []
    mcts = PUCTMCTS(
        evaluator,
        MCTSConfig(
            simulations_per_move=config.simulations_per_move,
            c_puct=config.c_puct,
            root_dirichlet_alpha=config.root_dirichlet_alpha,
            root_exploration_fraction=config.root_exploration_fraction,
            seed=seed,
        ),
    )
    while not position.is_terminal:
        result = mcts.search(position, root=tree, add_root_noise=True, temperature=1.0)
        temperature = 1.0 if position.ply < config.temperature_cutoff_ply else 0.0
        policy = visit_counts_to_policy(result.visit_counts, temperature, position.legal_mask())
        action = choose_action(policy, rng)
        pending.append((position, policy, result.visit_counts, action))
        tree = mcts.advance_tree(result.root, action)
        position = position.play(action)

    assert position.terminal_value is not None
    samples = []
    final_ply = position.ply
    for sample_position, policy, visits, action in pending:
        flips = final_ply - sample_position.ply
        value = float(position.terminal_value) * ((-1.0) ** flips)
        samples.append(
            SelfPlaySample.from_position(
                sample_position,
                policy=policy,
                value=value,
                visit_counts=visits,
                action=action,
                game_id=game_id,
            )
        )
    return SelfPlayGame(samples=samples, terminal_value=float(position.terminal_value), plies=position.ply)
