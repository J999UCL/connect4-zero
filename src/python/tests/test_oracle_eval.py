from __future__ import annotations

import numpy as np
import torch

from c4zero_oracles.gerard import (
    INVALID_MOVE_VALUE,
    Score4PolicyAdapter,
    evaluate_against_portable_oracle,
    evaluate_model_on_records,
)
from c4zero_oracles.portable import Position, solve


class ConstantPolicy(torch.nn.Module):
    def __init__(self, action: int) -> None:
        super().__init__()
        self.action = action

    def forward(self, x):
        logits = torch.zeros((x.shape[0], 16), dtype=torch.float32, device=x.device)
        logits[:, self.action] = 10.0
        return logits, torch.zeros((x.shape[0],), dtype=torch.float32, device=x.device)


class PlaneEchoPolicy(torch.nn.Module):
    def forward(self, x):
        logits = torch.zeros((x.shape[0], 16), dtype=torch.float32, device=x.device)
        logits[:, 0] = x[:, 0].sum(dim=(1, 2, 3))
        logits[:, 1] = x[:, 1].sum(dim=(1, 2, 3))
        return logits, torch.zeros((x.shape[0],), dtype=torch.float32, device=x.device)


def record(move_values, heights=None):
    return {
        "bb0": (1 << 0) | (1 << 17),
        "bb1": 1 << 5,
        "turn": 0,
        "heights": heights or [0] * 16,
        "oracle_move_values": np.asarray(move_values, dtype=np.int32),
    }


def test_eval_set_metrics_drop_unprobed_rows_and_score_model_action():
    values = [-5] * 16
    values[0] = 100_010
    values[1] = -100_010
    values[2] = 0
    unprobed = [INVALID_MOVE_VALUE] * 16

    metrics = evaluate_model_on_records(
        ConstantPolicy(action=0),
        [record(values), record(unprobed)],
        batch_size=2,
    )

    assert metrics.n_input == 2
    assert metrics.n_evaluated == 1
    assert metrics.n_dropped_unprobed == 1
    assert metrics.optimality_rate == 1.0
    assert metrics.mate_find_rate == 1.0
    assert metrics.blunder_rate == 0.0
    assert metrics.action_counts[0] == 1


def test_eval_masks_illegal_policy_actions():
    values = [0] * 16
    values[0] = 1
    values[1] = 7
    heights = [4] + [0] * 15

    metrics = evaluate_model_on_records(
        ConstantPolicy(action=0),
        [record(values, heights=heights)],
        batch_size=1,
    )

    assert metrics.n_evaluated == 1
    assert metrics.action_counts[1] == 1
    assert metrics.optimality_rate == 1.0


def test_score4_signed_obs_adapter_maps_to_current_and_opponent_planes():
    adapter = Score4PolicyAdapter(PlaneEchoPolicy())
    obs = torch.zeros((1, 64), dtype=torch.float32)
    obs[0, 0] = 1.0
    obs[0, 1] = 1.0
    obs[0, 5] = -1.0

    logits, values = adapter(obs)

    assert logits.shape == (1, 16)
    assert values.shape == (1,)
    assert logits[0, 0].item() == 2.0
    assert logits[0, 1].item() == 1.0


def test_portable_oracle_takes_immediate_win():
    position = Position().play(0).play(4).play(1).play(5).play(2).play(6)

    value, action, move_values = solve(position, max_depth=2)

    assert action == 3
    assert value > 90_000
    assert move_values[3] > 90_000


def test_portable_oracle_head_to_head_smoke():
    result = evaluate_against_portable_oracle(ConstantPolicy(action=5), depth=1, games=2, seed=1)

    assert result.games == 2
    assert result.wins + result.losses + result.draws == 2
    assert result.avg_plies > 0.0
