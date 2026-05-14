import torch
import pytest

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import BOARD_SIZE, CURRENT_PLAYER
from connect4_zero.search import BatchedRandomRolloutEvaluator


def test_batched_rollout_returns_one_value_per_state() -> None:
    states = Connect4x4x4Batch(batch_size=3)
    evaluator = BatchedRandomRolloutEvaluator(rollouts_per_state=4, seed=5)

    values = evaluator.evaluate_batch(states)

    assert values.shape == (3,)
    assert values.dtype == torch.float32
    assert torch.all(values.ge(-1.0))
    assert torch.all(values.le(1.0))


def test_batched_rollout_handles_terminal_values() -> None:
    states = Connect4x4x4Batch(batch_size=2)
    states.done[:] = True
    states.outcome[0] = 1
    states.outcome[1] = 0
    evaluator = BatchedRandomRolloutEvaluator(rollouts_per_state=3, seed=5)

    values = evaluator.evaluate_batch(states)

    assert values.tolist() == [-1.0, 0.0]


def test_batched_rollout_forced_immediate_win_returns_one() -> None:
    states = Connect4x4x4Batch(batch_size=2)
    states.heights.fill_(BOARD_SIZE)
    states.heights[:, 0, 0] = 3
    states.board[:, 0, 0, 0:3] = CURRENT_PLAYER
    evaluator = BatchedRandomRolloutEvaluator(rollouts_per_state=4, seed=3)

    values = evaluator.evaluate_batch(states)

    assert values.tolist() == [1.0, 1.0]


def test_batched_rollout_raises_if_max_steps_is_too_small() -> None:
    states = Connect4x4x4Batch(batch_size=2)
    evaluator = BatchedRandomRolloutEvaluator(rollouts_per_state=2, seed=3, max_steps=1)

    with pytest.raises(RuntimeError, match="max_steps"):
        evaluator.evaluate_batch(states)
