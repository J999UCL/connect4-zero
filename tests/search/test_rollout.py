import torch
import pytest

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import BOARD_DTYPE, BOARD_SIZE, CURRENT_PLAYER
from connect4_zero.search import RandomRolloutEvaluator


def test_rollout_returns_value_in_expected_range() -> None:
    game = Connect4x4x4Batch(batch_size=1)
    evaluator = RandomRolloutEvaluator(rollout_batch_size=16, seed=7)

    value = evaluator.evaluate(game)

    assert -1.0 <= value <= 1.0


def test_rollout_requires_single_state() -> None:
    game = Connect4x4x4Batch(batch_size=2)
    evaluator = RandomRolloutEvaluator(rollout_batch_size=4)

    with pytest.raises(ValueError, match="batch_size=1"):
        evaluator.evaluate(game)


def test_terminal_draw_evaluates_to_zero() -> None:
    game = Connect4x4x4Batch(batch_size=1)
    game.done[0] = True
    game.outcome[0] = 0
    evaluator = RandomRolloutEvaluator(rollout_batch_size=4)

    assert evaluator.evaluate(game) == 0.0


def test_terminal_previous_player_win_evaluates_as_loss_for_next_player() -> None:
    game = Connect4x4x4Batch(batch_size=1)
    game.done[0] = True
    game.outcome[0] = 1
    evaluator = RandomRolloutEvaluator(rollout_batch_size=4)

    assert evaluator.evaluate(game) == -1.0


def test_forced_immediate_win_rollout_returns_one() -> None:
    game = Connect4x4x4Batch(batch_size=1)
    game.heights.fill_(BOARD_SIZE)
    game.heights[0, 0, 0] = 3
    game.board[0, 0, 0, 0:3] = CURRENT_PLAYER
    evaluator = RandomRolloutEvaluator(rollout_batch_size=8, seed=3)

    assert evaluator.evaluate(game) == 1.0


def test_rollout_raises_if_max_steps_is_too_small() -> None:
    game = Connect4x4x4Batch(batch_size=1)
    evaluator = RandomRolloutEvaluator(rollout_batch_size=4, seed=3, max_steps=1)

    with pytest.raises(RuntimeError, match="max_steps"):
        evaluator.evaluate(game)


def test_cuda_rollout_smoke_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    game = Connect4x4x4Batch(batch_size=1)
    evaluator = RandomRolloutEvaluator(rollout_batch_size=4, device="cuda", seed=3)

    value = evaluator.evaluate(game)

    assert -1.0 <= value <= 1.0


def test_mps_rollout_smoke_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    game = Connect4x4x4Batch(batch_size=1)
    evaluator = RandomRolloutEvaluator(rollout_batch_size=4, device="mps", seed=3)

    value = evaluator.evaluate(game)

    assert -1.0 <= value <= 1.0
