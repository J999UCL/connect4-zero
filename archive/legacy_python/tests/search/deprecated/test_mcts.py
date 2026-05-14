import torch
import pytest

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE, BOARD_DTYPE, BOARD_SIZE, CURRENT_PLAYER
from connect4_zero.search.deprecated import MCTS, MCTSConfig


class ConstantEvaluator:
    def __init__(self, value: float) -> None:
        self.value = value

    def evaluate(self, state: Connect4x4x4Batch) -> float:
        return self.value


def draw_board() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[1, -1, -1, 1], [1, 1, -1, 1], [-1, 1, -1, -1], [-1, 1, 1, -1]],
                [[1, 1, 1, -1], [-1, 1, 1, 1], [1, -1, 1, 1], [1, -1, -1, -1]],
                [[-1, -1, -1, 1], [1, -1, 1, -1], [1, -1, -1, 1], [-1, 1, 1, 1]],
                [[-1, 1, -1, 1], [1, -1, 1, -1], [1, -1, 1, 1], [-1, 1, -1, -1]],
            ]
        ],
        dtype=BOARD_DTYPE,
    )


def one_legal_nonterminal_root() -> Connect4x4x4Batch:
    game = Connect4x4x4Batch(batch_size=1)
    game.board = draw_board()
    game.board[0, 0, 0, 2:4] = 0
    game.heights.fill_(BOARD_SIZE)
    game.heights[0, 0, 0] = 2
    return game


def immediate_win_root() -> Connect4x4x4Batch:
    game = Connect4x4x4Batch(batch_size=1)
    game.board[0, 0, 0, 0:3] = CURRENT_PLAYER
    game.heights[0, 0, 0] = 3
    return game


def test_search_returns_policy_statistics_for_single_root() -> None:
    game = Connect4x4x4Batch(batch_size=1)
    mcts = MCTS(
        config=MCTSConfig(num_simulations=8, exploration_constant=1.0),
        evaluator=ConstantEvaluator(0.0),
    )

    result = mcts.search(game)

    assert result.visit_counts.shape == (ACTION_SIZE,)
    assert result.policy.shape == (ACTION_SIZE,)
    assert result.q_values.shape == (ACTION_SIZE,)
    assert torch.isclose(result.policy.sum(), torch.tensor(1.0))
    assert result.visit_counts.sum().item() == 8


def test_search_does_not_mutate_input_root() -> None:
    game = Connect4x4x4Batch(batch_size=1)
    original_board = game.board.clone()
    original_heights = game.heights.clone()

    mcts = MCTS(config=MCTSConfig(num_simulations=4), evaluator=ConstantEvaluator(0.0))
    mcts.search(game)

    assert torch.equal(game.board, original_board)
    assert torch.equal(game.heights, original_heights)
    assert not game.done.any().item()


def test_full_columns_receive_zero_policy_probability() -> None:
    game = Connect4x4x4Batch(batch_size=1)
    game.heights[0, 0, 0] = BOARD_SIZE
    mcts = MCTS(
        config=MCTSConfig(num_simulations=16, exploration_constant=1.0),
        evaluator=ConstantEvaluator(0.0),
    )

    result = mcts.search(game)

    assert result.policy[0].item() == 0.0
    assert result.visit_counts[0].item() == 0.0
    assert torch.isclose(result.policy.sum(), torch.tensor(1.0))


def test_terminal_roots_are_rejected() -> None:
    game = Connect4x4x4Batch(batch_size=1)
    game.done[0] = True
    mcts = MCTS(config=MCTSConfig(num_simulations=1), evaluator=ConstantEvaluator(0.0))

    with pytest.raises(ValueError, match="non-terminal"):
        mcts.search(game)


def test_search_requires_single_root_state() -> None:
    game = Connect4x4x4Batch(batch_size=2)
    mcts = MCTS(config=MCTSConfig(num_simulations=1), evaluator=ConstantEvaluator(0.0))

    with pytest.raises(ValueError, match="batch_size=1"):
        mcts.search(game)


def test_backprop_flips_leaf_evaluator_value_to_root_perspective() -> None:
    game = one_legal_nonterminal_root()
    mcts = MCTS(config=MCTSConfig(num_simulations=1), evaluator=ConstantEvaluator(0.75))

    result = mcts.search(game)

    assert result.visit_counts[0].item() == 1.0
    assert result.q_values[0].item() == pytest.approx(-0.75)
    assert result.root_value == pytest.approx(-0.75)


def test_immediate_winning_move_is_preferred() -> None:
    game = immediate_win_root()
    mcts = MCTS(
        config=MCTSConfig(num_simulations=64, exploration_constant=1.0),
        evaluator=ConstantEvaluator(0.0),
    )

    result = mcts.search(game)

    assert result.policy.argmax().item() == 0
    assert result.q_values[0].item() == pytest.approx(1.0)
    assert result.visit_counts[0].item() > result.visit_counts[1:].max().item()


def test_mps_search_smoke_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    game = Connect4x4x4Batch(batch_size=1, device="mps")
    mcts = MCTS(
        config=MCTSConfig(
            num_simulations=8,
            rollout_batch_size=4,
            rollout_device="mps",
            seed=11,
        )
    )

    result = mcts.search(game)

    assert result.visit_counts.device.type == "mps"
    assert result.policy.device.type == "mps"
    assert result.visit_counts.shape == (ACTION_SIZE,)
    assert result.policy.shape == (ACTION_SIZE,)
    assert torch.isclose(result.policy.sum().cpu(), torch.tensor(1.0))
    assert result.visit_counts.sum().cpu().item() == 8
