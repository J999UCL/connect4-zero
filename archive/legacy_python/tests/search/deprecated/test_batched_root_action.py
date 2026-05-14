import torch
import pytest

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE, BOARD_SIZE, CURRENT_PLAYER
from connect4_zero.search.deprecated import BatchedRootActionConfig, BatchedRootActionMCTS


def immediate_win_root(batch_size: int = 1) -> Connect4x4x4Batch:
    game = Connect4x4x4Batch(batch_size=batch_size)
    game.board[:, 0, 0, 0:3] = CURRENT_PLAYER
    game.heights[:, 0, 0] = 3
    return game


def test_search_batch_returns_policy_statistics_for_many_roots() -> None:
    roots = Connect4x4x4Batch(batch_size=4)
    search = BatchedRootActionMCTS(
        config=BatchedRootActionConfig(
            num_selection_waves=2,
            leaves_per_root=3,
            rollouts_per_leaf=2,
            seed=7,
        )
    )

    result = search.search_batch(roots)

    assert result.visit_counts.shape == (4, ACTION_SIZE)
    assert result.policy.shape == (4, ACTION_SIZE)
    assert result.q_values.shape == (4, ACTION_SIZE)
    assert result.root_values.shape == (4,)
    assert torch.allclose(result.policy.sum(dim=1), torch.ones(4))


def test_search_batch_does_not_mutate_roots() -> None:
    roots = Connect4x4x4Batch(batch_size=2)
    original_board = roots.board.clone()
    original_heights = roots.heights.clone()
    search = BatchedRootActionMCTS(
        config=BatchedRootActionConfig(num_selection_waves=1, leaves_per_root=2, rollouts_per_leaf=2)
    )

    search.search_batch(roots)

    assert torch.equal(roots.board, original_board)
    assert torch.equal(roots.heights, original_heights)
    assert not roots.done.any().item()


def test_full_columns_receive_zero_policy_probability() -> None:
    roots = Connect4x4x4Batch(batch_size=2)
    roots.heights[:, 0, 0] = BOARD_SIZE
    search = BatchedRootActionMCTS(
        config=BatchedRootActionConfig(num_selection_waves=1, leaves_per_root=2, rollouts_per_leaf=2)
    )

    result = search.search_batch(roots)

    assert torch.all(result.visit_counts[:, 0].eq(0))
    assert torch.all(result.policy[:, 0].eq(0))
    assert torch.allclose(result.policy.sum(dim=1), torch.ones(2))


def test_immediate_winning_move_is_revisited() -> None:
    roots = immediate_win_root()
    search = BatchedRootActionMCTS(
        config=BatchedRootActionConfig(
            num_selection_waves=8,
            leaves_per_root=1,
            rollouts_per_leaf=2,
            seed=11,
        )
    )

    result = search.search_batch(roots)

    assert result.q_values[0, 0].item() == pytest.approx(1.0)
    assert result.policy[0].argmax().item() == 0
    assert result.visit_counts[0, 0].item() > result.visit_counts[0, 1:].max().item()


def test_terminal_roots_return_zero_policy_and_terminal_value() -> None:
    roots = Connect4x4x4Batch(batch_size=2)
    roots.done[:] = True
    roots.outcome[0] = 1
    roots.outcome[1] = 0
    search = BatchedRootActionMCTS(
        config=BatchedRootActionConfig(num_selection_waves=1, leaves_per_root=2, rollouts_per_leaf=2)
    )

    result = search.search_batch(roots)

    assert torch.all(result.policy.eq(0))
    assert result.root_values.tolist() == [-1.0, 0.0]
