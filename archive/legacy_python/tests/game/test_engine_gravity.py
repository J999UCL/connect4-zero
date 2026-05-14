import pytest
import torch

from connect4_zero.game.constants import BOARD_SIZE, OPPONENT_PLAYER
from connect4_zero.game.engine import Connect4x4x4Batch


def test_repeated_moves_fill_column_bottom_to_top() -> None:
    game = Connect4x4x4Batch(batch_size=1)

    for expected_height in range(1, BOARD_SIZE + 1):
        result = game.step(torch.tensor([0]))
        assert result.legal.item()
        assert game.heights[0, 0, 0].item() == expected_height

    result = game.step(torch.tensor([0]))

    assert not result.legal.item()
    assert game.heights[0, 0, 0].item() == BOARD_SIZE


def test_non_terminal_move_is_canonicalized_for_next_player() -> None:
    game = Connect4x4x4Batch(batch_size=1)

    result = game.step(torch.tensor([0]))

    assert result.legal.item()
    assert not result.done.item()
    assert game.board[0, 0, 0, 0].item() == OPPONENT_PLAYER


def test_wrong_action_shape_raises_value_error() -> None:
    game = Connect4x4x4Batch(batch_size=2)

    with pytest.raises(ValueError):
        game.step(torch.tensor([0]))
