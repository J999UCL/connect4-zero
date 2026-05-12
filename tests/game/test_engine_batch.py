import torch

from connect4_zero.game.constants import ACTION_SIZE, BOARD_SIZE, CURRENT_PLAYER
from connect4_zero.game.engine import Connect4x4x4Batch


def test_mixed_batch_legal_illegal_and_winning_games_are_independent() -> None:
    game = Connect4x4x4Batch(batch_size=3)
    game.heights[1, 0, 0] = BOARD_SIZE
    game.board[2, 0, 0, 0:3] = CURRENT_PLAYER
    game.heights[2, 0, 0] = 3

    result = game.step(torch.tensor([1, 0, 0]))

    assert result.legal.tolist() == [True, False, True]
    assert result.won.tolist() == [False, False, True]
    assert result.done.tolist() == [False, False, True]
    assert game.heights[:, 0, 0].tolist() == [0, BOARD_SIZE, BOARD_SIZE]
    assert game.heights[0, 0, 1].item() == 1


def test_legal_mask_shape_dtype_and_done_behavior() -> None:
    game = Connect4x4x4Batch(batch_size=2)
    game.heights[0, 0, 0] = BOARD_SIZE
    game.done[1] = True

    legal_mask = game.legal_mask()

    assert legal_mask.shape == (2, ACTION_SIZE)
    assert legal_mask.dtype == torch.bool
    assert not legal_mask[0, 0].item()
    assert legal_mask[0, 1:].all().item()
    assert not legal_mask[1].any().item()


def test_clone_does_not_share_mutable_state() -> None:
    game = Connect4x4x4Batch(batch_size=1)
    cloned = game.clone()

    cloned.step(torch.tensor([0]))

    assert game.heights[0, 0, 0].item() == 0
    assert cloned.heights[0, 0, 0].item() == 1


def test_to_returns_moved_copy_on_cpu() -> None:
    game = Connect4x4x4Batch(batch_size=1)
    moved = game.to("cpu")

    assert moved is not game
    assert moved.board.device.type == "cpu"
    assert moved.heights.device.type == "cpu"
