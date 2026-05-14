import torch

from connect4_zero.game.constants import BOARD_DTYPE, BOARD_SIZE, CURRENT_PLAYER, OPPONENT_PLAYER
from connect4_zero.game.engine import Connect4x4x4Batch
from connect4_zero.game.geometry import make_win_lines


def _draw_board() -> torch.Tensor:
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


def test_every_generated_line_is_detected_as_current_player_win() -> None:
    win_lines = make_win_lines()
    game = Connect4x4x4Batch(batch_size=win_lines.shape[0])

    for batch_idx, line in enumerate(win_lines):
        game.board[batch_idx, line[:, 0], line[:, 1], line[:, 2]] = CURRENT_PLAYER

    assert game.check_wins(player=CURRENT_PLAYER).all().item()


def test_partial_line_is_not_a_win() -> None:
    line = make_win_lines()[0]
    game = Connect4x4x4Batch(batch_size=1)
    game.board[0, line[:3, 0], line[:3, 1], line[:3, 2]] = CURRENT_PLAYER

    assert not game.check_wins(player=CURRENT_PLAYER).item()


def test_opponent_line_is_not_current_player_win() -> None:
    line = make_win_lines()[0]
    game = Connect4x4x4Batch(batch_size=1)
    game.board[0, line[:, 0], line[:, 1], line[:, 2]] = OPPONENT_PLAYER

    assert not game.check_wins(player=CURRENT_PLAYER).item()
    assert game.check_wins(player=OPPONENT_PLAYER).item()


def test_winning_step_is_checked_before_canonicalization() -> None:
    game = Connect4x4x4Batch(batch_size=1)
    game.board[0, 0, 0, 0:3] = CURRENT_PLAYER
    game.heights[0, 0, 0] = 3

    result = game.step(torch.tensor([0]))

    assert result.legal.item()
    assert result.won.item()
    assert result.done.item()
    assert game.board[0, 0, 0].tolist() == [1, 1, 1, 1]


def test_full_board_without_any_winning_line_is_draw() -> None:
    game = Connect4x4x4Batch(batch_size=1)
    game.board = _draw_board()
    game.heights.fill_(BOARD_SIZE)

    assert not game.check_wins(player=CURRENT_PLAYER).item()
    assert not game.check_wins(player=OPPONENT_PLAYER).item()
    assert game.is_draw().item()


def test_final_legal_move_can_mark_draw() -> None:
    game = Connect4x4x4Batch(batch_size=1)
    game.board = _draw_board()
    game.board[0, 0, 0, 3] = 0
    game.heights.fill_(BOARD_SIZE)
    game.heights[0, 0, 0] = 3

    result = game.step(torch.tensor([0]))

    assert result.legal.item()
    assert result.draw.item()
    assert result.done.item()
    assert result.outcome.item() == 0
