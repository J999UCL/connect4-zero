import torch

from connect4_zero.game.constants import ACTION_SIZE, BOARD_SIZE
from connect4_zero.game.geometry import make_action_to_xy, make_win_line_flat_indices, make_win_lines


def test_action_mapping_covers_all_columns() -> None:
    action_to_xy = make_action_to_xy()

    assert action_to_xy.shape == (ACTION_SIZE, 2)
    assert action_to_xy[0].tolist() == [0, 0]
    assert action_to_xy[15].tolist() == [3, 3]
    assert len({tuple(coord) for coord in action_to_xy.tolist()}) == ACTION_SIZE


def test_win_line_generation_has_expected_shape_and_count() -> None:
    win_lines = make_win_lines()

    assert win_lines.shape == (76, BOARD_SIZE, 3)
    for line in win_lines.tolist():
        assert len({tuple(coord) for coord in line}) == BOARD_SIZE
        for coord in line:
            assert all(0 <= value < BOARD_SIZE for value in coord)


def test_flat_win_line_indices_are_unique_within_each_line() -> None:
    flat_indices = make_win_line_flat_indices()

    assert flat_indices.shape == (76, BOARD_SIZE)
    assert flat_indices.dtype == torch.long
    for line in flat_indices.tolist():
        assert len(set(line)) == BOARD_SIZE
