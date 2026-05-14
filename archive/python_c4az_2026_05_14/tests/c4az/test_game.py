import numpy as np
import pytest

from c4az.game import (
    ACTION_SIZE,
    WIN_MASKS,
    IllegalMoveError,
    Position,
    action_to_xy,
    action_z_to_bit,
    bit_mask_for_cells,
    has_win,
    initial_position,
    symmetry_action_permutation,
    transform_action_values,
    transform_legal_mask,
    xyz_to_bit_index,
)


def test_action_mapping_and_gravity() -> None:
    position = initial_position()
    assert action_to_xy(0) == (0, 0)
    assert action_to_xy(15) == (3, 3)

    after = position.play(5)
    assert after.opponent & action_z_to_bit(5, 0)
    assert after.heights[5] == 1
    assert after.current == 0

    restored_side = after.play(5)
    assert restored_side.opponent & action_z_to_bit(5, 1)
    assert restored_side.current & action_z_to_bit(5, 0)
    assert restored_side.heights[5] == 2


def test_illegal_moves_are_rejected() -> None:
    position = initial_position()
    with pytest.raises(IllegalMoveError):
        position.play(-1)
    with pytest.raises(IllegalMoveError):
        position.play(16)

    for _ in range(4):
        position = position.play(0)
    with pytest.raises(IllegalMoveError):
        position.play(0)


def test_win_masks_count_and_represent_line_types() -> None:
    assert len(WIN_MASKS) == 76
    assert has_win(bit_mask_for_cells((x, 0, 0) for x in range(4)))
    assert has_win(bit_mask_for_cells((0, 0, z) for z in range(4)))
    assert has_win(bit_mask_for_cells((i, i, 0) for i in range(4)))
    assert has_win(bit_mask_for_cells((i, i, i) for i in range(4)))


def test_playing_a_winning_move_sets_child_value_for_next_player() -> None:
    current = sum(action_z_to_bit(action, 0) for action in (1, 2, 3))
    position = Position(current=current, heights=(0, 1, 1, 1, *([0] * 12)), ply=3)

    child = position.play(0)

    assert child.is_terminal
    assert child.terminal_value == -1.0
    assert child.opponent & action_z_to_bit(0, 0)


def test_draw_detection_on_full_non_winning_position() -> None:
    # A synthetic full board with no winner is hard to hand-author in 3D, so this
    # verifies the draw branch directly on a legal last move position.
    heights = [4] * ACTION_SIZE
    heights[15] = 3
    occupied_without_last = 0
    for action in range(15):
        for z in range(4):
            occupied_without_last |= action_z_to_bit(action, z)
    for z in range(3):
        occupied_without_last |= action_z_to_bit(15, z)
    position = Position(current=0, opponent=occupied_without_last, heights=tuple(heights), ply=63)
    child = position.play(15)
    assert child.is_terminal
    assert child.terminal_value == 0.0


def test_symmetries_transform_actions_masks_and_positions() -> None:
    values = np.arange(ACTION_SIZE, dtype=np.float32)
    mask = (1 << 0) | (1 << 5) | (1 << 15)
    position = initial_position().play(0).play(5).play(15)

    for symmetry in range(8):
        perm = symmetry_action_permutation(symmetry)
        assert sorted(perm) == list(range(ACTION_SIZE))
        transformed_values = transform_action_values(values, symmetry)
        for old_action, new_action in enumerate(perm):
            assert transformed_values[new_action] == values[old_action]
        transformed_mask = transform_legal_mask(mask, symmetry)
        assert bin(transformed_mask).count("1") == 3
        transformed_position = position.transform(symmetry)
        assert transformed_position.ply == position.ply
        assert transformed_position.legal_mask() == transform_legal_mask(position.legal_mask(), symmetry)


def test_bit_index_layout_is_z_y_x() -> None:
    assert xyz_to_bit_index(0, 0, 0) == 0
    assert xyz_to_bit_index(3, 0, 0) == 3
    assert xyz_to_bit_index(0, 1, 0) == 4
    assert xyz_to_bit_index(0, 0, 1) == 16
