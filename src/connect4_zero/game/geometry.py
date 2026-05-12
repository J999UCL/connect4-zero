"""Geometry helpers for the 4x4x4 board and 16-action gravity space."""

from itertools import product
from typing import List, Optional, Tuple, Union

import torch

from connect4_zero.game.constants import (
    ACTION_SIZE,
    BOARD_SIZE,
    INDEX_DTYPE,
    WIN_LENGTH,
)

DeviceLike = Optional[Union[str, torch.device]]
Coordinate = Tuple[int, int, int]
Line = Tuple[Coordinate, Coordinate, Coordinate, Coordinate]


def make_action_to_xy(device: DeviceLike = None) -> torch.Tensor:
    """Return a tensor mapping each action index to its ``(x, y)`` column."""
    coords = [(action // BOARD_SIZE, action % BOARD_SIZE) for action in range(ACTION_SIZE)]
    return torch.tensor(coords, dtype=INDEX_DTYPE, device=device)


def split_actions(actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert action indices to ``x`` and ``y`` tensors."""
    actions = actions.to(dtype=INDEX_DTYPE)
    return actions // BOARD_SIZE, actions % BOARD_SIZE


def make_win_lines(device: DeviceLike = None) -> torch.Tensor:
    """Return all length-4 winning lines as ``(76, 4, 3)`` coordinates."""
    lines = _generate_win_lines()
    return torch.tensor(lines, dtype=INDEX_DTYPE, device=device)


def make_win_line_flat_indices(device: DeviceLike = None) -> torch.Tensor:
    """Return flattened board indices for every winning line as ``(76, 4)``."""
    lines = make_win_lines(device=device)
    x = lines[..., 0]
    y = lines[..., 1]
    z = lines[..., 2]
    return x * BOARD_SIZE * BOARD_SIZE + y * BOARD_SIZE + z


def _generate_win_lines() -> List[Line]:
    directions = _canonical_directions()
    lines: List[Line] = []

    for dx, dy, dz in directions:
        for start in product(range(BOARD_SIZE), repeat=3):
            end = (
                start[0] + (WIN_LENGTH - 1) * dx,
                start[1] + (WIN_LENGTH - 1) * dy,
                start[2] + (WIN_LENGTH - 1) * dz,
            )
            if _in_bounds(end):
                line = tuple(
                    (
                        start[0] + step * dx,
                        start[1] + step * dy,
                        start[2] + step * dz,
                    )
                    for step in range(WIN_LENGTH)
                )
                lines.append(line)  # type: ignore[arg-type]

    return lines


def _canonical_directions() -> List[Coordinate]:
    directions: List[Coordinate] = []
    for direction in product((-1, 0, 1), repeat=3):
        if direction == (0, 0, 0):
            continue
        if _first_nonzero_is_positive(direction):
            directions.append(direction)
    return directions


def _first_nonzero_is_positive(direction: Coordinate) -> bool:
    for component in direction:
        if component != 0:
            return component > 0
    return False


def _in_bounds(coord: Coordinate) -> bool:
    return all(0 <= component < BOARD_SIZE for component in coord)
