from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Iterable

import numpy as np
import torch

BOARD_SIZE = 4
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
CELL_COUNT = BOARD_SIZE * BOARD_SIZE * BOARD_SIZE
FULL_MASK = (1 << ACTION_SIZE) - 1


class IllegalMoveError(ValueError):
    """Raised when an action cannot be played from a position."""


def action_to_xy(action: int) -> tuple[int, int]:
    if action < 0 or action >= ACTION_SIZE:
        raise IllegalMoveError(f"action out of range: {action}")
    return action % BOARD_SIZE, action // BOARD_SIZE


def xyz_to_bit_index(x: int, y: int, z: int) -> int:
    return z * ACTION_SIZE + y * BOARD_SIZE + x


def action_z_to_bit(action: int, z: int) -> int:
    if z < 0 or z >= BOARD_SIZE:
        raise IllegalMoveError(f"z out of range: {z}")
    return 1 << (z * ACTION_SIZE + action)


def bit_mask_for_cells(cells: Iterable[tuple[int, int, int]]) -> int:
    mask = 0
    for x, y, z in cells:
        mask |= 1 << xyz_to_bit_index(x, y, z)
    return mask


def _directions() -> list[tuple[int, int, int]]:
    dirs: list[tuple[int, int, int]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                first = next(v for v in (dx, dy, dz) if v != 0)
                if first > 0:
                    dirs.append((dx, dy, dz))
    return dirs


def _compute_win_masks() -> tuple[int, ...]:
    masks: list[int] = []
    for dx, dy, dz in _directions():
        for z in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                for x in range(BOARD_SIZE):
                    end_x = x + dx * (BOARD_SIZE - 1)
                    end_y = y + dy * (BOARD_SIZE - 1)
                    end_z = z + dz * (BOARD_SIZE - 1)
                    if not (
                        0 <= end_x < BOARD_SIZE
                        and 0 <= end_y < BOARD_SIZE
                        and 0 <= end_z < BOARD_SIZE
                    ):
                        continue
                    cells = [(x + i * dx, y + i * dy, z + i * dz) for i in range(BOARD_SIZE)]
                    masks.append(bit_mask_for_cells(cells))
    unique = tuple(dict.fromkeys(masks))
    if len(unique) != 76:
        raise RuntimeError(f"expected 76 winning masks, got {len(unique)}")
    return unique


WIN_MASKS = _compute_win_masks()


def has_win(bits: int) -> bool:
    return any((bits & mask) == mask for mask in WIN_MASKS)


def _initial_heights() -> tuple[int, ...]:
    return (0,) * ACTION_SIZE


@dataclass(frozen=True, slots=True)
class Position:
    """Canonical bitboard position.

    ``current`` always stores the stones of the side to move, and ``opponent``
    stores the stones of the player who just moved.
    """

    current: int = 0
    opponent: int = 0
    heights: tuple[int, ...] = _initial_heights()
    ply: int = 0
    terminal_value: float | None = None

    @property
    def is_terminal(self) -> bool:
        return self.terminal_value is not None

    def legal_mask(self) -> int:
        if self.is_terminal:
            return 0
        mask = 0
        for action, height in enumerate(self.heights):
            if height < BOARD_SIZE:
                mask |= 1 << action
        return mask

    def legal_actions(self) -> list[int]:
        mask = self.legal_mask()
        return [action for action in range(ACTION_SIZE) if mask & (1 << action)]

    def play(self, action: int) -> Position:
        if self.is_terminal:
            raise IllegalMoveError("cannot play from a terminal position")
        if action < 0 or action >= ACTION_SIZE:
            raise IllegalMoveError(f"action out of range: {action}")
        height = self.heights[action]
        if height >= BOARD_SIZE:
            raise IllegalMoveError(f"column is full: {action}")

        bit = action_z_to_bit(action, height)
        if (self.current | self.opponent) & bit:
            raise IllegalMoveError(f"target cell is already occupied: action={action} z={height}")

        mover_bits = self.current | bit
        heights = list(self.heights)
        heights[action] += 1
        next_ply = self.ply + 1

        if has_win(mover_bits):
            return Position(
                current=self.opponent,
                opponent=mover_bits,
                heights=tuple(heights),
                ply=next_ply,
                terminal_value=-1.0,
            )
        if next_ply == CELL_COUNT:
            return Position(
                current=self.opponent,
                opponent=mover_bits,
                heights=tuple(heights),
                ply=next_ply,
                terminal_value=0.0,
            )
        return Position(current=self.opponent, opponent=mover_bits, heights=tuple(heights), ply=next_ply)

    def transform(self, symmetry: int) -> Position:
        perm = symmetry_action_permutation(symmetry)
        new_heights = [0] * ACTION_SIZE
        for old_action, new_action in enumerate(perm):
            new_heights[new_action] = self.heights[old_action]
        return replace(
            self,
            current=transform_bits(self.current, symmetry),
            opponent=transform_bits(self.opponent, symmetry),
            heights=tuple(new_heights),
        )


def initial_position() -> Position:
    return Position()


def _symmetry_xy(symmetry: int, x: int, y: int) -> tuple[int, int]:
    n = BOARD_SIZE - 1
    transforms = (
        lambda a, b: (a, b),
        lambda a, b: (n - b, a),
        lambda a, b: (n - a, n - b),
        lambda a, b: (b, n - a),
        lambda a, b: (n - a, b),
        lambda a, b: (a, n - b),
        lambda a, b: (b, a),
        lambda a, b: (n - b, n - a),
    )
    if symmetry < 0 or symmetry >= len(transforms):
        raise ValueError(f"symmetry out of range: {symmetry}")
    return transforms[symmetry](x, y)


@lru_cache(maxsize=None)
def symmetry_action_permutation(symmetry: int) -> tuple[int, ...]:
    perm: list[int] = []
    for action in range(ACTION_SIZE):
        x, y = action_to_xy(action)
        nx, ny = _symmetry_xy(symmetry, x, y)
        perm.append(ny * BOARD_SIZE + nx)
    return tuple(perm)


@lru_cache(maxsize=None)
def inverse_symmetry_action_permutation(symmetry: int) -> tuple[int, ...]:
    perm = symmetry_action_permutation(symmetry)
    inv = [0] * ACTION_SIZE
    for old, new in enumerate(perm):
        inv[new] = old
    return tuple(inv)


def transform_bits(bits: int, symmetry: int) -> int:
    result = 0
    for z in range(BOARD_SIZE):
        for action in range(ACTION_SIZE):
            old_index = z * ACTION_SIZE + action
            if not (bits & (1 << old_index)):
                continue
            new_action = symmetry_action_permutation(symmetry)[action]
            result |= 1 << (z * ACTION_SIZE + new_action)
    return result


def transform_action_values(values: np.ndarray, symmetry: int) -> np.ndarray:
    if values.shape[-1] != ACTION_SIZE:
        raise ValueError(f"expected last dimension {ACTION_SIZE}, got {values.shape[-1]}")
    out = np.zeros_like(values)
    perm = symmetry_action_permutation(symmetry)
    for old_action, new_action in enumerate(perm):
        out[..., new_action] = values[..., old_action]
    return out


def transform_legal_mask(mask: int, symmetry: int) -> int:
    out = 0
    for old_action, new_action in enumerate(symmetry_action_permutation(symmetry)):
        if mask & (1 << old_action):
            out |= 1 << new_action
    return out


def encode_positions(
    positions: list[Position] | tuple[Position, ...],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    planes = torch.zeros((len(positions), 2, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=dtype, device=device)
    for batch_index, position in enumerate(positions):
        for channel, bits in enumerate((position.current, position.opponent)):
            for index in range(CELL_COUNT):
                if bits & (1 << index):
                    z = index // ACTION_SIZE
                    rem = index % ACTION_SIZE
                    y = rem // BOARD_SIZE
                    x = rem % BOARD_SIZE
                    planes[batch_index, channel, z, y, x] = 1.0
    return planes


def position_from_planes(current: np.ndarray, opponent: np.ndarray, heights: Iterable[int], ply: int) -> Position:
    current_bits = 0
    opponent_bits = 0
    for z in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                bit = 1 << xyz_to_bit_index(x, y, z)
                if current[z, y, x]:
                    current_bits |= bit
                if opponent[z, y, x]:
                    opponent_bits |= bit
    return Position(current=current_bits, opponent=opponent_bits, heights=tuple(int(h) for h in heights), ply=ply)
