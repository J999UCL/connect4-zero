from __future__ import annotations

from enum import IntEnum

import numpy as np

from c4zero_tools.datasets import Sample


BOARD_SIZE = 4
NUM_ACTIONS = 16


class Symmetry(IntEnum):
    IDENTITY = 0
    ROT90 = 1
    ROT180 = 2
    ROT270 = 3
    MIRROR_X = 4
    MIRROR_Y = 5
    DIAGONAL = 6
    ANTI_DIAGONAL = 7


def transform_sample(sample: Sample, symmetry: int | Symmetry) -> Sample:
    symmetry = Symmetry(symmetry)
    permutation = action_permutation(symmetry)
    current_bits = transform_bits(sample.current_bits, symmetry)
    opponent_bits = transform_bits(sample.opponent_bits, symmetry)
    policy = transform_policy(sample.policy, permutation)
    visits = transform_visits(sample.visit_counts, permutation)
    legal_mask = transform_legal_mask(sample.legal_mask, permutation)
    action = permutation[sample.action] if 0 <= sample.action < NUM_ACTIONS else sample.action
    return Sample(
        current_bits=current_bits,
        opponent_bits=opponent_bits,
        heights=heights_from_bits(current_bits | opponent_bits),
        ply=sample.ply,
        game_id=sample.game_id,
        legal_mask=legal_mask,
        action=action,
        policy=policy,
        visit_counts=visits,
        value=sample.value,
    )


def action_permutation(symmetry: Symmetry) -> tuple[int, ...]:
    out = [0] * NUM_ACTIONS
    for action in range(NUM_ACTIONS):
        x, y = action % BOARD_SIZE, action // BOARD_SIZE
        mapped_x, mapped_y = map_xy(x, y, symmetry)
        out[action] = mapped_y * BOARD_SIZE + mapped_x
    return tuple(out)


def transform_bits(bits: int, symmetry: Symmetry) -> int:
    out = 0
    for z in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                source = cell_index(x, y, z)
                if bits & (1 << source):
                    mapped_x, mapped_y = map_xy(x, y, symmetry)
                    out |= 1 << cell_index(mapped_x, mapped_y, z)
    return out


def transform_policy(policy: np.ndarray, permutation: tuple[int, ...]) -> np.ndarray:
    out = np.zeros_like(policy)
    for action, mapped in enumerate(permutation):
        out[mapped] = policy[action]
    return out.astype(np.float32, copy=False)


def transform_visits(visits: np.ndarray, permutation: tuple[int, ...]) -> np.ndarray:
    out = np.zeros_like(visits)
    for action, mapped in enumerate(permutation):
        out[mapped] = visits[action]
    return out.astype(np.uint32, copy=False)


def transform_legal_mask(mask: int, permutation: tuple[int, ...]) -> int:
    out = 0
    for action, mapped in enumerate(permutation):
        if mask & (1 << action):
            out |= 1 << mapped
    return out


def heights_from_bits(occupancy: int) -> tuple[int, ...]:
    heights = []
    for action in range(NUM_ACTIONS):
        height = 0
        while height < BOARD_SIZE and (occupancy & (1 << cell_index(action % BOARD_SIZE, action // BOARD_SIZE, height))):
            height += 1
        heights.append(height)
    return tuple(heights)


def map_xy(x: int, y: int, symmetry: Symmetry) -> tuple[int, int]:
    if symmetry == Symmetry.IDENTITY:
        return x, y
    if symmetry == Symmetry.ROT90:
        return 3 - y, x
    if symmetry == Symmetry.ROT180:
        return 3 - x, 3 - y
    if symmetry == Symmetry.ROT270:
        return y, 3 - x
    if symmetry == Symmetry.MIRROR_X:
        return 3 - x, y
    if symmetry == Symmetry.MIRROR_Y:
        return x, 3 - y
    if symmetry == Symmetry.DIAGONAL:
        return y, x
    if symmetry == Symmetry.ANTI_DIAGONAL:
        return 3 - y, 3 - x
    raise ValueError(f"unknown symmetry: {symmetry}")


def cell_index(x: int, y: int, z: int) -> int:
    return z * 16 + y * 4 + x
