"""Action-space symmetries for the square 4x4 column grid."""

from typing import Callable, List, Tuple

import torch

from connect4_zero.game.constants import ACTION_SIZE, BOARD_SIZE, INDEX_DTYPE
from connect4_zero.game.geometry import DeviceLike

XYTransform = Callable[[int, int], Tuple[int, int]]


def make_symmetry_permutations(device: DeviceLike = None) -> torch.Tensor:
    """Return the 8 D4 action permutations as ``(8, 16)``.

    Each row maps an original action index to the transformed action index.
    """
    transforms = _d4_transforms()
    permutations: List[List[int]] = []

    for transform in transforms:
        row: List[int] = []
        for action in range(ACTION_SIZE):
            x, y = divmod(action, BOARD_SIZE)
            tx, ty = transform(x, y)
            row.append(tx * BOARD_SIZE + ty)
        permutations.append(row)

    return torch.tensor(permutations, dtype=INDEX_DTYPE, device=device)


def _d4_transforms() -> List[XYTransform]:
    return [
        lambda x, y: (x, y),
        lambda x, y: (y, BOARD_SIZE - 1 - x),
        lambda x, y: (BOARD_SIZE - 1 - x, BOARD_SIZE - 1 - y),
        lambda x, y: (BOARD_SIZE - 1 - y, x),
        lambda x, y: (x, BOARD_SIZE - 1 - y),
        lambda x, y: (BOARD_SIZE - 1 - y, BOARD_SIZE - 1 - x),
        lambda x, y: (BOARD_SIZE - 1 - x, y),
        lambda x, y: (y, x),
    ]
