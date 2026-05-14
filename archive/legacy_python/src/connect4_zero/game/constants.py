"""Constants shared by the 4x4x4 Connect Four game engine."""

from typing import Final

import torch

BOARD_SIZE: Final[int] = 4
WIN_LENGTH: Final[int] = BOARD_SIZE
BOARD_CELLS: Final[int] = BOARD_SIZE**3
ACTION_SIZE: Final[int] = BOARD_SIZE**2

EMPTY: Final[int] = 0
CURRENT_PLAYER: Final[int] = 1
OPPONENT_PLAYER: Final[int] = -1

BOARD_DTYPE: Final[torch.dtype] = torch.int8
INDEX_DTYPE: Final[torch.dtype] = torch.long
