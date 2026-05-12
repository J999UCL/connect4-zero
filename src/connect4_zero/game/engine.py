"""Batched PyTorch engine for gravity-based 4x4x4 Connect Four."""

from typing import Optional, Union

import torch

from connect4_zero.game.constants import (
    ACTION_SIZE,
    BOARD_DTYPE,
    BOARD_SIZE,
    CURRENT_PLAYER,
    OPPONENT_PLAYER,
    WIN_LENGTH,
)
from connect4_zero.game.geometry import make_win_line_flat_indices
from connect4_zero.game.types import StepResult

DeviceLike = Optional[Union[str, torch.device]]


class Connect4x4x4Batch:
    """Vectorized batch of canonical 4x4x4 Connect Four games.

    Board coordinates are ordered as ``(x, y, z)`` with ``z=0`` at the bottom of
    a gravity column. The player to move is always represented by ``+1`` and the
    opponent by ``-1``. After every legal non-terminal move, the board is
    multiplied by ``-1`` so the next game state is canonical for the next player.
    """

    def __init__(self, batch_size: int, device: DeviceLike = None) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.batch_size = int(batch_size)
        self.device = torch.device("cpu" if device is None else device)
        self._win_line_flat_indices = make_win_line_flat_indices(device=self.device)
        self._batch_indices = torch.arange(self.batch_size, device=self.device)

        self.board = torch.zeros(
            (self.batch_size, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE),
            dtype=BOARD_DTYPE,
            device=self.device,
        )
        self.heights = torch.zeros(
            (self.batch_size, BOARD_SIZE, BOARD_SIZE),
            dtype=BOARD_DTYPE,
            device=self.device,
        )
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        # 1 means the player who just moved won; 0 means unfinished or draw.
        self.outcome = torch.zeros(self.batch_size, dtype=BOARD_DTYPE, device=self.device)

    def reset(self) -> None:
        """Reset all games in the batch to the empty starting state."""
        self.board.zero_()
        self.heights.zero_()
        self.done.zero_()
        self.outcome.zero_()

    def legal_mask(self) -> torch.Tensor:
        """Return a ``(B, 16)`` mask of currently legal column actions."""
        columns_open = self.heights.reshape(self.batch_size, ACTION_SIZE) < BOARD_SIZE
        games_active = ~self.done
        return columns_open & games_active.unsqueeze(1)

    def step(self, actions: torch.Tensor) -> StepResult:
        """Apply one column action per game and return post-step status.

        Illegal moves do not mutate state. A move is illegal if the action is
        out of range, the selected column is full, or the game is already done.
        """
        actions = self._validate_actions(actions)
        safe_actions = actions.clamp(0, ACTION_SIZE - 1)
        x = safe_actions // BOARD_SIZE
        y = safe_actions % BOARD_SIZE

        in_range = (actions >= 0) & (actions < ACTION_SIZE)
        selected_heights = self.heights[self._batch_indices, x, y]
        column_open = selected_heights < BOARD_SIZE
        legal = in_range & column_open & ~self.done

        won = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        draw = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        legal_indices = torch.nonzero(legal, as_tuple=False).flatten()
        if legal_indices.numel() > 0:
            self._place_legal_moves(legal_indices, x, y)

            won = legal & self.check_wins(player=CURRENT_PLAYER)
            draw = legal & ~won & self.is_draw()
            newly_done = won | draw

            self.done |= newly_done
            self.outcome[won] = CURRENT_PLAYER

            flip_mask = legal & ~newly_done
            self.board[flip_mask] *= OPPONENT_PLAYER

        return StepResult(
            legal=legal.clone(),
            won=won.clone(),
            draw=draw.clone(),
            done=self.done.clone(),
            outcome=self.outcome.clone(),
        )

    def check_wins(
        self,
        board: Optional[torch.Tensor] = None,
        player: int = CURRENT_PLAYER,
    ) -> torch.Tensor:
        """Return a ``(B,)`` mask for whether ``player`` has any winning line."""
        if player not in (CURRENT_PLAYER, OPPONENT_PLAYER):
            raise ValueError(f"player must be {CURRENT_PLAYER} or {OPPONENT_PLAYER}, got {player}")

        board = self.board if board is None else board
        self._validate_board(board)

        indices = self._indices_for_board(board)
        flat_board = board.reshape(board.shape[0], -1)
        flat_indices = indices.reshape(1, -1).expand(board.shape[0], -1)
        line_values = torch.gather(flat_board, dim=1, index=flat_indices)
        line_values = line_values.reshape(board.shape[0], -1, WIN_LENGTH)
        target = int(player) * WIN_LENGTH
        return line_values.sum(dim=2).eq(target).any(dim=1)

    def is_draw(self) -> torch.Tensor:
        """Return a ``(B,)`` mask for full boards with no winning line."""
        full = self.heights.reshape(self.batch_size, ACTION_SIZE).ge(BOARD_SIZE).all(dim=1)
        current_wins = self.check_wins(player=CURRENT_PLAYER)
        opponent_wins = self.check_wins(player=OPPONENT_PLAYER)
        return full & ~current_wins & ~opponent_wins

    def clone(self) -> "Connect4x4x4Batch":
        """Return a deep copy of this batch on the same device."""
        cloned = Connect4x4x4Batch(self.batch_size, device=self.device)
        cloned.board = self.board.clone()
        cloned.heights = self.heights.clone()
        cloned.done = self.done.clone()
        cloned.outcome = self.outcome.clone()
        return cloned

    def to(self, device: Union[str, torch.device]) -> "Connect4x4x4Batch":
        """Return a deep copy of this batch moved to ``device``."""
        moved = Connect4x4x4Batch(self.batch_size, device=device)
        moved.board = self.board.to(device=moved.device)
        moved.heights = self.heights.to(device=moved.device)
        moved.done = self.done.to(device=moved.device)
        moved.outcome = self.outcome.to(device=moved.device)
        return moved

    def _place_legal_moves(
        self,
        legal_indices: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        legal_x = x[legal_indices]
        legal_y = y[legal_indices]
        legal_z = self.heights[legal_indices, legal_x, legal_y].to(dtype=torch.long)

        self.board[legal_indices, legal_x, legal_y, legal_z] = CURRENT_PLAYER
        self.heights[legal_indices, legal_x, legal_y] += 1

    def _validate_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device)
        actions = actions.to(device=self.device, dtype=torch.long)
        if actions.shape != (self.batch_size,):
            raise ValueError(f"actions must have shape ({self.batch_size},), got {tuple(actions.shape)}")
        return actions

    def _validate_board(self, board: torch.Tensor) -> None:
        if board.ndim != 4:
            raise ValueError(
                "board must have shape "
                f"(B, {BOARD_SIZE}, {BOARD_SIZE}, {BOARD_SIZE}), got {tuple(board.shape)}"
            )

        expected_shape = (board.shape[0], BOARD_SIZE, BOARD_SIZE, BOARD_SIZE)
        if tuple(board.shape) != expected_shape:
            raise ValueError(
                "board must have shape "
                f"(B, {BOARD_SIZE}, {BOARD_SIZE}, {BOARD_SIZE}), got {tuple(board.shape)}"
            )

    def _indices_for_board(self, board: torch.Tensor) -> torch.Tensor:
        if board.device == self._win_line_flat_indices.device:
            return self._win_line_flat_indices
        return make_win_line_flat_indices(device=board.device)
