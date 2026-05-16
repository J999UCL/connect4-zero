from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

BOARD_SIZE = 4
ACTION_SIZE = 16
MAX_PLIES = 64
MATE_BASE = 100_000
INVALID_MOVE_VALUE = -(2**31)
CENTER_ORDER = (5, 6, 9, 10, 1, 2, 4, 7, 8, 11, 13, 14, 0, 3, 12, 15)


def cell_mask(action: int, z: int) -> int:
    if action < 0 or action >= ACTION_SIZE or z < 0 or z >= BOARD_SIZE:
        raise ValueError(f"invalid cell action={action} z={z}")
    return 1 << (z * 16 + action)


def _in_bounds(x: int, y: int, z: int) -> bool:
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and 0 <= z < BOARD_SIZE


def _build_winning_masks() -> tuple[int, ...]:
    masks: set[int] = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if dx < 0 or (dx == 0 and dy < 0) or (dx == 0 and dy == 0 and dz < 0):
                    continue
                for x in range(BOARD_SIZE):
                    for y in range(BOARD_SIZE):
                        for z in range(BOARD_SIZE):
                            if _in_bounds(x - dx, y - dy, z - dz):
                                continue
                            if not _in_bounds(x + 3 * dx, y + 3 * dy, z + 3 * dz):
                                continue
                            mask = 0
                            for i in range(BOARD_SIZE):
                                action = (y + i * dy) * BOARD_SIZE + (x + i * dx)
                                mask |= cell_mask(action, z + i * dz)
                            masks.add(mask)
    return tuple(sorted(masks))


WINNING_MASKS = _build_winning_masks()


def has_winning_line(bits: int) -> bool:
    return any((bits & mask) == mask for mask in WINNING_MASKS)


@dataclass(frozen=True, slots=True)
class Position:
    current: int = 0
    opponent: int = 0
    heights: tuple[int, ...] = (0,) * ACTION_SIZE
    ply: int = 0

    def legal_actions(self) -> tuple[int, ...]:
        if self.terminal_value() is not None:
            return ()
        return tuple(action for action in CENTER_ORDER if self.heights[action] < BOARD_SIZE)

    def legal_mask(self) -> int:
        mask = 0
        for action in self.legal_actions():
            mask |= 1 << action
        return mask

    def terminal_value(self) -> float | None:
        if has_winning_line(self.opponent):
            return -1.0
        if self.ply >= MAX_PLIES:
            return 0.0
        return None

    def play(self, action: int) -> Position:
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action}")
        heights = list(self.heights)
        z = heights[action]
        heights[action] += 1
        placed = self.current | cell_mask(action, z)
        return Position(
            current=self.opponent,
            opponent=placed,
            heights=tuple(heights),
            ply=self.ply + 1,
        )


def position_from_actions(actions: list[int] | tuple[int, ...]) -> Position:
    position = Position()
    for action in actions:
        position = position.play(action)
    return position


def _line_score(mine: int, theirs: int) -> int:
    score = 0
    for mask in WINNING_MASKS:
        own = (mine & mask).bit_count()
        opp = (theirs & mask).bit_count()
        if own and opp:
            continue
        if own == 1:
            score += 1
        elif own == 2:
            score += 8
        elif own == 3:
            score += 80
        elif own == 4:
            score += MATE_BASE
        elif opp == 1:
            score -= 1
        elif opp == 2:
            score -= 10
        elif opp == 3:
            score -= 120
        elif opp == 4:
            score -= MATE_BASE
    return score


def evaluate_position(position: Position) -> int:
    terminal = position.terminal_value()
    if terminal is not None:
        return int(terminal * MATE_BASE)
    return _line_score(position.current, position.opponent)


@lru_cache(maxsize=1_000_000)
def _negamax_cached(
    current: int,
    opponent: int,
    heights: tuple[int, ...],
    ply: int,
    depth: int,
    ply_distance: int,
    alpha: int,
    beta: int,
) -> int:
    position = Position(current=current, opponent=opponent, heights=heights, ply=ply)
    terminal = position.terminal_value()
    if terminal is not None:
        if terminal < 0.0:
            return -MATE_BASE + ply_distance
        return 0
    if depth <= 0:
        return evaluate_position(position)

    best = -MATE_BASE * 10
    for action in position.legal_actions():
        child = position.play(action)
        value = -_negamax_cached(
            child.current,
            child.opponent,
            child.heights,
            child.ply,
            depth - 1,
            ply_distance + 1,
            -beta,
            -alpha,
        )
        best = max(best, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return best


def solve(position: Position, max_depth: int) -> tuple[int, int, list[int]]:
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")
    move_values = [INVALID_MOVE_VALUE] * ACTION_SIZE
    best_value = -MATE_BASE * 10
    best_action = -1
    alpha = -MATE_BASE * 10
    beta = MATE_BASE * 10
    for action in position.legal_actions():
        child = position.play(action)
        value = -_negamax_cached(
            child.current,
            child.opponent,
            child.heights,
            child.ply,
            max_depth - 1,
            1,
            -beta,
            -alpha,
        )
        move_values[action] = value
        if value > best_value:
            best_value = value
            best_action = action
        alpha = max(alpha, value)
    if best_action < 0:
        return evaluate_position(position), -1, move_values
    return best_value, best_action, move_values
