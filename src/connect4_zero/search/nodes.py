"""Node storage boundaries for MCTS.

The first implementation stores a plain tree. MCTS talks through ``NodeStore``
so a future Zobrist-backed DAG can reuse states without changing the search
algorithm's selection/evaluation/backprop logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple

from connect4_zero.game import Connect4x4x4Batch


@dataclass
class SearchNode:
    """A search node whose values are from this node's player-to-move perspective."""

    state: Connect4x4x4Batch
    legal_actions: Tuple[int, ...]
    terminal_value: Optional[float] = None
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[int, "SearchNode"] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.terminal_value is not None

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.legal_actions)


class NodeStore(Protocol):
    """Storage boundary for MCTS nodes."""

    def create_root(self, state: Connect4x4x4Batch) -> SearchNode:
        """Create the root node for a search."""

    def get_or_create_child(
        self,
        parent: SearchNode,
        action: int,
        state: Connect4x4x4Batch,
        terminal_value: Optional[float],
    ) -> SearchNode:
        """Return the child for ``parent/action``, creating it if needed."""


class TreeNodeStore:
    """Simple one-search tree storage."""

    def create_root(self, state: Connect4x4x4Batch) -> SearchNode:
        return self._create_node(state, terminal_value=None)

    def get_or_create_child(
        self,
        parent: SearchNode,
        action: int,
        state: Connect4x4x4Batch,
        terminal_value: Optional[float],
    ) -> SearchNode:
        if action not in parent.children:
            parent.children[action] = self._create_node(state, terminal_value=terminal_value)
        return parent.children[action]

    def _create_node(
        self,
        state: Connect4x4x4Batch,
        terminal_value: Optional[float],
    ) -> SearchNode:
        legal_actions = tuple(
            int(action)
            for action in state.legal_mask()[0].nonzero(as_tuple=False).flatten().cpu().tolist()
        )
        return SearchNode(
            state=state,
            legal_actions=legal_actions,
            terminal_value=terminal_value,
        )
