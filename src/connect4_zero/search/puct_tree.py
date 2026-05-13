"""Explicit tree objects for AlphaZero PUCT search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE
from connect4_zero.search.tree import legal_actions_for_state, terminal_value_for_state


@dataclass
class PUCTNode:
    """One PUCT node, valued from this node's player-to-move perspective."""

    state: Connect4x4x4Batch
    parent: Optional["PUCTNode"]
    parent_action: Optional[int]
    legal_actions: Tuple[int, ...]
    prior: float = 1.0
    terminal_value: Optional[float] = None
    visits: int = 0
    value_sum: float = 0.0
    depth: int = 0
    child_priors: torch.Tensor = field(default_factory=lambda: torch.zeros(ACTION_SIZE, dtype=torch.float32))
    evaluation_value: float = 0.0
    is_expanded: bool = False
    pending_actions: set[int] = field(default_factory=set)
    children: List[Optional["PUCTNode"]] = field(
        default_factory=lambda: [None for _ in range(ACTION_SIZE)]
    )

    @property
    def is_terminal(self) -> bool:
        return self.terminal_value is not None

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def selectable_actions(self) -> Tuple[int, ...]:
        return tuple(action for action in self.legal_actions if action not in self.pending_actions)


@dataclass
class PUCTSearchTree:
    """A single-root PUCT tree."""

    root: PUCTNode
    num_nodes: int = 1
    max_depth: int = 0

    @classmethod
    def from_state(cls, state: Connect4x4x4Batch) -> "PUCTSearchTree":
        _validate_single_state(state)
        root_state = state.clone()
        root = PUCTNode(
            state=root_state,
            parent=None,
            parent_action=None,
            legal_actions=legal_actions_for_state(root_state),
            terminal_value=terminal_value_for_state(root_state),
        )
        return cls(root=root)

    def reserve_child(self, parent: PUCTNode, action: int) -> None:
        if parent.is_terminal:
            raise RuntimeError("cannot expand a terminal node")
        if action not in parent.legal_actions:
            raise RuntimeError(f"cannot reserve illegal action {action}")
        if parent.children[action] is not None:
            raise RuntimeError(f"cannot reserve already expanded action {action}")
        if action in parent.pending_actions:
            raise RuntimeError(f"cannot reserve already pending action {action}")
        parent.pending_actions.add(action)

    def attach_child(
        self,
        parent: PUCTNode,
        action: int,
        child_state: Connect4x4x4Batch,
        terminal_value: Optional[float],
    ) -> PUCTNode:
        if parent.children[action] is not None:
            parent.pending_actions.discard(action)
            return parent.children[action]
        if action not in parent.legal_actions:
            parent.pending_actions.discard(action)
            raise RuntimeError(f"cannot attach illegal action {action}")
        _validate_single_state(child_state)

        parent.pending_actions.discard(action)
        child = PUCTNode(
            state=child_state,
            parent=parent,
            parent_action=action,
            legal_actions=legal_actions_for_state(child_state),
            prior=float(parent.child_priors[action].item()),
            terminal_value=terminal_value,
            depth=parent.depth + 1,
        )
        parent.children[action] = child
        self.num_nodes += 1
        self.max_depth = max(self.max_depth, child.depth)
        return child

    def reuse_child(self, action: int) -> Optional["PUCTSearchTree"]:
        if action < 0 or action >= ACTION_SIZE:
            return None
        child = self.root.children[action]
        if child is None:
            return None
        return self.from_existing_root(child)

    @classmethod
    def from_existing_root(cls, root: PUCTNode) -> "PUCTSearchTree":
        root.parent = None
        root.parent_action = None
        num_nodes, max_depth = _rebase_and_count(root, depth=0)
        return cls(root=root, num_nodes=num_nodes, max_depth=max_depth)


def _validate_single_state(state: Connect4x4x4Batch) -> None:
    if state.batch_size != 1:
        raise ValueError(f"PUCT nodes require batch_size=1, got {state.batch_size}")


def _rebase_and_count(node: PUCTNode, depth: int) -> tuple[int, int]:
    node.depth = depth
    num_nodes = 1
    max_depth = depth
    for child in node.children:
        if child is None:
            continue
        child.parent = node
        child_count, child_max_depth = _rebase_and_count(child, depth=depth + 1)
        num_nodes += child_count
        max_depth = max(max_depth, child_max_depth)
    return num_nodes, max_depth
