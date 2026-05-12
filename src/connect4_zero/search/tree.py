"""Explicit tree objects for batched deep MCTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from connect4_zero.game import Connect4x4x4Batch, StepResult
from connect4_zero.game.constants import ACTION_SIZE


@dataclass
class TreeNode:
    """One MCTS node, valued from this node's player-to-move perspective."""

    state: Connect4x4x4Batch
    parent: Optional["TreeNode"]
    parent_action: Optional[int]
    legal_actions: Tuple[int, ...]
    terminal_value: Optional[float] = None
    visits: int = 0
    value_sum: float = 0.0
    virtual_visits: int = 0
    virtual_value_sum: float = 0.0
    depth: int = 0
    children: List[Optional["TreeNode"]] = field(
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

    @property
    def effective_visits(self) -> int:
        return self.visits + self.virtual_visits

    @property
    def effective_mean_value(self) -> float:
        visits = self.effective_visits
        if visits == 0:
            return 0.0
        return (self.value_sum + self.virtual_value_sum) / visits

    @property
    def is_fully_expanded(self) -> bool:
        return all(self.children[action] is not None for action in self.legal_actions)

    def unexpanded_actions(self) -> Tuple[int, ...]:
        return tuple(action for action in self.legal_actions if self.children[action] is None)


@dataclass
class SearchTree:
    """A single-root game tree for one current self-play position."""

    root: TreeNode
    num_nodes: int = 1
    max_depth: int = 0

    @classmethod
    def from_state(cls, state: Connect4x4x4Batch) -> "SearchTree":
        """Create a tree whose root is a cloned single-state batch."""
        _validate_single_state(state)
        root_state = state.clone()
        terminal_value = terminal_value_for_state(root_state)
        root = TreeNode(
            state=root_state,
            parent=None,
            parent_action=None,
            legal_actions=legal_actions_for_state(root_state),
            terminal_value=terminal_value,
        )
        return cls(root=root)

    def expand_child(self, parent: TreeNode, action: int) -> TreeNode:
        """Expand ``parent/action`` and return the resulting child node."""
        if parent.is_terminal:
            raise RuntimeError("cannot expand a terminal node")
        if action not in parent.legal_actions:
            raise RuntimeError(f"cannot expand illegal action {action}")

        existing = parent.children[action]
        if existing is not None:
            return existing

        child_state = parent.state.clone()
        result = child_state.step(torch.tensor([action], dtype=torch.long, device=child_state.device))
        if not bool(result.legal[0].item()):
            raise RuntimeError(f"engine rejected legal action {action}")

        child = TreeNode(
            state=child_state,
            parent=parent,
            parent_action=action,
            legal_actions=legal_actions_for_state(child_state),
            terminal_value=terminal_value_for_child(result),
            depth=parent.depth + 1,
        )
        parent.children[action] = child
        self.num_nodes += 1
        self.max_depth = max(self.max_depth, child.depth)
        return child

    def reuse_child(self, action: int) -> Optional["SearchTree"]:
        """Return a new tree rooted at the existing child for ``action``."""
        if action < 0 or action >= ACTION_SIZE:
            return None

        child = self.root.children[action]
        if child is None:
            return None
        return self.from_existing_root(child)

    @classmethod
    def from_existing_root(cls, root: TreeNode) -> "SearchTree":
        """Detach ``root`` from its parent and rebase subtree depths."""
        root.parent = None
        root.parent_action = None
        num_nodes, max_depth = _rebase_and_count(root, depth=0)
        return cls(root=root, num_nodes=num_nodes, max_depth=max_depth)


def legal_actions_for_state(state: Connect4x4x4Batch) -> Tuple[int, ...]:
    """Return legal action indices for a single-state batch."""
    _validate_single_state(state)
    return tuple(
        int(action)
        for action in state.legal_mask()[0].nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    )


def terminal_value_for_state(state: Connect4x4x4Batch) -> Optional[float]:
    """Return exact value for a terminal root, from player-to-move perspective."""
    _validate_single_state(state)
    if bool(state.done[0].item()):
        return -1.0 if int(state.outcome[0].item()) == 1 else 0.0
    if not bool(state.legal_mask()[0].any().item()):
        return 0.0
    return None


def terminal_value_for_child(result: StepResult) -> Optional[float]:
    """Return exact child value after a one-state step, if terminal."""
    if bool(result.won[0].item()):
        return -1.0
    if bool(result.draw[0].item()):
        return 0.0
    return None


def _validate_single_state(state: Connect4x4x4Batch) -> None:
    if state.batch_size != 1:
        raise ValueError(f"tree nodes require batch_size=1, got {state.batch_size}")


def _rebase_and_count(node: TreeNode, depth: int) -> Tuple[int, int]:
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
