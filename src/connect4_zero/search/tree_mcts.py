"""Batched deep MCTS with GPU-batched leaf rollout evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Protocol

import torch

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE
from connect4_zero.search.rollout import BatchedRandomRolloutEvaluator
from connect4_zero.search.tree import SearchTree, TreeNode
from connect4_zero.search.types import BatchedSearchResult, TreeMCTSConfig


class BatchedLeafEvaluator(Protocol):
    """Evaluate many leaf states from their player-to-move perspectives."""

    def evaluate_batch(self, states: Connect4x4x4Batch) -> torch.Tensor:
        """Return one scalar value per input state."""


@dataclass
class SelectedPath:
    """A selected simulation path waiting for exact or rollout evaluation."""

    tree_index: int
    path: List[TreeNode]
    terminal_value: Optional[float]

    @property
    def leaf(self) -> TreeNode:
        return self.path[-1]


class BatchedTreeMCTS:
    """Run real multi-depth MCTS over a batch of independent root states."""

    def __init__(
        self,
        config: Optional[TreeMCTSConfig] = None,
        evaluator: Optional[BatchedLeafEvaluator] = None,
    ) -> None:
        self.config = config if config is not None else TreeMCTSConfig()
        self.evaluator = evaluator if evaluator is not None else BatchedRandomRolloutEvaluator(
            rollouts_per_state=self.config.rollouts_per_leaf,
            device=self.config.rollout_device,
            seed=self.config.seed,
            max_steps=self.config.max_rollout_steps,
            max_rollouts_per_chunk=self.config.max_rollouts_per_chunk,
        )
        self.last_trees: List[SearchTree] = []
        self.last_leaf_evaluations = 0
        self.last_terminal_evaluations = 0
        self.last_leaf_batch_sizes: List[int] = []

    def search_batch(self, roots: Connect4x4x4Batch) -> BatchedSearchResult:
        """Search a batch of roots without mutating the caller's states."""
        search_roots = self._prepare_roots(roots)
        trees = [SearchTree.from_state(self._slice_state(search_roots, index)) for index in range(search_roots.batch_size)]
        self._reset_diagnostics(trees)

        remaining = [
            0 if tree.root.is_terminal else self.config.simulations_per_root
            for tree in trees
        ]
        total_remaining = sum(remaining)

        while total_remaining > 0:
            pending: List[SelectedPath] = []

            while len(pending) < self.config.max_leaf_batch_size and total_remaining > 0:
                made_progress = False
                for tree_index, tree in enumerate(trees):
                    if remaining[tree_index] <= 0:
                        continue

                    selected = self._select_path(tree_index, tree)
                    remaining[tree_index] -= 1
                    total_remaining -= 1
                    made_progress = True

                    if selected.terminal_value is None:
                        self._apply_virtual_loss(selected.path)
                        pending.append(selected)
                    else:
                        self._backpropagate(selected.path, selected.terminal_value)
                        self.last_terminal_evaluations += 1

                    if len(pending) >= self.config.max_leaf_batch_size or total_remaining == 0:
                        break

                if not made_progress:
                    break

            if pending:
                self._evaluate_pending(pending)

        return self._build_result(trees, device=search_roots.device)

    def _prepare_roots(self, roots: Connect4x4x4Batch) -> Connect4x4x4Batch:
        if self.config.rollout_device is None:
            return roots.clone()

        rollout_device = torch.device(self.config.rollout_device)
        if rollout_device.type in ("cuda", "mps"):
            return roots.to("cpu")
        return roots.to(rollout_device)

    def _slice_state(self, roots: Connect4x4x4Batch, index: int) -> Connect4x4x4Batch:
        state = Connect4x4x4Batch(1, device=roots.device)
        state.board = roots.board[index : index + 1].clone()
        state.heights = roots.heights[index : index + 1].clone()
        state.done = roots.done[index : index + 1].clone()
        state.outcome = roots.outcome[index : index + 1].clone()
        return state

    def _select_path(self, tree_index: int, tree: SearchTree) -> SelectedPath:
        node = tree.root
        path = [node]

        while True:
            if node.is_terminal:
                return SelectedPath(tree_index=tree_index, path=path, terminal_value=node.terminal_value)

            unexpanded_actions = node.unexpanded_actions()
            if unexpanded_actions:
                child = tree.expand_child(node, unexpanded_actions[0])
                path.append(child)
                return SelectedPath(tree_index=tree_index, path=path, terminal_value=child.terminal_value)

            if not node.legal_actions:
                node.terminal_value = 0.0
                return SelectedPath(tree_index=tree_index, path=path, terminal_value=0.0)

            node = self._select_child(node)
            path.append(node)

    def _select_child(self, node: TreeNode) -> TreeNode:
        best_child: Optional[TreeNode] = None
        best_score = -math.inf

        for action in node.legal_actions:
            child = node.children[action]
            if child is None:
                continue

            score = self._uct_score(parent=node, child=child)
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            raise RuntimeError("cannot select a child from a node with no expanded children")
        return best_child

    def _uct_score(self, parent: TreeNode, child: TreeNode) -> float:
        parent_visits = parent.effective_visits
        child_visits = child.effective_visits
        q_value = -child.effective_mean_value
        exploration = self.config.exploration_constant * math.sqrt(
            math.log(parent_visits + 1.0) / (child_visits + 1.0)
        )
        return q_value + exploration

    def _apply_virtual_loss(self, path: List[TreeNode]) -> None:
        if self.config.virtual_loss == 0:
            for node in path:
                node.virtual_visits += 1
            return

        for node in path:
            node.virtual_visits += 1
            node.virtual_value_sum += self.config.virtual_loss

    def _clear_virtual_loss(self, path: List[TreeNode]) -> None:
        for node in path:
            node.virtual_visits -= 1
            node.virtual_value_sum -= self.config.virtual_loss
            if node.virtual_visits < 0:
                raise RuntimeError("virtual visit count became negative")

    def _evaluate_pending(self, pending: List[SelectedPath]) -> None:
        states = self._make_leaf_batch(pending)
        values = self.evaluator.evaluate_batch(states).detach().cpu().tolist()
        self.last_leaf_evaluations += len(pending)
        self.last_leaf_batch_sizes.append(len(pending))

        for selected, value in zip(pending, values):
            self._clear_virtual_loss(selected.path)
            self._backpropagate(selected.path, float(value))

    def _make_leaf_batch(self, pending: List[SelectedPath]) -> Connect4x4x4Batch:
        device = pending[0].leaf.state.device
        states = Connect4x4x4Batch(len(pending), device=device)
        states.board = torch.cat([selected.leaf.state.board for selected in pending], dim=0).clone()
        states.heights = torch.cat([selected.leaf.state.heights for selected in pending], dim=0).clone()
        states.done = torch.cat([selected.leaf.state.done for selected in pending], dim=0).clone()
        states.outcome = torch.cat([selected.leaf.state.outcome for selected in pending], dim=0).clone()
        return states

    def _backpropagate(self, path: List[TreeNode], leaf_value: float) -> None:
        value = float(leaf_value)
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def _build_result(self, trees: List[SearchTree], device: torch.device) -> BatchedSearchResult:
        visit_counts = torch.zeros((len(trees), ACTION_SIZE), dtype=torch.float32, device=device)
        q_values = torch.zeros_like(visit_counts)
        root_values = torch.zeros(len(trees), dtype=torch.float32, device=device)

        for root_index, tree in enumerate(trees):
            root = tree.root
            root_values[root_index] = float(root.terminal_value if root.is_terminal else root.mean_value)

            for action in root.legal_actions:
                child = root.children[action]
                if child is None:
                    continue
                visit_counts[root_index, action] = float(child.visits)
                if child.visits > 0:
                    q_values[root_index, action] = float(-child.mean_value)

        policy = self._policy_from_visits(visit_counts, trees)
        return BatchedSearchResult(
            visit_counts=visit_counts,
            policy=policy,
            q_values=q_values,
            root_values=root_values,
        )

    def _policy_from_visits(self, visit_counts: torch.Tensor, trees: List[SearchTree]) -> torch.Tensor:
        legal_mask = torch.zeros_like(visit_counts, dtype=torch.bool)
        for root_index, tree in enumerate(trees):
            if tree.root.legal_actions:
                legal_mask[root_index, list(tree.root.legal_actions)] = True

        weights = visit_counts.pow(1.0 / self.config.policy_temperature)
        weights = weights.masked_fill(~legal_mask, 0.0)
        totals = weights.sum(dim=1, keepdim=True)
        return torch.where(totals > 0, weights / totals.clamp_min(1.0), torch.zeros_like(weights))

    def _reset_diagnostics(self, trees: List[SearchTree]) -> None:
        self.last_trees = trees
        self.last_leaf_evaluations = 0
        self.last_terminal_evaluations = 0
        self.last_leaf_batch_sizes = []
