"""Batched deep MCTS with GPU-batched leaf rollout evaluation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

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
    virtual_loss_path: List[TreeNode]

    @property
    def leaf(self) -> TreeNode:
        return self.path[-1]


@dataclass
class ExpansionRequest:
    """A reserved child expansion waiting for batched engine stepping."""

    tree_index: int
    tree: SearchTree
    parent: TreeNode
    parent_path: List[TreeNode]
    action: int


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
        self.last_reused_roots = 0
        self.last_fresh_roots = 0
        self.last_expansion_batch_sizes: List[int] = []
        self.last_expanded_children = 0
        self.last_timing_seconds = self._empty_timing()

    def search_batch(self, roots: Connect4x4x4Batch) -> BatchedSearchResult:
        """Search a batch of roots without mutating the caller's states."""
        return self.search_batch_with_trees(roots, root_trees=None)

    def search_batch_with_trees(
        self,
        roots: Connect4x4x4Batch,
        root_trees: Optional[Sequence[Optional[SearchTree]]] = None,
    ) -> BatchedSearchResult:
        """Search roots, reusing matching trees when supplied."""
        self._reset_diagnostics([])
        started_at = time.perf_counter()
        search_roots = self._prepare_roots(roots)
        trees = self._prepare_trees(search_roots, root_trees)
        self.last_trees = trees
        self._add_timing("prepare_trees", started_at)

        remaining = [
            0 if tree.root.is_terminal else self.config.simulations_per_root
            for tree in trees
        ]
        total_remaining = sum(remaining)

        while total_remaining > 0:
            expansion_requests: List[ExpansionRequest] = []

            while len(expansion_requests) < self.config.max_leaf_batch_size and total_remaining > 0:
                made_progress = False
                for tree_index, tree in enumerate(trees):
                    if remaining[tree_index] <= 0:
                        continue

                    selection_started_at = time.perf_counter()
                    selected = self._select_path_or_expansion(tree_index, tree)
                    self._add_timing("select", selection_started_at)
                    if selected is None:
                        continue

                    remaining[tree_index] -= 1
                    total_remaining -= 1
                    made_progress = True

                    if isinstance(selected, ExpansionRequest):
                        self._apply_virtual_loss(selected.parent_path)
                        expansion_requests.append(selected)
                    elif selected.terminal_value is None:
                        raise RuntimeError("selected path without expansion must be terminal")
                    else:
                        self._backpropagate_selected(selected, selected.terminal_value)
                        self.last_terminal_evaluations += 1

                    if len(expansion_requests) >= self.config.max_leaf_batch_size or total_remaining == 0:
                        break

                if not made_progress:
                    break

            if not expansion_requests:
                if total_remaining > 0:
                    raise RuntimeError("MCTS selection stalled with no expandable or terminal paths")
                break

            pending, terminal_paths = self._expand_requests(expansion_requests)
            for selected in terminal_paths:
                self._backpropagate_selected(selected, selected.terminal_value)
                self.last_terminal_evaluations += 1
            if pending:
                self._evaluate_pending(pending)

        build_started_at = time.perf_counter()
        result = self._build_result(trees, device=search_roots.device)
        self._add_timing("build_result", build_started_at)
        return result

    def advance_tree(self, tree: SearchTree, action: int) -> Optional[SearchTree]:
        """Reuse the searched child tree after ``action`` is played."""
        return tree.reuse_child(int(action))

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

    def _prepare_trees(
        self,
        roots: Connect4x4x4Batch,
        root_trees: Optional[Sequence[Optional[SearchTree]]],
    ) -> List[SearchTree]:
        if root_trees is not None and len(root_trees) != roots.batch_size:
            raise ValueError(
                f"root_trees must have length {roots.batch_size}, got {len(root_trees)}"
            )

        trees: List[SearchTree] = []
        reused = 0
        fresh = 0
        for index in range(roots.batch_size):
            root_state = self._slice_state(roots, index)
            candidate = root_trees[index] if root_trees is not None else None
            if candidate is not None and self._tree_matches_state(candidate, root_state):
                trees.append(candidate)
                reused += 1
            else:
                trees.append(SearchTree.from_state(root_state))
                fresh += 1

        self.last_reused_roots = reused
        self.last_fresh_roots = fresh
        return trees

    def _tree_matches_state(self, tree: SearchTree, state: Connect4x4x4Batch) -> bool:
        root = tree.root.state
        if root.device != state.device:
            return False
        return (
            torch.equal(root.board, state.board)
            and torch.equal(root.heights, state.heights)
            and torch.equal(root.done, state.done)
            and torch.equal(root.outcome, state.outcome)
        )

    def _select_path_or_expansion(
        self,
        tree_index: int,
        tree: SearchTree,
    ) -> SelectedPath | ExpansionRequest | None:
        node = tree.root
        path = [node]

        while True:
            if node.is_terminal:
                return SelectedPath(
                    tree_index=tree_index,
                    path=path,
                    terminal_value=node.terminal_value,
                    virtual_loss_path=[],
                )

            unexpanded_actions = node.unexpanded_actions()
            if unexpanded_actions:
                action = unexpanded_actions[0]
                tree.reserve_child(node, action)
                return ExpansionRequest(
                    tree_index=tree_index,
                    tree=tree,
                    parent=node,
                    parent_path=list(path),
                    action=action,
                )

            if not node.legal_actions:
                node.terminal_value = 0.0
                return SelectedPath(
                    tree_index=tree_index,
                    path=path,
                    terminal_value=0.0,
                    virtual_loss_path=[],
                )

            child = self._select_child(node)
            if child is None:
                return None
            node = child
            path.append(node)

    def _select_child(self, node: TreeNode) -> Optional[TreeNode]:
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

    def _expand_requests(
        self,
        requests: List[ExpansionRequest],
    ) -> tuple[List[SelectedPath], List[SelectedPath]]:
        started_at = time.perf_counter()
        parent_states = self._make_parent_batch(requests)
        actions = torch.tensor(
            [request.action for request in requests],
            dtype=torch.long,
            device=parent_states.device,
        )
        result = parent_states.step(actions)
        if not bool(result.legal.all().item()):
            for request in requests:
                request.parent.pending_actions.discard(request.action)
            raise RuntimeError("batched expansion attempted an illegal action")

        rollout_paths: List[SelectedPath] = []
        terminal_paths: List[SelectedPath] = []
        for index, request in enumerate(requests):
            child_state = self._slice_state(parent_states, index)
            terminal_value = self._terminal_value_at(result, index)
            child = request.tree.attach_child(
                parent=request.parent,
                action=request.action,
                child_state=child_state,
                terminal_value=terminal_value,
            )
            selected = SelectedPath(
                tree_index=request.tree_index,
                path=request.parent_path + [child],
                terminal_value=terminal_value,
                virtual_loss_path=request.parent_path,
            )
            if terminal_value is None:
                rollout_paths.append(selected)
            else:
                terminal_paths.append(selected)

        self.last_expanded_children += len(requests)
        self.last_expansion_batch_sizes.append(len(requests))
        self._add_timing("expand", started_at)
        return rollout_paths, terminal_paths

    def _evaluate_pending(self, pending: List[SelectedPath]) -> None:
        started_at = time.perf_counter()
        states = self._make_leaf_batch(pending)
        values = self.evaluator.evaluate_batch(states).detach().cpu().tolist()
        self._add_timing("rollout_eval", started_at)
        self.last_leaf_evaluations += len(pending)
        self.last_leaf_batch_sizes.append(len(pending))

        for selected, value in zip(pending, values):
            self._backpropagate_selected(selected, float(value))

    def _backpropagate_selected(self, selected: SelectedPath, leaf_value: float | None) -> None:
        if leaf_value is None:
            raise RuntimeError("cannot backpropagate an unresolved leaf value")
        self._clear_virtual_loss(selected.virtual_loss_path)
        started_at = time.perf_counter()
        self._backpropagate(selected.path, float(leaf_value))
        self._add_timing("backprop", started_at)

    def _make_parent_batch(self, requests: List[ExpansionRequest]) -> Connect4x4x4Batch:
        device = requests[0].parent.state.device
        states = Connect4x4x4Batch(len(requests), device=device)
        states.board = torch.cat([request.parent.state.board for request in requests], dim=0).clone()
        states.heights = torch.cat([request.parent.state.heights for request in requests], dim=0).clone()
        states.done = torch.cat([request.parent.state.done for request in requests], dim=0).clone()
        states.outcome = torch.cat([request.parent.state.outcome for request in requests], dim=0).clone()
        return states

    def _terminal_value_at(self, result, index: int) -> Optional[float]:
        if bool(result.won[index].item()):
            return -1.0
        if bool(result.draw[index].item()):
            return 0.0
        return None

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
        self.last_expansion_batch_sizes = []
        self.last_expanded_children = 0
        self.last_timing_seconds = self._empty_timing()

    def _empty_timing(self) -> dict[str, float]:
        return {
            "prepare_trees": 0.0,
            "select": 0.0,
            "expand": 0.0,
            "rollout_eval": 0.0,
            "backprop": 0.0,
            "build_result": 0.0,
        }

    def _add_timing(self, key: str, started_at: float) -> None:
        self.last_timing_seconds[key] += time.perf_counter() - started_at
