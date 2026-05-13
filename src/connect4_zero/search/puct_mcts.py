"""Batched AlphaZero-style PUCT search with neural leaf evaluation."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

import torch

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE
from connect4_zero.search.puct_tree import PUCTNode, PUCTSearchTree
from connect4_zero.search.types import BatchedSearchResult, PUCTMCTSConfig, PolicyValueBatch


class PolicyValueEvaluator(Protocol):
    """Evaluate many states from their player-to-move perspectives."""

    def evaluate_batch(self, states: Connect4x4x4Batch) -> PolicyValueBatch:
        """Return legal priors and scalar values."""


@dataclass
class PUCTSelectedPath:
    """A selected path ready for value backpropagation."""

    tree_index: int
    path: List[PUCTNode]
    value: float


@dataclass
class PUCTExpansionRequest:
    """A reserved child expansion waiting for batched engine stepping."""

    tree_index: int
    tree: PUCTSearchTree
    parent: PUCTNode
    parent_path: List[PUCTNode]
    action: int


class BatchedPUCTMCTS:
    """Run AlphaZero-style PUCT over a batch of independent roots."""

    def __init__(
        self,
        evaluator: PolicyValueEvaluator,
        config: Optional[PUCTMCTSConfig] = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config if config is not None else PUCTMCTSConfig()
        self.last_trees: List[PUCTSearchTree] = []
        self.last_reused_roots = 0
        self.last_fresh_roots = 0
        self.last_leaf_evaluations = 0
        self.last_terminal_evaluations = 0
        self.last_leaf_batch_sizes: List[int] = []
        self.last_expansion_batch_sizes: List[int] = []
        self.last_expanded_children = 0
        self.last_root_visits_before: List[int] = []
        self.last_root_visits_after: List[int] = []
        self.last_new_visits_added: List[int] = []
        self.last_depth_histogram: dict[int, int] = {}
        self.last_timing_seconds = self._empty_timing()
        self._noise_generator = torch.Generator(device="cpu")
        if self.config.seed is not None:
            self._noise_generator.manual_seed(self.config.seed)

    def search_batch(self, roots: Connect4x4x4Batch) -> BatchedSearchResult:
        return self.search_batch_with_trees(roots, root_trees=None)

    def search_batch_with_trees(
        self,
        roots: Connect4x4x4Batch,
        root_trees: Optional[Sequence[Optional[PUCTSearchTree]]] = None,
    ) -> BatchedSearchResult:
        self._reset_diagnostics()
        started_at = time.perf_counter()
        search_roots = self._prepare_roots(roots)
        trees = self._prepare_trees(search_roots, root_trees)
        self.last_trees = trees
        self._ensure_roots_expanded(trees)
        if self.config.add_root_noise:
            self._add_root_noise(trees)
        self.last_root_visits_before = [tree.root.visits for tree in trees]
        self._add_timing("prepare_trees", started_at)

        remaining = [
            0 if tree.root.is_terminal else self.config.simulations_per_root
            for tree in trees
        ]
        total_remaining = sum(remaining)

        while total_remaining > 0:
            expansion_requests: List[PUCTExpansionRequest] = []
            terminal_paths: List[PUCTSelectedPath] = []
            value_paths: List[PUCTSelectedPath] = []

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

                    if isinstance(selected, PUCTExpansionRequest):
                        expansion_requests.append(selected)
                    elif selected.path[-1].is_terminal:
                        terminal_paths.append(selected)
                    else:
                        value_paths.append(selected)

                    if len(expansion_requests) >= self.config.max_leaf_batch_size or total_remaining == 0:
                        break

                if not made_progress:
                    break

            for selected in terminal_paths:
                self._backpropagate_selected(selected)
                self.last_terminal_evaluations += 1
            for selected in value_paths:
                self._backpropagate_selected(selected)

            if expansion_requests:
                pending, terminal = self._expand_requests(expansion_requests)
                for selected in terminal:
                    self._backpropagate_selected(selected)
                    self.last_terminal_evaluations += 1
                if pending:
                    self._evaluate_and_backpropagate(pending)
            elif total_remaining > 0:
                raise RuntimeError("PUCT selection stalled with no expandable paths")

        build_started_at = time.perf_counter()
        self.last_root_visits_after = [tree.root.visits for tree in trees]
        self.last_new_visits_added = [
            after - before
            for before, after in zip(self.last_root_visits_before, self.last_root_visits_after)
        ]
        self.last_depth_histogram = self._depth_histogram(trees)
        result = self._build_result(trees, device=search_roots.device)
        self._add_timing("build_result", build_started_at)
        return result

    def advance_tree(self, tree: PUCTSearchTree, action: int) -> Optional[PUCTSearchTree]:
        return tree.reuse_child(int(action))

    def _prepare_roots(self, roots: Connect4x4x4Batch) -> Connect4x4x4Batch:
        if roots.device.type in ("cuda", "mps"):
            return roots.to("cpu")
        return roots.clone()

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
        root_trees: Optional[Sequence[Optional[PUCTSearchTree]]],
    ) -> List[PUCTSearchTree]:
        if root_trees is not None and len(root_trees) != roots.batch_size:
            raise ValueError(f"root_trees must have length {roots.batch_size}, got {len(root_trees)}")

        trees: List[PUCTSearchTree] = []
        reused = 0
        fresh = 0
        for index in range(roots.batch_size):
            root_state = self._slice_state(roots, index)
            candidate = root_trees[index] if root_trees is not None else None
            if candidate is not None and self._tree_matches_state(candidate, root_state):
                trees.append(candidate)
                reused += 1
            else:
                trees.append(PUCTSearchTree.from_state(root_state))
                fresh += 1
        self.last_reused_roots = reused
        self.last_fresh_roots = fresh
        return trees

    def _tree_matches_state(self, tree: PUCTSearchTree, state: Connect4x4x4Batch) -> bool:
        root = tree.root.state
        if root.device != state.device:
            return False
        return (
            torch.equal(root.board, state.board)
            and torch.equal(root.heights, state.heights)
            and torch.equal(root.done, state.done)
            and torch.equal(root.outcome, state.outcome)
        )

    def _ensure_roots_expanded(self, trees: List[PUCTSearchTree]) -> None:
        nodes = [tree.root for tree in trees if not tree.root.is_terminal and not tree.root.is_expanded]
        if nodes:
            self._evaluate_nodes(nodes)

    def _select_path_or_expansion(
        self,
        tree_index: int,
        tree: PUCTSearchTree,
    ) -> PUCTSelectedPath | PUCTExpansionRequest | None:
        node = tree.root
        path = [node]

        while True:
            if node.is_terminal:
                return PUCTSelectedPath(tree_index=tree_index, path=path, value=float(node.terminal_value))

            if node.depth >= self.config.max_selection_depth:
                return PUCTSelectedPath(tree_index=tree_index, path=path, value=float(node.evaluation_value))

            if not node.legal_actions:
                node.terminal_value = 0.0
                return PUCTSelectedPath(tree_index=tree_index, path=path, value=0.0)

            action = self._select_action(node)
            if action is None:
                return None

            child = node.children[action]
            if child is None:
                tree.reserve_child(node, action)
                return PUCTExpansionRequest(
                    tree_index=tree_index,
                    tree=tree,
                    parent=node,
                    parent_path=list(path),
                    action=action,
                )

            node = child
            path.append(node)

    def _select_action(self, node: PUCTNode) -> Optional[int]:
        best_action: Optional[int] = None
        best_score = -math.inf
        parent_visits = node.visits

        for action in node.selectable_actions():
            prior = float(node.child_priors[action].item())
            child = node.children[action]
            child_visits = 0 if child is None else child.visits
            q_value = 0.0 if child is None or child.visits == 0 else -child.mean_value
            exploration = self.config.c_puct * prior * math.sqrt(parent_visits + 1.0) / (1.0 + child_visits)
            score = q_value + exploration
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _expand_requests(
        self,
        requests: List[PUCTExpansionRequest],
    ) -> tuple[List[PUCTSelectedPath], List[PUCTSelectedPath]]:
        started_at = time.perf_counter()
        parent_states = self._make_parent_batch(requests)
        actions = torch.tensor([request.action for request in requests], dtype=torch.long, device=parent_states.device)
        result = parent_states.step(actions)
        if not bool(result.legal.all().item()):
            for request in requests:
                request.parent.pending_actions.discard(request.action)
            raise RuntimeError("PUCT batched expansion attempted an illegal action")

        pending: List[PUCTSelectedPath] = []
        terminal: List[PUCTSelectedPath] = []
        for index, request in enumerate(requests):
            child_state = self._slice_state(parent_states, index)
            terminal_value = self._terminal_value_at(result, index)
            child = request.tree.attach_child(
                parent=request.parent,
                action=request.action,
                child_state=child_state,
                terminal_value=terminal_value,
            )
            selected = PUCTSelectedPath(
                tree_index=request.tree_index,
                path=request.parent_path + [child],
                value=float(terminal_value) if terminal_value is not None else 0.0,
            )
            if terminal_value is None:
                pending.append(selected)
            else:
                terminal.append(selected)

        self.last_expanded_children += len(requests)
        self.last_expansion_batch_sizes.append(len(requests))
        self._add_timing("expand", started_at)
        return pending, terminal

    def _evaluate_and_backpropagate(self, pending: List[PUCTSelectedPath]) -> None:
        started_at = time.perf_counter()
        nodes = [selected.path[-1] for selected in pending]
        values = self._evaluate_nodes(nodes)
        self._add_timing("leaf_eval", started_at)
        self.last_leaf_evaluations += len(pending)
        self.last_leaf_batch_sizes.append(len(pending))

        for selected, value in zip(pending, values):
            self._backpropagate_selected(
                PUCTSelectedPath(tree_index=selected.tree_index, path=selected.path, value=float(value))
            )

    def _evaluate_nodes(self, nodes: List[PUCTNode]) -> List[float]:
        states = self._make_node_batch(nodes)
        evaluation = self.evaluator.evaluate_batch(states)
        priors = evaluation.priors.detach().cpu()
        values = evaluation.values.detach().cpu()
        for index, node in enumerate(nodes):
            node.child_priors = priors[index].to(dtype=torch.float32)
            node.evaluation_value = float(values[index].item())
            node.is_expanded = True
        return [float(value.item()) for value in values]

    def _backpropagate_selected(self, selected: PUCTSelectedPath) -> None:
        started_at = time.perf_counter()
        self._backpropagate(selected.path, selected.value)
        self._add_timing("backprop", started_at)

    def _backpropagate(self, path: List[PUCTNode], leaf_value: float) -> None:
        value = float(leaf_value)
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def _make_parent_batch(self, requests: List[PUCTExpansionRequest]) -> Connect4x4x4Batch:
        device = requests[0].parent.state.device
        states = Connect4x4x4Batch(len(requests), device=device)
        states.board = torch.cat([request.parent.state.board for request in requests], dim=0).clone()
        states.heights = torch.cat([request.parent.state.heights for request in requests], dim=0).clone()
        states.done = torch.cat([request.parent.state.done for request in requests], dim=0).clone()
        states.outcome = torch.cat([request.parent.state.outcome for request in requests], dim=0).clone()
        return states

    def _make_node_batch(self, nodes: List[PUCTNode]) -> Connect4x4x4Batch:
        device = nodes[0].state.device
        states = Connect4x4x4Batch(len(nodes), device=device)
        states.board = torch.cat([node.state.board for node in nodes], dim=0).clone()
        states.heights = torch.cat([node.state.heights for node in nodes], dim=0).clone()
        states.done = torch.cat([node.state.done for node in nodes], dim=0).clone()
        states.outcome = torch.cat([node.state.outcome for node in nodes], dim=0).clone()
        return states

    def _terminal_value_at(self, result, index: int) -> Optional[float]:
        if bool(result.won[index].item()):
            return -1.0
        if bool(result.draw[index].item()):
            return 0.0
        return None

    def _build_result(self, trees: List[PUCTSearchTree], device: torch.device) -> BatchedSearchResult:
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

        return BatchedSearchResult(
            visit_counts=visit_counts,
            policy=self._policy_from_visits(visit_counts, trees),
            q_values=q_values,
            root_values=root_values,
        )

    def _policy_from_visits(self, visit_counts: torch.Tensor, trees: List[PUCTSearchTree]) -> torch.Tensor:
        legal_mask = torch.zeros_like(visit_counts, dtype=torch.bool)
        for root_index, tree in enumerate(trees):
            if tree.root.legal_actions:
                legal_mask[root_index, list(tree.root.legal_actions)] = True
        weights = visit_counts.pow(1.0 / self.config.policy_temperature)
        weights = weights.masked_fill(~legal_mask, 0.0)
        totals = weights.sum(dim=1, keepdim=True)
        return torch.where(totals > 0, weights / totals.clamp_min(1.0), torch.zeros_like(weights))

    def _add_root_noise(self, trees: List[PUCTSearchTree]) -> None:
        for tree in trees:
            root = tree.root
            if root.is_terminal or not root.legal_actions:
                continue
            legal = list(root.legal_actions)
            concentration = torch.full((len(legal),), self.config.root_dirichlet_alpha, dtype=torch.float32)
            noise = torch._standard_gamma(concentration, generator=self._noise_generator)
            noise = noise / noise.sum().clamp_min(1e-12)
            current = root.child_priors[legal]
            mixed = (1.0 - self.config.root_exploration_fraction) * current
            mixed += self.config.root_exploration_fraction * noise
            root.child_priors[legal] = mixed / mixed.sum().clamp_min(1e-12)

    def _depth_histogram(self, trees: List[PUCTSearchTree]) -> dict[int, int]:
        counts: Counter[int] = Counter()
        for tree in trees:
            self._count_depths(tree.root, counts)
        return dict(sorted(counts.items()))

    def _count_depths(self, node: PUCTNode, counts: Counter[int]) -> None:
        counts[node.depth] += 1
        for child in node.children:
            if child is not None:
                self._count_depths(child, counts)

    def _reset_diagnostics(self) -> None:
        self.last_trees = []
        self.last_reused_roots = 0
        self.last_fresh_roots = 0
        self.last_leaf_evaluations = 0
        self.last_terminal_evaluations = 0
        self.last_leaf_batch_sizes = []
        self.last_expansion_batch_sizes = []
        self.last_expanded_children = 0
        self.last_root_visits_before = []
        self.last_root_visits_after = []
        self.last_new_visits_added = []
        self.last_depth_histogram = {}
        self.last_timing_seconds = self._empty_timing()

    def _empty_timing(self) -> dict[str, float]:
        return {
            "prepare_trees": 0.0,
            "select": 0.0,
            "expand": 0.0,
            "leaf_eval": 0.0,
            "backprop": 0.0,
            "build_result": 0.0,
        }

    def _add_timing(self, key: str, started_at: float) -> None:
        self.last_timing_seconds[key] += time.perf_counter() - started_at
