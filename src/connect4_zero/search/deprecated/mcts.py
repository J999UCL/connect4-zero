"""Single-root classical MCTS for 4x4x4 Connect Four."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from connect4_zero.game import Connect4x4x4Batch, StepResult
from connect4_zero.game.constants import ACTION_SIZE
from connect4_zero.search.deprecated.nodes import NodeStore, SearchNode, TreeNodeStore
from connect4_zero.search.rollout import RandomRolloutEvaluator
from connect4_zero.search.types import Evaluator, MCTSConfig, SearchResult


class MCTS:
    """Classical UCB1 MCTS over one canonical game state."""

    def __init__(
        self,
        config: Optional[MCTSConfig] = None,
        evaluator: Optional[Evaluator] = None,
        node_store: Optional[NodeStore] = None,
    ) -> None:
        self.config = config if config is not None else MCTSConfig()
        self.evaluator = evaluator if evaluator is not None else RandomRolloutEvaluator(
            rollout_batch_size=self.config.rollout_batch_size,
            device=self.config.rollout_device,
            seed=self.config.seed,
            max_steps=self.config.max_rollout_steps,
        )
        self.node_store = node_store if node_store is not None else TreeNodeStore()

    def search(self, root: Connect4x4x4Batch) -> SearchResult:
        """Run MCTS from ``root`` without mutating the caller's state."""
        self._validate_root(root)
        root_node = self.node_store.create_root(root.clone())

        for _ in range(self.config.num_simulations):
            path = self._select_and_expand(root_node)
            leaf = path[-1]
            value = leaf.terminal_value if leaf.terminal_value is not None else self.evaluator.evaluate(leaf.state)
            self._backpropagate(path, value)

        return self._build_result(root_node)

    def _validate_root(self, root: Connect4x4x4Batch) -> None:
        if root.batch_size != 1:
            raise ValueError(f"MCTS.search requires batch_size=1, got {root.batch_size}")
        if bool(root.done[0].item()):
            raise ValueError("MCTS.search requires a non-terminal root state")

    def _select_and_expand(self, root: SearchNode) -> List[SearchNode]:
        node = root
        path = [node]

        while not node.is_terminal:
            unexpanded_actions = self._unexpanded_actions(node)
            if unexpanded_actions:
                child = self._expand_child(node, unexpanded_actions[0])
                path.append(child)
                return path

            _, node = self._select_child(node)
            path.append(node)

        return path

    def _unexpanded_actions(self, node: SearchNode) -> List[int]:
        return [action for action in node.legal_actions if action not in node.children]

    def _expand_child(self, parent: SearchNode, action: int) -> SearchNode:
        child_state = parent.state.clone()
        result = child_state.step(torch.tensor([action], dtype=torch.long, device=child_state.device))
        if not bool(result.legal[0].item()):
            raise RuntimeError(f"attempted to expand illegal action {action}")

        terminal_value = self._terminal_value_for_child(result)
        return self.node_store.get_or_create_child(
            parent=parent,
            action=action,
            state=child_state,
            terminal_value=terminal_value,
        )

    def _terminal_value_for_child(self, result: StepResult) -> Optional[float]:
        if bool(result.won[0].item()):
            return -1.0
        if bool(result.draw[0].item()):
            return 0.0
        return None

    def _select_child(self, node: SearchNode) -> Tuple[int, SearchNode]:
        if not node.children:
            raise RuntimeError("cannot select from a node with no children")

        best_action = -1
        best_child: Optional[SearchNode] = None
        best_score = -math.inf

        for action in sorted(node.children):
            child = node.children[action]
            score = self._ucb_score(parent=node, child=child)
            if score > best_score:
                best_action = action
                best_child = child
                best_score = score

        if best_child is None:
            raise RuntimeError("failed to select a child")
        return best_action, best_child

    def _ucb_score(self, parent: SearchNode, child: SearchNode) -> float:
        if child.visits == 0:
            return math.inf

        exploitation = -child.mean_value
        exploration = self.config.exploration_constant * math.sqrt(
            math.log(parent.visits + 1) / child.visits
        )
        return exploitation + exploration

    def _backpropagate(self, path: List[SearchNode], leaf_value: float) -> None:
        value = float(leaf_value)
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def _build_result(self, root: SearchNode) -> SearchResult:
        device = root.state.device
        visit_counts = torch.zeros(ACTION_SIZE, dtype=torch.float32, device=device)
        q_values = torch.zeros(ACTION_SIZE, dtype=torch.float32, device=device)

        for action, child in root.children.items():
            visit_counts[action] = float(child.visits)
            if child.visits > 0:
                q_values[action] = float(-child.mean_value)

        total_visits = visit_counts.sum()
        policy = torch.zeros(ACTION_SIZE, dtype=torch.float32, device=device)
        if float(total_visits.item()) > 0:
            policy = visit_counts / total_visits

        return SearchResult(
            visit_counts=visit_counts,
            policy=policy,
            q_values=q_values,
            root_value=root.mean_value,
        )
