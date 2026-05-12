"""Batched root/action search for high-throughput data generation."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE
from connect4_zero.search.rollout import BatchedRandomRolloutEvaluator
from connect4_zero.search.types import BatchedRootActionConfig, BatchedSearchResult


class BatchedRootActionMCTS:
    """Root-parallel, action-parallel MCTS-style evaluator.

    This is intentionally shallower than the reference object-tree ``MCTS``.
    It expands/evaluates root actions in large tensor batches so we can produce
    AlphaZero seed data quickly.
    """

    def __init__(
        self,
        config: Optional[BatchedRootActionConfig] = None,
        evaluator: Optional[BatchedRandomRolloutEvaluator] = None,
    ) -> None:
        self.config = config if config is not None else BatchedRootActionConfig()
        self.evaluator = evaluator if evaluator is not None else BatchedRandomRolloutEvaluator(
            rollouts_per_state=self.config.rollouts_per_leaf,
            device=self.config.rollout_device,
            seed=self.config.seed,
            max_steps=self.config.max_rollout_steps,
            max_rollouts_per_chunk=self.config.max_rollouts_per_chunk,
        )

    def search_batch(self, roots: Connect4x4x4Batch) -> BatchedSearchResult:
        """Search a batch of roots without mutating the caller's states."""
        roots = self._prepare_roots(roots)
        legal_mask = roots.legal_mask()
        visit_counts = torch.zeros((roots.batch_size, ACTION_SIZE), dtype=torch.float32, device=roots.device)
        value_sums = torch.zeros_like(visit_counts)
        solved_wins = torch.zeros_like(visit_counts, dtype=torch.bool)

        if self.config.evaluate_all_actions_first:
            root_indices, actions = legal_mask.nonzero(as_tuple=True)
            self._evaluate_and_accumulate(
                roots,
                root_indices,
                actions,
                visit_counts,
                value_sums,
                solved_wins,
            )

        for _ in range(self.config.num_selection_waves):
            root_indices, actions = self._select_wave(legal_mask, visit_counts, value_sums, solved_wins)
            if root_indices.numel() == 0:
                break
            self._evaluate_and_accumulate(
                roots,
                root_indices,
                actions,
                visit_counts,
                value_sums,
                solved_wins,
            )

        q_values = self._mean_values(visit_counts, value_sums)
        policy = self._policy_from_visits(visit_counts, legal_mask)
        root_values = (policy * q_values).sum(dim=1)
        root_values[roots.done & roots.outcome.eq(1)] = -1.0
        root_values[roots.done & roots.outcome.eq(0)] = 0.0

        return BatchedSearchResult(
            visit_counts=visit_counts,
            policy=policy,
            q_values=q_values,
            root_values=root_values,
        )

    def _prepare_roots(self, roots: Connect4x4x4Batch) -> Connect4x4x4Batch:
        if self.config.rollout_device is None:
            return roots.clone()
        return roots.to(self.config.rollout_device)

    def _select_wave(
        self,
        legal_mask: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor,
        solved_wins: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self._ucb_scores(legal_mask, visit_counts, value_sums, solved_wins)
        selected_scores, selected_actions = torch.topk(scores, k=self.config.leaves_per_root, dim=1)
        selected_roots = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
        selected_roots = selected_roots.expand_as(selected_actions)

        valid = torch.isfinite(selected_scores).reshape(-1)
        return selected_roots.reshape(-1)[valid], selected_actions.reshape(-1)[valid]

    def _ucb_scores(
        self,
        legal_mask: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor,
        solved_wins: torch.Tensor,
    ) -> torch.Tensor:
        root_has_solved_win = solved_wins.any(dim=1, keepdim=True)
        selection_mask = torch.where(root_has_solved_win, solved_wins, legal_mask)
        q_values = self._mean_values(visit_counts, value_sums)
        total_visits = visit_counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        exploration = self.config.exploration_constant * torch.sqrt(
            torch.log(total_visits + 1.0) / visit_counts.clamp_min(1.0)
        )
        scores = q_values + exploration
        return scores.masked_fill(~selection_mask, -torch.inf)

    def _evaluate_and_accumulate(
        self,
        roots: Connect4x4x4Batch,
        root_indices: torch.Tensor,
        actions: torch.Tensor,
        visit_counts: torch.Tensor,
        value_sums: torch.Tensor,
        solved_wins: torch.Tensor,
    ) -> None:
        if root_indices.numel() == 0:
            return

        q_estimates, terminal_wins = self._evaluate_candidates(roots, root_indices, actions)
        flat_indices = root_indices * ACTION_SIZE + actions
        flat_visits = visit_counts.reshape(-1)
        flat_values = value_sums.reshape(-1)
        flat_visits.scatter_add_(0, flat_indices, torch.ones_like(q_estimates))
        flat_values.scatter_add_(0, flat_indices, q_estimates)
        if bool(terminal_wins.any().item()):
            solved_wins[root_indices[terminal_wins], actions[terminal_wins]] = True

    def _evaluate_candidates(
        self,
        roots: Connect4x4x4Batch,
        root_indices: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        child_states = self._make_child_batch(roots, root_indices)
        result = child_states.step(actions.to(device=child_states.device, dtype=torch.long))
        if not bool(result.legal.all().item()):
            raise RuntimeError("batched search attempted to evaluate an illegal action")

        q_estimates = torch.zeros(actions.shape[0], dtype=torch.float32, device=child_states.device)
        q_estimates[result.won] = 1.0
        q_estimates[result.draw] = 0.0

        nonterminal = result.legal & ~result.done
        if bool(nonterminal.any().item()):
            nonterminal_states = self._index_batch(child_states, nonterminal)
            child_values = self.evaluator.evaluate_batch(nonterminal_states).to(device=child_states.device)
            q_estimates[nonterminal] = -child_values

        return q_estimates, result.won

    def _make_child_batch(
        self,
        roots: Connect4x4x4Batch,
        root_indices: torch.Tensor,
    ) -> Connect4x4x4Batch:
        child_states = Connect4x4x4Batch(root_indices.numel(), device=roots.device)
        child_states.board = roots.board[root_indices].clone()
        child_states.heights = roots.heights[root_indices].clone()
        child_states.done = roots.done[root_indices].clone()
        child_states.outcome = roots.outcome[root_indices].clone()
        return child_states

    def _index_batch(
        self,
        batch: Connect4x4x4Batch,
        mask: torch.Tensor,
    ) -> Connect4x4x4Batch:
        selected = Connect4x4x4Batch(int(mask.sum().item()), device=batch.device)
        selected.board = batch.board[mask].clone()
        selected.heights = batch.heights[mask].clone()
        selected.done = batch.done[mask].clone()
        selected.outcome = batch.outcome[mask].clone()
        return selected

    def _mean_values(self, visit_counts: torch.Tensor, value_sums: torch.Tensor) -> torch.Tensor:
        return torch.where(
            visit_counts > 0,
            value_sums / visit_counts.clamp_min(1.0),
            torch.zeros_like(value_sums),
        )

    def _policy_from_visits(self, visit_counts: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        weights = visit_counts.pow(1.0 / self.config.policy_temperature)
        weights = weights.masked_fill(~legal_mask, 0.0)
        totals = weights.sum(dim=1, keepdim=True)
        return torch.where(totals > 0, weights / totals.clamp_min(1.0), torch.zeros_like(weights))
