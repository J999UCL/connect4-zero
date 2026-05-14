"""Neural policy/value evaluator used by PUCT search."""

from __future__ import annotations

from typing import Optional

import torch

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE
from connect4_zero.model import Connect4ResNet3D, encode_boards
from connect4_zero.search.tree import terminal_value_for_state
from connect4_zero.search.types import PolicyValueBatch


class NeuralPolicyValueEvaluator:
    """Batch model inference with legal-action masking and terminal bypass."""

    def __init__(
        self,
        model: Connect4ResNet3D,
        device: str | torch.device = "cpu",
        inference_batch_size: int = 4096,
    ) -> None:
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        self.model = model.to(device)
        self.device = torch.device(device)
        self.inference_batch_size = int(inference_batch_size)

    def evaluate_batch(self, states: Connect4x4x4Batch) -> PolicyValueBatch:
        """Return legal priors and values from the current-player perspective."""
        priors = torch.zeros((states.batch_size, ACTION_SIZE), dtype=torch.float32, device=states.device)
        values = torch.zeros(states.batch_size, dtype=torch.float32, device=states.device)

        nonterminal_indices: list[int] = []
        for index in range(states.batch_size):
            state = _slice_state(states, index)
            terminal_value = terminal_value_for_state(state)
            if terminal_value is None:
                nonterminal_indices.append(index)
            else:
                values[index] = float(terminal_value)

        if not nonterminal_indices:
            return PolicyValueBatch(priors=priors, values=values)

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(nonterminal_indices), self.inference_batch_size):
                selected = nonterminal_indices[start : start + self.inference_batch_size]
                index_tensor = torch.tensor(selected, dtype=torch.long, device=states.device)
                boards = states.board[index_tensor]
                legal_mask = states.legal_mask()[index_tensor]
                encoded = encode_boards(boards).to(device=self.device)
                logits, model_values = self.model(encoded)
                masked_logits = logits.to(device=states.device).masked_fill(~legal_mask, -1e9)
                chunk_priors = torch.softmax(masked_logits, dim=1)
                chunk_priors = chunk_priors.masked_fill(~legal_mask, 0.0)
                totals = chunk_priors.sum(dim=1, keepdim=True)
                fallback = legal_mask.to(dtype=torch.float32)
                fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
                chunk_priors = torch.where(totals > 0, chunk_priors / totals.clamp_min(1e-12), fallback)

                priors[index_tensor] = chunk_priors
                values[index_tensor] = model_values.to(device=states.device, dtype=torch.float32)

        return PolicyValueBatch(priors=priors, values=values)


def _slice_state(states: Connect4x4x4Batch, index: int) -> Connect4x4x4Batch:
    state = Connect4x4x4Batch(1, device=states.device)
    state.board = states.board[index : index + 1].clone()
    state.heights = states.heights[index : index + 1].clone()
    state.done = states.done[index : index + 1].clone()
    state.outcome = states.outcome[index : index + 1].clone()
    return state
