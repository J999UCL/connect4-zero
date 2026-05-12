"""Dataset utilities for safetensor self-play shards."""

from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from connect4_zero.game.constants import CURRENT_PLAYER, OPPONENT_PLAYER
from connect4_zero.game.symmetries import make_symmetry_permutations


class SelfPlayDataset(Dataset[Dict[str, torch.Tensor]]):
    """Load AlphaZero samples from a JSONL manifest of safetensor shards."""

    def __init__(
        self,
        manifest_path: Path | str,
        apply_symmetries: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root_dir = self.manifest_path.parent
        self.records = self._read_manifest()
        self.apply_symmetries = bool(apply_symmetries)
        self._permutations = make_symmetry_permutations(device="cpu")
        self._cumulative_counts = self._make_cumulative_counts()
        self._cached_shard_index = -1
        self._cached_tensors: Dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        multiplier = 8 if self.apply_symmetries else 1
        return self._cumulative_counts[-1] * multiplier

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        raw_index, symmetry_index = self._raw_index(index)
        shard_index, local_index = self._locate(raw_index)
        tensors = self._load_shard(shard_index)

        board = tensors["boards"][local_index]
        policy = tensors["policies"][local_index].to(dtype=torch.float32)
        visit_counts = tensors["visit_counts"][local_index]
        q_values = tensors["q_values"][local_index].to(dtype=torch.float32)
        legal_mask = tensors["legal_masks"][local_index]
        action = tensors["actions"][local_index].to(dtype=torch.long)

        if symmetry_index is not None:
            board, policy, visit_counts, q_values, legal_mask, action = self._apply_symmetry(
                board,
                policy,
                visit_counts,
                q_values,
                legal_mask,
                action,
                symmetry_index,
            )

        return {
            "input": self._board_to_planes(board),
            "board": board,
            "policy": policy,
            "value": tensors["values"][local_index].to(dtype=torch.float32),
            "visit_counts": visit_counts,
            "q_values": q_values,
            "legal_mask": legal_mask,
            "action": action,
            "ply": tensors["plies"][local_index],
        }

    def _read_manifest(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as manifest:
            for line in manifest:
                if line.strip():
                    records.append(json.loads(line))
        if not records:
            raise ValueError(f"manifest contains no shards: {self.manifest_path}")
        return records

    def _make_cumulative_counts(self) -> List[int]:
        counts = [0]
        for record in self.records:
            counts.append(counts[-1] + int(record["num_samples"]))
        return counts

    def _raw_index(self, index: int) -> Tuple[int, int | None]:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)

        if not self.apply_symmetries:
            return index, None
        return index // 8, index % 8

    def _locate(self, raw_index: int) -> Tuple[int, int]:
        shard_index = bisect_right(self._cumulative_counts, raw_index) - 1
        local_index = raw_index - self._cumulative_counts[shard_index]
        return shard_index, local_index

    def _load_shard(self, shard_index: int) -> Dict[str, torch.Tensor]:
        if shard_index != self._cached_shard_index:
            path = self.root_dir / self.records[shard_index]["shard"]
            self._cached_tensors = load_file(path)
            self._cached_shard_index = shard_index
        return self._cached_tensors

    def _apply_symmetry(
        self,
        board: torch.Tensor,
        policy: torch.Tensor,
        visit_counts: torch.Tensor,
        q_values: torch.Tensor,
        legal_mask: torch.Tensor,
        action: torch.Tensor,
        symmetry_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        permutation = self._permutations[symmetry_index].to(dtype=torch.long)
        board = self._permute_columns(board, permutation)

        transformed_policy = torch.empty_like(policy)
        transformed_policy[permutation] = policy
        transformed_visits = torch.empty_like(visit_counts)
        transformed_visits[permutation] = visit_counts
        transformed_q = torch.empty_like(q_values)
        transformed_q[permutation] = q_values
        transformed_legal = torch.empty_like(legal_mask)
        transformed_legal[permutation] = legal_mask
        transformed_action = permutation[action]
        return board, transformed_policy, transformed_visits, transformed_q, transformed_legal, transformed_action

    def _permute_columns(self, board: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
        columns = board.reshape(16, board.shape[-1])
        transformed = torch.empty_like(columns)
        transformed[permutation] = columns
        return transformed.reshape_as(board)

    def _board_to_planes(self, board: torch.Tensor) -> torch.Tensor:
        current = board.eq(CURRENT_PLAYER)
        opponent = board.eq(OPPONENT_PLAYER)
        return torch.stack((current, opponent), dim=0).to(dtype=torch.float32)
