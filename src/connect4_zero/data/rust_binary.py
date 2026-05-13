"""Loader for Rust-generated custom binary self-play shards."""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import zstandard as zstd
from torch.utils.data import Dataset

from connect4_zero.game.constants import ACTION_SIZE, BOARD_CELLS, BOARD_SIZE
from connect4_zero.game.symmetries import make_symmetry_permutations

MAGIC = b"C4ZRS001"
FORMAT = "connect4_zero.rust.selfplay.v1"


@dataclass(frozen=True)
class RustShard:
    """Decoded tensors from one Rust binary shard."""

    boards: torch.Tensor
    policies: torch.Tensor
    values: torch.Tensor
    visit_counts: torch.Tensor
    q_values: torch.Tensor
    legal_masks: torch.Tensor
    actions: torch.Tensor
    plies: torch.Tensor


class RustBinarySelfPlayDataset(Dataset[Dict[str, torch.Tensor]]):
    """Load AlphaZero samples from Rust zstd-compressed binary shards."""

    def __init__(self, manifest_path: Path | str, apply_symmetries: bool = False) -> None:
        self.manifest_path = Path(manifest_path)
        self.root_dir = self.manifest_path.parent
        self.manifest = self._read_manifest()
        self.records = list(self.manifest["shards"])
        self.apply_symmetries = bool(apply_symmetries)
        self._permutations = make_symmetry_permutations(device="cpu")
        self._cumulative_counts = self._make_cumulative_counts()
        self._cached_shard_index = -1
        self._cached_shard: RustShard | None = None

    def __len__(self) -> int:
        multiplier = 8 if self.apply_symmetries else 1
        return self._cumulative_counts[-1] * multiplier

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        raw_index, symmetry_index = self._raw_index(index)
        shard_index, local_index = self._locate(raw_index)
        shard = self._load_shard(shard_index)

        board = shard.boards[local_index]
        policy = shard.policies[local_index].to(dtype=torch.float32)
        visit_counts = shard.visit_counts[local_index]
        q_values = shard.q_values[local_index].to(dtype=torch.float32)
        legal_mask = shard.legal_masks[local_index]
        action = shard.actions[local_index].to(dtype=torch.long)

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
            "value": shard.values[local_index].to(dtype=torch.float32),
            "visit_counts": visit_counts,
            "q_values": q_values,
            "legal_mask": legal_mask,
            "action": action,
            "ply": shard.plies[local_index],
        }

    def _read_manifest(self) -> Dict[str, Any]:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT:
            raise ValueError(f"unsupported Rust shard format: {manifest.get('format')}")
        if int(manifest.get("format_version", 0)) != 1:
            raise ValueError(f"unsupported Rust shard version: {manifest.get('format_version')}")
        return manifest

    def _make_cumulative_counts(self) -> List[int]:
        counts = [0]
        for record in self.records:
            counts.append(counts[-1] + int(record["samples"]))
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

    def _load_shard(self, shard_index: int) -> RustShard:
        if shard_index != self._cached_shard_index:
            path = self.root_dir / self.records[shard_index]["path"]
            self._cached_shard = read_rust_shard(path)
            self._cached_shard_index = shard_index
        assert self._cached_shard is not None
        return self._cached_shard

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
        columns = board.reshape(ACTION_SIZE, BOARD_SIZE)
        transformed_columns = torch.empty_like(columns)
        transformed_columns[permutation] = columns
        board = transformed_columns.reshape_as(board)

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

    def _board_to_planes(self, board: torch.Tensor) -> torch.Tensor:
        current = board.eq(1)
        opponent = board.eq(-1)
        return torch.stack((current, opponent), dim=0).to(dtype=torch.float32)


def read_rust_shard(path: Path | str) -> RustShard:
    with Path(path).open("rb") as source:
        with zstd.ZstdDecompressor().stream_reader(source) as reader:
            payload = reader.read()
    cursor = _Cursor(payload)
    magic = cursor.read_bytes(8)
    if magic != MAGIC:
        raise ValueError("invalid Rust shard magic")
    n = cursor.read_u64()

    current_bits = cursor.read_array("<u8", n)
    opponent_bits = cursor.read_array("<u8", n)
    heights = cursor.read_array("u1", n * ACTION_SIZE).reshape(n, ACTION_SIZE)
    policies = cursor.read_array("<f4", n * ACTION_SIZE).reshape(n, ACTION_SIZE)
    values = cursor.read_array("i1", n)
    visit_counts = cursor.read_array("<u4", n * ACTION_SIZE).reshape(n, ACTION_SIZE)
    q_values = cursor.read_array("<f4", n * ACTION_SIZE).reshape(n, ACTION_SIZE)
    legal_packed = cursor.read_array("<u2", n)
    actions = cursor.read_array("u1", n)
    plies = cursor.read_array("u1", n)

    boards = np.zeros((n, BOARD_CELLS), dtype=np.int8)
    for row in range(n):
        current = int(current_bits[row])
        opponent = int(opponent_bits[row])
        for bit in range(BOARD_CELLS):
            mask = 1 << bit
            if current & mask:
                boards[row, bit] = 1
            elif opponent & mask:
                boards[row, bit] = -1

    legal_masks = np.zeros((n, ACTION_SIZE), dtype=np.bool_)
    for row, packed in enumerate(legal_packed):
        packed_int = int(packed)
        for action in range(ACTION_SIZE):
            legal_masks[row, action] = bool(packed_int & (1 << action))

    # Validate enough shape data to catch format drift without storing heights in every sample.
    if heights.shape != (n, ACTION_SIZE):
        raise ValueError("invalid heights column in Rust shard")

    return RustShard(
        boards=torch.from_numpy(boards.reshape(n, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE).copy()),
        policies=torch.from_numpy(policies.copy()),
        values=torch.from_numpy(values.copy()),
        visit_counts=torch.from_numpy(visit_counts.astype(np.int32, copy=True)),
        q_values=torch.from_numpy(q_values.copy()),
        legal_masks=torch.from_numpy(legal_masks),
        actions=torch.from_numpy(actions.copy()),
        plies=torch.from_numpy(plies.copy()),
    )


class _Cursor:
    def __init__(self, payload: bytes) -> None:
        self.payload = memoryview(payload)
        self.offset = 0

    def read_bytes(self, size: int) -> bytes:
        if self.offset + size > len(self.payload):
            raise ValueError("unexpected end of Rust shard")
        start = self.offset
        self.offset += size
        return self.payload[start : self.offset].tobytes()

    def read_u64(self) -> int:
        return int(np.frombuffer(self.read_bytes(8), dtype="<u8", count=1)[0])

    def read_array(self, dtype: str, count: int) -> np.ndarray:
        dtype_obj = np.dtype(dtype)
        raw = self.read_bytes(dtype_obj.itemsize * count)
        return np.frombuffer(raw, dtype=dtype_obj, count=count)
