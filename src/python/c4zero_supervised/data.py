from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Iterator

import numpy as np
import torch

from c4zero_tools.datasets import validate_manifest

MAGIC = b"C4AZSP01"
HEADER = struct.Struct("<8sIIQ")
SAMPLE_DTYPE = np.dtype(
    [
        ("current_bits", "<u8"),
        ("opponent_bits", "<u8"),
        ("heights", "u1", (16,)),
        ("ply", "u1"),
        ("game_id", "<u8"),
        ("legal_mask", "<u2"),
        ("action", "u1"),
        ("policy", "<f4", (16,)),
        ("visit_counts", "<u4", (16,)),
        ("value", "<f4"),
    ]
)

STAGE0_CATEGORY_ORDER = (
    "immediate_win",
    "immediate_block",
    "safe_move_vs_blunder",
    "playable_vs_floating_threat",
    "fork_create",
    "fork_block",
    "minimax_depth3_policy",
)


@dataclass(frozen=True, slots=True)
class SupervisedBatch:
    inputs: torch.Tensor
    target_policy: torch.Tensor
    target_value: torch.Tensor
    legal_mask: torch.Tensor
    category_id: torch.Tensor
    action: torch.Tensor


class Stage0Dataset:
    def __init__(
        self,
        manifest_path: Path,
        current_bits: np.ndarray,
        opponent_bits: np.ndarray,
        policy: np.ndarray,
        value: np.ndarray,
        legal_mask: np.ndarray,
        action: np.ndarray,
        ply: np.ndarray,
        game_id: np.ndarray,
        category_id: np.ndarray,
        category_names: tuple[str, ...],
    ) -> None:
        self.manifest_path = manifest_path
        self.current_bits = current_bits
        self.opponent_bits = opponent_bits
        self.policy = policy
        self.value = value
        self.legal_mask = legal_mask
        self.action = action
        self.ply = ply
        self.game_id = game_id
        self.category_id = category_id
        self.category_names = category_names

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "Stage0Dataset":
        manifest_path = Path(manifest_path)
        manifest = validate_manifest(manifest_path)
        if manifest.get("dataset_kind") != "curriculum_stage0":
            raise ValueError(f"expected curriculum_stage0 manifest, got {manifest.get('dataset_kind')!r}")
        root = manifest_path.parent
        shards = []
        for shard in manifest["shard_paths"]:
            shard_path = Path(shard)
            if not shard_path.is_absolute():
                shard_path = root / shard_path
            shards.append(_read_shard_array(shard_path))
        records = np.concatenate(shards) if len(shards) > 1 else shards[0]
        category_counts = manifest.get("config", {}).get("category_counts", {})
        category_id = _stage0_category_ids(records["game_id"], category_counts)
        return cls(
            manifest_path=manifest_path,
            current_bits=records["current_bits"].copy(),
            opponent_bits=records["opponent_bits"].copy(),
            policy=records["policy"].copy(),
            value=records["value"].copy(),
            legal_mask=records["legal_mask"].copy(),
            action=records["action"].copy(),
            ply=records["ply"].copy(),
            game_id=records["game_id"].copy(),
            category_id=category_id,
            category_names=STAGE0_CATEGORY_ORDER,
        )

    def __len__(self) -> int:
        return int(self.policy.shape[0])

    def iter_epoch_indices(
        self,
        batch_size: int,
        rng: np.random.Generator,
        shuffle: bool = True,
    ) -> Iterator[np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = np.arange(len(self), dtype=np.int64)
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield indices[start : start + batch_size]

    def batch(self, indices: np.ndarray, device: torch.device | str = "cpu") -> SupervisedBatch:
        inputs = encode_bitboards(self.current_bits[indices], self.opponent_bits[indices], device=device)
        return SupervisedBatch(
            inputs=inputs,
            target_policy=torch.as_tensor(self.policy[indices], dtype=torch.float32, device=device),
            target_value=torch.as_tensor(self.value[indices], dtype=torch.float32, device=device),
            legal_mask=torch.as_tensor(self.legal_mask[indices].astype(np.int64), dtype=torch.int64, device=device),
            category_id=torch.as_tensor(self.category_id[indices].astype(np.int64), dtype=torch.int64, device=device),
            action=torch.as_tensor(self.action[indices].astype(np.int64), dtype=torch.int64, device=device),
        )

    def describe(self) -> dict[str, object]:
        return {
            "manifest": str(self.manifest_path),
            "num_samples": len(self),
            "num_categories": len(self.category_names),
            "category_counts": {
                name: int((self.category_id == index).sum())
                for index, name in enumerate(self.category_names)
            },
            "mean_ply": float(self.ply.mean()),
            "mean_policy_support": float((self.policy > 0.0).sum(axis=1).mean()),
        }


def encode_bitboards(
    current_bits: np.ndarray,
    opponent_bits: np.ndarray,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    bit_masks = np.uint64(1) << np.arange(64, dtype=np.uint64)
    current = ((current_bits.astype(np.uint64)[:, None] & bit_masks[None, :]) != 0).astype(np.float32)
    opponent = ((opponent_bits.astype(np.uint64)[:, None] & bit_masks[None, :]) != 0).astype(np.float32)
    planes = np.stack(
        [current.reshape(-1, 4, 4, 4), opponent.reshape(-1, 4, 4, 4)],
        axis=1,
    )
    return torch.as_tensor(planes, dtype=torch.float32, device=device)


def _read_shard_array(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    magic, major, _minor, count = HEADER.unpack_from(payload, 0)
    if magic != MAGIC:
        raise ValueError(f"{path} is not a c4zero shard")
    if major != 1:
        raise ValueError(f"unsupported c4zero shard major schema: {major}")
    expected_size = HEADER.size + count * SAMPLE_DTYPE.itemsize
    if len(payload) != expected_size:
        raise ValueError(f"{path} has size {len(payload)}, expected {expected_size}")
    return np.frombuffer(payload, dtype=SAMPLE_DTYPE, count=count, offset=HEADER.size)


def _stage0_category_ids(game_ids: np.ndarray, category_counts: dict[str, int]) -> np.ndarray:
    boundaries = []
    total = 0
    for index, name in enumerate(STAGE0_CATEGORY_ORDER):
        count = int(category_counts.get(name, 0))
        boundaries.append((total, total + count, index, name))
        total += count
    category_id = np.full(game_ids.shape, -1, dtype=np.int16)
    for start, end, index, _name in boundaries:
        mask = (game_ids >= start) & (game_ids < end)
        category_id[mask] = index
    if np.any(category_id < 0):
        bad = game_ids[category_id < 0][:10].tolist()
        raise ValueError(f"could not infer Stage 0 category for game ids: {bad}")
    return category_id


def write_dataset_summary(path: str | Path, datasets: dict[str, Stage0Dataset]) -> None:
    path = Path(path)
    path.write_text(
        json.dumps({name: dataset.describe() for name, dataset in datasets.items()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
