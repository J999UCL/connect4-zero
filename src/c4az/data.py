from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from c4az.game import ACTION_SIZE, Position, encode_positions, transform_action_values

SCHEMA_VERSION = "c4az.selfplay.v1"


@dataclass(frozen=True, slots=True)
class SelfPlaySample:
    current_bits: int
    opponent_bits: int
    policy: np.ndarray
    value: float
    visit_counts: np.ndarray
    legal_mask: int
    action: int
    ply: int
    game_id: int

    @classmethod
    def from_position(
        cls,
        position: Position,
        *,
        policy: np.ndarray,
        value: float,
        visit_counts: np.ndarray,
        action: int,
        game_id: int,
    ) -> "SelfPlaySample":
        return cls(
            current_bits=position.current,
            opponent_bits=position.opponent,
            policy=policy.astype(np.float32),
            value=float(value),
            visit_counts=visit_counts.astype(np.uint32),
            legal_mask=position.legal_mask(),
            action=int(action),
            ply=int(position.ply),
            game_id=int(game_id),
        )


def write_dataset(
    output_dir: Path,
    samples: list[SelfPlaySample],
    *,
    metadata: dict,
    shard_size: int = 4096,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    shard_records = []
    for shard_index, start in enumerate(range(0, len(samples), shard_size)):
        shard_samples = samples[start : start + shard_size]
        shard_path = shard_dir / f"shard-{shard_index:06d}.npz"
        _write_npz_shard(shard_path, shard_samples)
        shard_records.append({"path": str(shard_path.relative_to(output_dir)), "samples": len(shard_samples)})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(samples),
        "num_games": len({sample.game_id for sample in samples}),
        "shards": shard_records,
        **metadata,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _write_npz_shard(path: Path, samples: list[SelfPlaySample]) -> None:
    np.savez_compressed(
        path,
        current_bits=np.array([s.current_bits for s in samples], dtype=np.uint64),
        opponent_bits=np.array([s.opponent_bits for s in samples], dtype=np.uint64),
        policies=np.stack([s.policy for s in samples]).astype(np.float32),
        values=np.array([s.value for s in samples], dtype=np.float32),
        visit_counts=np.stack([s.visit_counts for s in samples]).astype(np.uint32),
        legal_masks=np.array([s.legal_mask for s in samples], dtype=np.uint16),
        actions=np.array([s.action for s in samples], dtype=np.uint8),
        plies=np.array([s.ply for s in samples], dtype=np.uint8),
        game_ids=np.array([s.game_id for s in samples], dtype=np.uint64),
    )


class SelfPlayDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, manifest_path: Path, *, augment_symmetries: bool = False) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema: {self.manifest['schema_version']}")
        self.augment_symmetries = augment_symmetries
        self.shards = [_LoadedShard.from_path(self.root / record["path"]) for record in self.manifest["shards"]]
        self.starts = np.cumsum([0] + [len(shard) for shard in self.shards[:-1]]).tolist()
        self.base_len = sum(len(shard) for shard in self.shards)

    def __len__(self) -> int:
        return self.base_len * (8 if self.augment_symmetries else 1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        symmetry = 0
        base_index = index
        if self.augment_symmetries:
            base_index = index // 8
            symmetry = index % 8
        shard, local_index = self._locate(base_index)
        sample = shard.sample(local_index)
        if symmetry:
            sample = _transform_sample(sample, symmetry)
        position = Position(
            current=sample.current_bits,
            opponent=sample.opponent_bits,
            heights=_heights_from_bits(sample.current_bits | sample.opponent_bits),
            ply=sample.ply,
        )
        return {
            "input": encode_positions([position])[0],
            "policy": torch.from_numpy(sample.policy.astype(np.float32)),
            "value": torch.tensor(sample.value, dtype=torch.float32),
            "visit_counts": torch.from_numpy(sample.visit_counts.astype(np.int64)),
            "legal_mask": torch.tensor(sample.legal_mask, dtype=torch.int64),
            "action": torch.tensor(sample.action, dtype=torch.int64),
            "ply": torch.tensor(sample.ply, dtype=torch.int64),
        }

    def _locate(self, index: int) -> tuple["_LoadedShard", int]:
        if index < 0 or index >= self.base_len:
            raise IndexError(index)
        for shard_index, start in enumerate(self.starts):
            shard = self.shards[shard_index]
            if start <= index < start + len(shard):
                return shard, index - start
        raise IndexError(index)


@dataclass(slots=True)
class _LoadedShard:
    current_bits: np.ndarray
    opponent_bits: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    visit_counts: np.ndarray
    legal_masks: np.ndarray
    actions: np.ndarray
    plies: np.ndarray
    game_ids: np.ndarray

    @classmethod
    def from_path(cls, path: Path) -> "_LoadedShard":
        with np.load(path) as data:
            return cls(**{name: data[name].copy() for name in data.files})

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def sample(self, index: int) -> SelfPlaySample:
        return SelfPlaySample(
            current_bits=int(self.current_bits[index]),
            opponent_bits=int(self.opponent_bits[index]),
            policy=self.policies[index].copy(),
            value=float(self.values[index]),
            visit_counts=self.visit_counts[index].copy(),
            legal_mask=int(self.legal_masks[index]),
            action=int(self.actions[index]),
            ply=int(self.plies[index]),
            game_id=int(self.game_ids[index]),
        )


def _transform_sample(sample: SelfPlaySample, symmetry: int) -> SelfPlaySample:
    position = Position(
        current=sample.current_bits,
        opponent=sample.opponent_bits,
        heights=_heights_from_bits(sample.current_bits | sample.opponent_bits),
        ply=sample.ply,
    ).transform(symmetry)
    policy = transform_action_values(sample.policy, symmetry)
    visits = transform_action_values(sample.visit_counts, symmetry)
    action = int(np.argmax(transform_action_values(_one_hot(sample.action), symmetry)))
    return replace(
        sample,
        current_bits=position.current,
        opponent_bits=position.opponent,
        policy=policy.astype(np.float32),
        visit_counts=visits.astype(np.uint32),
        legal_mask=position.legal_mask(),
        action=action,
    )


def _one_hot(action: int) -> np.ndarray:
    values = np.zeros(ACTION_SIZE, dtype=np.float32)
    values[action] = 1.0
    return values


def _heights_from_bits(occupied: int) -> tuple[int, ...]:
    heights = []
    for action in range(ACTION_SIZE):
        height = 0
        for z in range(4):
            if occupied & (1 << (z * ACTION_SIZE + action)):
                height = z + 1
        heights.append(height)
    return tuple(heights)
