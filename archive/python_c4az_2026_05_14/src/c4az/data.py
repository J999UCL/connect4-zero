from __future__ import annotations

import json
from dataclasses import dataclass, replace
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
        return _sample_to_item(self.sample(index))

    def sample(self, index: int, *, symmetry: int = 0) -> SelfPlaySample:
        if symmetry < 0 or symmetry >= 8:
            raise ValueError(f"symmetry out of range: {symmetry}")
        if self.augment_symmetries:
            index, symmetry = divmod(index, 8)
        shard, local_index = self._locate(index)
        sample = shard.sample(local_index)
        if symmetry:
            sample = _transform_sample(sample, symmetry)
        return sample

    def game_ids(self) -> tuple[int, ...]:
        ids: set[int] = set()
        for shard in self.shards:
            ids.update(int(game_id) for game_id in shard.game_ids)
        return tuple(sorted(ids))

    def sample_refs_for_games(self, game_ids: set[int]) -> list[int]:
        refs: list[int] = []
        offset = 0
        for shard in self.shards:
            for local_index, game_id in enumerate(shard.game_ids):
                if int(game_id) in game_ids:
                    refs.append(offset + local_index)
            offset += len(shard)
        return refs

    def _locate(self, index: int) -> tuple["_LoadedShard", int]:
        if index < 0 or index >= self.base_len:
            raise IndexError(index)
        for shard_index, start in enumerate(self.starts):
            shard = self.shards[shard_index]
            if start <= index < start + len(shard):
                return shard, index - start
        raise IndexError(index)


class ReplayBuffer:
    """Uniform random minibatch sampler over the most recent self-play games."""

    def __init__(
        self,
        manifest_paths: Iterable[Path],
        *,
        replay_games: int | None = None,
        augment_symmetries: bool = True,
        seed: int | None = None,
    ) -> None:
        self.manifest_paths = tuple(Path(path) for path in manifest_paths)
        if not self.manifest_paths:
            raise ValueError("ReplayBuffer requires at least one manifest")
        if replay_games is not None and replay_games <= 0:
            raise ValueError("replay_games must be positive")

        self.datasets = tuple(SelfPlayDataset(path, augment_symmetries=False) for path in self.manifest_paths)
        self.augment_symmetries = augment_symmetries
        self.rng = np.random.default_rng(seed)

        selected_games = self._select_recent_games(replay_games)
        self.selected_games = selected_games
        self.sample_refs = self._build_sample_refs(selected_games)
        if not self.sample_refs:
            raise ValueError("ReplayBuffer contains no positions")

    @property
    def num_games(self) -> int:
        return len(self.selected_games)

    @property
    def num_positions(self) -> int:
        return len(self.sample_refs)

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        ref_indices = self.rng.integers(0, len(self.sample_refs), size=batch_size)
        symmetries = (
            self.rng.integers(0, 8, size=batch_size)
            if self.augment_symmetries
            else np.zeros(batch_size, dtype=np.int64)
        )
        samples: list[SelfPlaySample] = []
        for ref_index, symmetry in zip(ref_indices, symmetries):
            dataset_index, sample_index = self.sample_refs[int(ref_index)]
            samples.append(self.datasets[dataset_index].sample(sample_index, symmetry=int(symmetry)))
        return _collate_samples(samples)

    def stats(self) -> dict[str, object]:
        return {
            "manifests": [str(path) for path in self.manifest_paths],
            "games": self.num_games,
            "positions": self.num_positions,
            "augment_symmetries": self.augment_symmetries,
        }

    def _select_recent_games(self, replay_games: int | None) -> tuple[tuple[int, int], ...]:
        ordered_games = [
            (dataset_index, game_id)
            for dataset_index, dataset in enumerate(self.datasets)
            for game_id in dataset.game_ids()
        ]
        if not ordered_games:
            raise ValueError("no games found in replay manifests")
        if replay_games is None:
            return tuple(ordered_games)
        return tuple(ordered_games[-replay_games:])

    def _build_sample_refs(self, selected_games: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
        refs: list[tuple[int, int]] = []
        selected_by_dataset: dict[int, set[int]] = {}
        for dataset_index, game_id in selected_games:
            selected_by_dataset.setdefault(dataset_index, set()).add(game_id)
        for dataset_index, game_ids in selected_by_dataset.items():
            refs.extend((dataset_index, sample_index) for sample_index in self.datasets[dataset_index].sample_refs_for_games(game_ids))
        return refs


def _sample_to_position(sample: SelfPlaySample) -> Position:
    return Position(
        current=sample.current_bits,
        opponent=sample.opponent_bits,
        heights=_heights_from_bits(sample.current_bits | sample.opponent_bits),
        ply=sample.ply,
    )


def _sample_to_item(sample: SelfPlaySample) -> dict[str, torch.Tensor]:
    return {
        "input": encode_positions([_sample_to_position(sample)])[0],
        "policy": torch.from_numpy(sample.policy.astype(np.float32)),
        "value": torch.tensor(sample.value, dtype=torch.float32),
        "visit_counts": torch.from_numpy(sample.visit_counts.astype(np.int64)),
        "legal_mask": torch.tensor(sample.legal_mask, dtype=torch.int64),
        "action": torch.tensor(sample.action, dtype=torch.int64),
        "ply": torch.tensor(sample.ply, dtype=torch.int64),
    }


def _collate_samples(samples: list[SelfPlaySample]) -> dict[str, torch.Tensor]:
    positions = [_sample_to_position(sample) for sample in samples]
    return {
        "input": encode_positions(positions),
        "policy": torch.from_numpy(np.stack([sample.policy for sample in samples]).astype(np.float32)),
        "value": torch.tensor([sample.value for sample in samples], dtype=torch.float32),
        "visit_counts": torch.from_numpy(np.stack([sample.visit_counts for sample in samples]).astype(np.int64)),
        "legal_mask": torch.tensor([sample.legal_mask for sample in samples], dtype=torch.int64),
        "action": torch.tensor([sample.action for sample in samples], dtype=torch.int64),
        "ply": torch.tensor([sample.ply for sample in samples], dtype=torch.int64),
    }


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
