from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import random

from c4zero_tools.datasets import Sample, read_shard, validate_manifest
from c4zero_train.symmetry import transform_sample


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    replay_games: int
    batch_size: int
    reuse_target: int = 4
    avg_plies: int = 32

    @classmethod
    def for_preset(cls, preset: str) -> "ReplayConfig":
        if preset == "tiny":
            return cls(replay_games=2048, batch_size=512)
        if preset == "small":
            return cls(replay_games=8192, batch_size=1024)
        if preset == "medium":
            return cls(replay_games=16384, batch_size=2048)
        raise ValueError(f"unknown replay preset: {preset}")

    def train_steps_for_generated_games(self, generated_games: int) -> int:
        numerator = self.reuse_target * generated_games * self.avg_plies
        return max(1, (numerator + self.batch_size - 1) // self.batch_size)


@dataclass(frozen=True, slots=True)
class ReplaySamplingConfig:
    mode: str = "uniform"
    recent_games: int = 4_000
    recent_fraction: float = 0.75

    def __post_init__(self) -> None:
        if self.mode not in {"uniform", "recent-mix"}:
            raise ValueError("replay sampling mode must be 'uniform' or 'recent-mix'")
        if self.recent_games <= 0:
            raise ValueError("recent_games must be positive")
        if not 0.0 <= self.recent_fraction <= 1.0:
            raise ValueError("recent_fraction must be in [0, 1]")


class ReplayBuffer:
    def __init__(self, samples: list[Sample], num_games: int | None = None, game_samples: list[list[Sample]] | None = None) -> None:
        self.samples = samples
        self._game_samples = game_samples if game_samples is not None else _group_samples_by_game_id(samples)
        self._num_games = num_games if num_games is not None else len(self._game_samples)
        if not samples:
            raise ValueError("ReplayBuffer requires at least one sample")
        if not self._game_samples:
            raise ValueError("ReplayBuffer requires at least one game")

    @classmethod
    def from_manifests(cls, manifests: list[str | Path], replay_games: int | str) -> "ReplayBuffer":
        manifest_records = []
        for manifest_path in manifests:
            manifest_path = Path(manifest_path)
            manifest = validate_manifest(manifest_path)
            manifest_records.append((_manifest_chronology_key(manifest_path, manifest), manifest_path, manifest))

        by_game: dict[tuple[float, int, str], list[Sample]] = {}
        for chronology_key, manifest_path, manifest in sorted(manifest_records, key=lambda record: record[0]):
            root = manifest_path.parent
            for shard in manifest["shard_paths"]:
                shard_path = Path(shard)
                if not shard_path.is_absolute():
                    shard_path = root / shard_path
                for sample in read_shard(shard_path):
                    game_key = (chronology_key[0], sample.game_id, chronology_key[1])
                    by_game.setdefault(game_key, []).append(sample)
        if replay_games == "all":
            kept_game_ids = sorted(by_game)
        else:
            if not isinstance(replay_games, int) or replay_games <= 0:
                raise ValueError("replay_games must be a positive integer or 'all'")
            kept_game_ids = sorted(by_game)[-replay_games:]
        game_samples = [by_game[game_id] for game_id in kept_game_ids]
        samples = [sample for game in game_samples for sample in game]
        return cls(samples, num_games=len(kept_game_ids), game_samples=game_samples)

    def sample_batch(
        self,
        batch_size: int,
        rng: random.Random,
        augment_symmetries: bool = False,
        sampling_config: ReplaySamplingConfig | None = None,
    ) -> list[Sample]:
        batch = self.sample_base_batch(batch_size, rng, sampling_config=sampling_config)
        if not augment_symmetries:
            return batch
        return [transform_sample(sample, rng.randrange(8)) for sample in batch]

    def sample_base_batch(
        self,
        batch_size: int,
        rng: random.Random,
        sampling_config: ReplaySamplingConfig | None = None,
    ) -> list[Sample]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        sampling_config = sampling_config or ReplaySamplingConfig()
        if sampling_config.mode == "uniform":
            return _sample_from_pool(self.samples, batch_size, rng)
        if sampling_config.mode == "recent-mix":
            return self._sample_recent_mix(batch_size, rng, sampling_config)
        raise ValueError(f"unknown replay sampling mode: {sampling_config.mode}")

    def sample_orbit_batch(
        self,
        base_batch_size: int,
        rng: random.Random,
        sampling_config: ReplaySamplingConfig | None = None,
    ) -> list[Sample]:
        base_batch = self.sample_base_batch(base_batch_size, rng, sampling_config=sampling_config)
        return [
            transform_sample(sample, symmetry)
            for sample in base_batch
            for symmetry in range(8)
        ]

    def _sample_recent_mix(
        self,
        batch_size: int,
        rng: random.Random,
        sampling_config: ReplaySamplingConfig,
    ) -> list[Sample]:
        recent_games = min(sampling_config.recent_games, len(self._game_samples))
        recent_pool = [sample for game in self._game_samples[-recent_games:] for sample in game]
        archive_pool = [sample for game in self._game_samples[:-recent_games] for sample in game]
        if not archive_pool or sampling_config.recent_fraction >= 1.0:
            return _sample_from_pool(recent_pool, batch_size, rng)
        if sampling_config.recent_fraction <= 0.0:
            return _sample_from_pool(archive_pool, batch_size, rng)

        recent_count = int(round(batch_size * sampling_config.recent_fraction))
        recent_count = max(1, min(batch_size - 1, recent_count))
        archive_count = batch_size - recent_count
        batch = (
            _sample_from_pool(recent_pool, recent_count, rng)
            + _sample_from_pool(archive_pool, archive_count, rng)
        )
        rng.shuffle(batch)
        return batch

    def metadata(self) -> dict[str, int]:
        return {
            "num_samples": len(self.samples),
            "num_games": self._num_games,
        }

    def sampling_metadata(self, sampling_config: ReplaySamplingConfig) -> dict[str, int | float | str]:
        if sampling_config.mode == "uniform":
            return {"replay_sampling": "uniform"}
        recent_games = min(sampling_config.recent_games, len(self._game_samples))
        recent_samples = sum(len(game) for game in self._game_samples[-recent_games:])
        archive_games = len(self._game_samples) - recent_games
        archive_samples = len(self.samples) - recent_samples
        return {
            "replay_sampling": "recent-mix",
            "recent_games": recent_games,
            "recent_fraction": sampling_config.recent_fraction,
            "recent_samples": recent_samples,
            "archive_games": archive_games,
            "archive_samples": archive_samples,
        }


def _manifest_chronology_key(manifest_path: Path, manifest: dict) -> tuple[float, str]:
    created_at = manifest.get("created_at")
    if created_at is None:
        return (manifest_path.stat().st_mtime, str(manifest_path.resolve()))
    return (_parse_created_at(created_at).timestamp(), str(manifest_path.resolve()))


def _parse_created_at(created_at: str) -> datetime:
    if not isinstance(created_at, str):
        raise ValueError("manifest created_at must be an ISO-8601 string")
    normalized = created_at
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"manifest created_at is not valid ISO-8601: {created_at}") from error
    if parsed.tzinfo is None:
        raise ValueError("manifest created_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def _sample_from_pool(pool: list[Sample], count: int, rng: random.Random) -> list[Sample]:
    if not pool:
        raise ValueError("cannot sample from an empty replay pool")
    return [pool[rng.randrange(len(pool))] for _ in range(count)]


def _group_samples_by_game_id(samples: list[Sample]) -> list[list[Sample]]:
    grouped: dict[int, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.game_id, []).append(sample)
    return [grouped[game_id] for game_id in sorted(grouped)]
