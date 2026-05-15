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


class ReplayBuffer:
    def __init__(self, samples: list[Sample], num_games: int | None = None) -> None:
        self.samples = samples
        self._num_games = num_games if num_games is not None else len({sample.game_id for sample in samples})
        if not samples:
            raise ValueError("ReplayBuffer requires at least one sample")

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
        samples = [sample for game_id in kept_game_ids for sample in by_game[game_id]]
        return cls(samples, num_games=len(kept_game_ids))

    def sample_batch(self, batch_size: int, rng: random.Random, augment_symmetries: bool = False) -> list[Sample]:
        batch = [self.samples[rng.randrange(len(self.samples))] for _ in range(batch_size)]
        if not augment_symmetries:
            return batch
        return [transform_sample(sample, rng.randrange(8)) for sample in batch]

    def metadata(self) -> dict[str, int]:
        return {
            "num_samples": len(self.samples),
            "num_games": self._num_games,
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
