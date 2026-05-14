from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

from c4zero_tools.datasets import Sample, read_shard, validate_manifest


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
    def from_manifests(cls, manifests: list[str | Path], replay_games: int) -> "ReplayBuffer":
        by_game: dict[tuple[int, int], list[Sample]] = {}
        previous_created_at: str | None = None
        for manifest_index, manifest_path in enumerate(manifests):
            manifest_path = Path(manifest_path)
            manifest = validate_manifest(manifest_path)
            created_at = manifest.get("created_at")
            if created_at is not None:
                if previous_created_at is not None and created_at < previous_created_at:
                    raise ValueError("manifest created_at values must be passed oldest-to-newest")
                previous_created_at = created_at
            root = manifest_path.parent
            for shard in manifest["shard_paths"]:
                shard_path = Path(shard)
                if not shard_path.is_absolute():
                    shard_path = root / shard_path
                for sample in read_shard(shard_path):
                    by_game.setdefault((manifest_index, sample.game_id), []).append(sample)
        kept_game_ids = sorted(by_game)[-replay_games:]
        samples = [sample for game_id in kept_game_ids for sample in by_game[game_id]]
        return cls(samples, num_games=len(kept_game_ids))

    def sample_batch(self, batch_size: int, rng: random.Random) -> list[Sample]:
        return [self.samples[rng.randrange(len(self.samples))] for _ in range(batch_size)]

    def metadata(self) -> dict[str, int]:
        return {
            "num_samples": len(self.samples),
            "num_games": self._num_games,
        }
