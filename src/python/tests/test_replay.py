import json
import random

import pytest

from c4zero_tools.datasets import HEADER, MAGIC, POLICY, SAMPLE_PREFIX, VALUE, VISITS
from c4zero_tools.version import current_version_info
from c4zero_train.replay import ReplayBuffer, ReplaySamplingConfig


def write_manifest_with_one_sample(root, value, created_at=None, game_id=0):
    (root / "shards").mkdir(parents=True)
    payload = bytearray()
    payload += HEADER.pack(MAGIC, 1, 0, 1)
    payload += SAMPLE_PREFIX.pack(1, 2, bytes([0] * 16), 0, game_id, 0xFFFF, 0)
    payload += POLICY.pack(*([1.0 / 16.0] * 16))
    payload += VISITS.pack(*([1] * 16))
    payload += VALUE.pack(value)
    (root / "shards" / "shard-000000.c4az").write_bytes(payload)
    manifest = {
        "schema_version": current_version_info()["dataset_schema_version"],
        "num_games": 1,
        "num_samples": 1,
        "shard_paths": ["shards/shard-000000.c4az"],
        "version": current_version_info(),
    }
    if created_at is not None:
        manifest["created_at"] = created_at
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root / "manifest.json"


def write_manifest_with_samples(root, values, created_at=None):
    (root / "shards").mkdir(parents=True)
    payload = bytearray()
    payload += HEADER.pack(MAGIC, 1, 0, len(values))
    for game_id, value in enumerate(values):
        payload += SAMPLE_PREFIX.pack(1, 2, bytes([0] * 16), 0, game_id, 0xFFFF, 0)
        payload += POLICY.pack(*([1.0 / 16.0] * 16))
        payload += VISITS.pack(*([1] * 16))
        payload += VALUE.pack(value)
    (root / "shards" / "shard-000000.c4az").write_bytes(payload)
    manifest = {
        "schema_version": current_version_info()["dataset_schema_version"],
        "num_games": len(values),
        "num_samples": len(values),
        "shard_paths": ["shards/shard-000000.c4az"],
        "version": current_version_info(),
    }
    if created_at is not None:
        manifest["created_at"] = created_at
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root / "manifest.json"


def test_replay_treats_same_game_ids_in_different_manifests_as_different_games(tmp_path):
    first = write_manifest_with_one_sample(tmp_path / "first", 1.0)
    second = write_manifest_with_one_sample(tmp_path / "second", -1.0)
    replay = ReplayBuffer.from_manifests([first, second], replay_games=2)
    assert replay.metadata()["num_games"] == 2
    assert len(replay.samples) == 2
    assert sorted(sample.value for sample in replay.samples) == [-1.0, 1.0]


def test_replay_keeps_latest_manifest_ordered_games(tmp_path):
    first = write_manifest_with_one_sample(tmp_path / "first", 1.0, "2026-05-14T00:00:00Z")
    second = write_manifest_with_one_sample(tmp_path / "second", -1.0, "2026-05-14T01:00:00Z")
    replay = ReplayBuffer.from_manifests([first, second], replay_games=1)
    assert replay.metadata()["num_games"] == 1
    assert len(replay.samples) == 1
    assert replay.samples[0].value == -1.0


def test_replay_games_all_keeps_entire_chronological_window(tmp_path):
    first = write_manifest_with_one_sample(tmp_path / "first", 1.0, "2026-05-14T00:00:00Z")
    second = write_manifest_with_one_sample(tmp_path / "second", 0.0, "2026-05-14T01:00:00Z")
    third = write_manifest_with_one_sample(tmp_path / "third", -1.0, "2026-05-14T02:00:00Z")
    replay = ReplayBuffer.from_manifests([third, first, second], replay_games="all")
    assert replay.metadata()["num_games"] == 3
    assert len(replay.samples) == 3
    assert [sample.value for sample in replay.samples] == [1.0, 0.0, -1.0]


def test_replay_sorts_manifest_timestamps_before_windowing(tmp_path):
    first = write_manifest_with_one_sample(tmp_path / "first", 1.0, "2026-05-14T01:00:00Z")
    second = write_manifest_with_one_sample(tmp_path / "second", -1.0, "2026-05-14T00:00:00Z")
    replay = ReplayBuffer.from_manifests([first, second], replay_games=1)
    assert replay.metadata()["num_games"] == 1
    assert replay.samples[0].value == 1.0


def test_replay_keeps_latest_game_ids_within_same_manifest_time(tmp_path):
    older = write_manifest_with_one_sample(tmp_path / "older", 1.0, "2026-05-14T00:00:00Z", game_id=9)
    newer = write_manifest_with_one_sample(tmp_path / "newer", -1.0, "2026-05-14T00:00:00Z", game_id=10)
    replay = ReplayBuffer.from_manifests([newer, older], replay_games=1)
    assert replay.metadata()["num_games"] == 1
    assert replay.samples[0].game_id == 10
    assert replay.samples[0].value == -1.0


def test_replay_rejects_unzoned_manifest_timestamp(tmp_path):
    manifest = write_manifest_with_one_sample(tmp_path / "manifest", 1.0, "2026-05-14T00:00:00")
    with pytest.raises(ValueError, match="must include a timezone"):
        ReplayBuffer.from_manifests([manifest], replay_games=1)


def test_replay_rejects_manifest_with_no_samples(tmp_path):
    (tmp_path / "shards").mkdir()
    shard = tmp_path / "shards" / "empty.c4az"
    shard.write_bytes(HEADER.pack(MAGIC, 1, 0, 0))
    manifest = {
        "schema_version": current_version_info()["dataset_schema_version"],
        "num_games": 0,
        "num_samples": 0,
        "shard_paths": ["shards/empty.c4az"],
        "version": current_version_info(),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="ReplayBuffer requires at least one sample"):
        ReplayBuffer.from_manifests([manifest_path], replay_games=1)


def test_replay_surfaces_missing_shard_artifact(tmp_path):
    manifest = {
        "schema_version": current_version_info()["dataset_schema_version"],
        "num_games": 1,
        "num_samples": 1,
        "shard_paths": ["shards/missing.c4az"],
        "version": current_version_info(),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        ReplayBuffer.from_manifests([manifest_path], replay_games=1)


def test_recent_mix_samples_more_from_recent_games(tmp_path):
    older = write_manifest_with_samples(tmp_path / "older", [1.0, 1.0, 1.0], "2026-05-14T00:00:00Z")
    newer = write_manifest_with_samples(tmp_path / "newer", [-1.0, -1.0], "2026-05-14T01:00:00Z")
    replay = ReplayBuffer.from_manifests([older, newer], replay_games="all")
    batch = replay.sample_base_batch(
        20,
        rng=random.Random(7),
        sampling_config=ReplaySamplingConfig(mode="recent-mix", recent_games=2, recent_fraction=0.75),
    )
    recent = sum(1 for sample in batch if sample.value == -1.0)
    archive = sum(1 for sample in batch if sample.value == 1.0)
    assert recent == 15
    assert archive == 5
    assert replay.sampling_metadata(ReplaySamplingConfig(mode="recent-mix", recent_games=2, recent_fraction=0.75)) == {
        "replay_sampling": "recent-mix",
        "recent_games": 2,
        "recent_fraction": 0.75,
        "recent_samples": 2,
        "archive_games": 3,
        "archive_samples": 3,
    }
