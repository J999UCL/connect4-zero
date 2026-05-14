import json

import pytest

from c4zero_tools.datasets import HEADER, MAGIC, POLICY, SAMPLE_PREFIX, VALUE, VISITS
from c4zero_tools.version import current_version_info
from c4zero_train.replay import ReplayBuffer


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
