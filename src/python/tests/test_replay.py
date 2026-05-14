import json

from c4zero_tools.datasets import HEADER, MAGIC, POLICY, SAMPLE_PREFIX, VALUE, VISITS
from c4zero_tools.version import current_version_info
from c4zero_train.replay import ReplayBuffer


def write_manifest_with_one_sample(root, value):
    (root / "shards").mkdir(parents=True)
    payload = bytearray()
    payload += HEADER.pack(MAGIC, 1, 0, 1)
    payload += SAMPLE_PREFIX.pack(1, 2, bytes([0] * 16), 0, 0, 0xFFFF, 0)
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
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root / "manifest.json"


def test_replay_treats_same_game_ids_in_different_manifests_as_different_games(tmp_path):
    first = write_manifest_with_one_sample(tmp_path / "first", 1.0)
    second = write_manifest_with_one_sample(tmp_path / "second", -1.0)
    replay = ReplayBuffer.from_manifests([first, second], replay_games=2)
    assert replay.metadata()["num_games"] == 2
    assert len(replay.samples) == 2
    assert sorted(sample.value for sample in replay.samples) == [-1.0, 1.0]
