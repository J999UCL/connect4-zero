import struct

import json

from c4zero_tools.datasets import HEADER, MAGIC, POLICY, SAMPLE_PREFIX, VALUE, VISITS, read_shard, validate_manifest
from c4zero_tools.version import current_version_info


def test_read_single_sample_shard(tmp_path):
    shard = tmp_path / "sample.c4az"
    payload = bytearray()
    payload += HEADER.pack(MAGIC, 1, 0, 1)
    payload += SAMPLE_PREFIX.pack(1, 2, bytes(range(16)), 3, 4, 0xFFFF, 5)
    payload += POLICY.pack(*([1.0 / 16.0] * 16))
    payload += VISITS.pack(*range(16))
    payload += VALUE.pack(-1.0)
    shard.write_bytes(payload)

    [sample] = read_shard(shard)
    assert sample.current_bits == 1
    assert sample.opponent_bits == 2
    assert sample.heights == tuple(range(16))
    assert sample.ply == 3
    assert sample.game_id == 4
    assert sample.legal_mask == 0xFFFF
    assert sample.action == 5
    assert sample.value == -1.0
    assert int(sample.visit_counts.sum()) == sum(range(16))


def test_manifest_accepts_relative_shard_paths(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard = shard_dir / "sample.c4az"
    payload = bytearray()
    payload += HEADER.pack(MAGIC, 1, 0, 0)
    shard.write_bytes(payload)

    manifest = {
        "schema_version": current_version_info()["dataset_schema_version"],
        "num_games": 0,
        "num_samples": 0,
        "shard_paths": ["shards/sample.c4az"],
        "version": current_version_info(),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_manifest(manifest_path)["shard_paths"] == ["shards/sample.c4az"]
