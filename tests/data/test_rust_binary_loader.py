import json
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd

from connect4_zero.data import RustBinarySelfPlayDataset


def test_rust_binary_dataset_loads_custom_shard(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard_path = shard_dir / "shard-000000.c4zst"

    payload = bytearray()
    payload.extend(b"C4ZRS001")
    payload.extend(np.array([1], dtype="<u8").tobytes())
    payload.extend(np.array([1], dtype="<u8").tobytes())
    payload.extend(np.array([0], dtype="<u8").tobytes())
    heights = np.zeros((1, 16), dtype="u1")
    heights[0, 0] = 1
    payload.extend(heights.tobytes())
    policy = np.zeros((1, 16), dtype="<f4")
    policy[0, 4] = 1.0
    payload.extend(policy.tobytes())
    payload.extend(np.array([1], dtype="i1").tobytes())
    visits = np.zeros((1, 16), dtype="<u4")
    visits[0, 4] = 9
    payload.extend(visits.tobytes())
    q_values = np.zeros((1, 16), dtype="<f4")
    q_values[0, 4] = 0.75
    payload.extend(q_values.tobytes())
    payload.extend(np.array([0xFFFE], dtype="<u2").tobytes())
    payload.extend(np.array([4], dtype="u1").tobytes())
    payload.extend(np.array([0], dtype="u1").tobytes())
    shard_path.write_bytes(zstd.ZstdCompressor().compress(bytes(payload)))

    manifest = {
        "format": "connect4_zero.rust.selfplay.v1",
        "format_version": 1,
        "samples": 1,
        "generator": "test",
        "model_path": None,
        "shards": [{"path": "shards/shard-000000.c4zst", "samples": 1, "compression": "zstd"}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = RustBinarySelfPlayDataset(tmp_path / "manifest.json")
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["board"].shape == (4, 4, 4)
    assert sample["board"][0, 0, 0].item() == 1
    assert sample["input"].shape == (2, 4, 4, 4)
    assert sample["policy"][4].item() == 1.0
    assert sample["value"].item() == 1.0
    assert sample["visit_counts"][4].item() == 9
    assert torch.isclose(sample["q_values"][4], torch.tensor(0.75))
    assert sample["legal_mask"][0].item() is False
    assert sample["action"].item() == 4
