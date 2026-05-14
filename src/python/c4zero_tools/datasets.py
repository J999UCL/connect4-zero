from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from c4zero_tools.version import current_version_info


MAGIC = b"C4AZSP01"
HEADER = struct.Struct("<8sIIQ")
SAMPLE_PREFIX = struct.Struct("<QQ16sBQHB")
POLICY = struct.Struct("<16f")
VISITS = struct.Struct("<16I")
VALUE = struct.Struct("<f")


@dataclass(frozen=True)
class Sample:
    current_bits: int
    opponent_bits: int
    heights: tuple[int, ...]
    ply: int
    game_id: int
    legal_mask: int
    action: int
    policy: np.ndarray
    visit_counts: np.ndarray
    value: float


def read_shard(path: str | Path) -> list[Sample]:
    path = Path(path)
    data = path.read_bytes()
    magic, major, _minor, count = HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"{path} is not a c4zero shard")
    if major != 1:
        raise ValueError(f"unsupported c4zero shard major schema: {major}")

    offset = HEADER.size
    samples: list[Sample] = []
    for _ in range(count):
        current, opponent, heights_raw, ply, game_id, legal_mask, action = SAMPLE_PREFIX.unpack_from(data, offset)
        offset += SAMPLE_PREFIX.size
        policy = np.array(POLICY.unpack_from(data, offset), dtype=np.float32)
        offset += POLICY.size
        visits = np.array(VISITS.unpack_from(data, offset), dtype=np.uint32)
        offset += VISITS.size
        (value,) = VALUE.unpack_from(data, offset)
        offset += VALUE.size
        samples.append(
            Sample(
                current_bits=current,
                opponent_bits=opponent,
                heights=tuple(heights_raw),
                ply=ply,
                game_id=game_id,
                legal_mask=legal_mask,
                action=action,
                policy=policy,
                visit_counts=visits,
                value=float(value),
            )
        )
    return samples


def validate_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    version = current_version_info()
    required = ["schema_version", "num_games", "num_samples", "shard_paths", "version"]
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"manifest missing required keys: {missing}")
    if manifest["schema_version"].split(".")[0] != version["dataset_schema_version"].split(".")[0]:
        raise ValueError("dataset schema major version mismatch")
    for key in ["game_rules_version", "encoder_version", "action_mapping_version", "symmetry_version"]:
        if manifest["version"].get(key) != version[key]:
            raise ValueError(f"{key} mismatch: {manifest['version'].get(key)} != {version[key]}")
    return manifest


def encode_sample(sample: Sample) -> np.ndarray:
    planes = np.zeros((2, 4, 4, 4), dtype=np.float32)
    for z in range(4):
        for y in range(4):
            for x in range(4):
                bit = 1 << (z * 16 + y * 4 + x)
                if sample.current_bits & bit:
                    planes[0, z, y, x] = 1.0
                if sample.opponent_bits & bit:
                    planes[1, z, y, x] = 1.0
    return planes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and inspect c4zero dataset manifests.")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    manifest = validate_manifest(args.manifest)
    root = args.manifest.parent
    sample_count = 0
    for shard in manifest["shard_paths"]:
      shard_path = Path(shard)
      if not shard_path.is_absolute():
          shard_path = root / shard_path
      sample_count += len(read_shard(shard_path))
    if sample_count != manifest["num_samples"]:
        raise ValueError(f"manifest sample count {manifest['num_samples']} != shard count {sample_count}")
    print(json.dumps({"num_games": manifest["num_games"], "num_samples": sample_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
