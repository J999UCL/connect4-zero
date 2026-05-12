"""Safetensors shard writer for generated self-play samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch
from safetensors.torch import save_file

from connect4_zero.data.types import SelfPlaySamples


class SelfPlayShardWriter:
    """Write append-free safetensor shards plus a JSONL manifest."""

    def __init__(
        self,
        output_dir: Path | str,
        samples_per_shard: int = 16384,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> None:
        if samples_per_shard <= 0:
            raise ValueError("samples_per_shard must be positive")
        self.output_dir = Path(output_dir)
        self.shard_dir = self.output_dir / "shards"
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.samples_per_shard = int(samples_per_shard)
        self.metadata = dict(metadata or {})
        self._next_shard_index = 0

        self.shard_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            with self.manifest_path.open("r", encoding="utf-8") as manifest:
                self._next_shard_index = sum(1 for _ in manifest)

    def write(self, samples: SelfPlaySamples) -> None:
        """Write ``samples`` across one or more shards."""
        samples.validate()
        for start in range(0, samples.num_samples, self.samples_per_shard):
            end = min(start + self.samples_per_shard, samples.num_samples)
            self._write_shard(samples.slice(start, end))

    def _write_shard(self, samples: SelfPlaySamples) -> None:
        shard_name = f"shard-{self._next_shard_index:06d}.safetensors"
        shard_path = self.shard_dir / shard_name
        tensors = self._coerce_tensors(samples)
        metadata = {
            **self.metadata,
            "num_samples": str(samples.num_samples),
            "format": "connect4_zero.selfplay.v1",
        }

        save_file(tensors, shard_path, metadata=metadata)
        self._append_manifest(shard_path, samples, tensors)
        self._next_shard_index += 1

    def _coerce_tensors(self, samples: SelfPlaySamples) -> Dict[str, torch.Tensor]:
        return {
            "boards": samples.boards.to(device="cpu", dtype=torch.int8).contiguous(),
            "policies": samples.policies.to(device="cpu", dtype=torch.float16).contiguous(),
            "values": samples.values.to(device="cpu", dtype=torch.int8).contiguous(),
            "visit_counts": samples.visit_counts.to(device="cpu", dtype=torch.int32).contiguous(),
            "q_values": samples.q_values.to(device="cpu", dtype=torch.float16).contiguous(),
            "legal_masks": samples.legal_masks.to(device="cpu", dtype=torch.bool).contiguous(),
            "actions": samples.actions.to(device="cpu", dtype=torch.uint8).contiguous(),
            "plies": samples.plies.to(device="cpu", dtype=torch.int16).contiguous(),
        }

    def _append_manifest(
        self,
        shard_path: Path,
        samples: SelfPlaySamples,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        record = {
            "shard": str(shard_path.relative_to(self.output_dir)),
            "num_samples": samples.num_samples,
            "tensors": {
                name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                for name, tensor in tensors.items()
            },
            "metadata": self.metadata,
        }
        with self.manifest_path.open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(record, sort_keys=True))
            manifest.write("\n")
