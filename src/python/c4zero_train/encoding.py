from __future__ import annotations

import numpy as np
import torch

from c4zero_tools.datasets import Sample


def encode_bits(current_bits: int, opponent_bits: int) -> np.ndarray:
    planes = np.zeros((2, 4, 4, 4), dtype=np.float32)
    for z in range(4):
        for y in range(4):
            for x in range(4):
                bit = 1 << (z * 16 + y * 4 + x)
                if current_bits & bit:
                    planes[0, z, y, x] = 1.0
                if opponent_bits & bit:
                    planes[1, z, y, x] = 1.0
    return planes


def encode_samples(samples: list[Sample], device: torch.device | str = "cpu") -> torch.Tensor:
    arrays = [encode_bits(sample.current_bits, sample.opponent_bits) for sample in samples]
    if not arrays:
        return torch.empty((0, 2, 4, 4, 4), dtype=torch.float32, device=device)
    return torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)
