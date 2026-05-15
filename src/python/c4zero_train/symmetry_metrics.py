from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from c4zero_tools.datasets import Sample
from c4zero_train.encoding import encode_samples
from c4zero_train.model import AlphaZeroNet
from c4zero_train.replay import ReplayBuffer
from c4zero_train.symmetry import NUM_ACTIONS, action_permutation, transform_policy, transform_sample


ACTION_GROUPS = {
    "corners": (0, 3, 12, 15),
    "edges": (1, 2, 4, 7, 8, 11, 13, 14),
    "centers": (5, 6, 9, 10),
}


@dataclass(frozen=True, slots=True)
class SymmetryProbeConfig:
    positions: int = 256
    seed: int = 10_001
    batch_size: int = 1024


def empty_sample() -> Sample:
    policy = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)
    visits = np.ones(NUM_ACTIONS, dtype=np.uint32)
    return Sample(
        current_bits=0,
        opponent_bits=0,
        heights=tuple([0] * NUM_ACTIONS),
        ply=0,
        game_id=0,
        legal_mask=(1 << NUM_ACTIONS) - 1,
        action=0,
        policy=policy,
        visit_counts=visits,
        value=0.0,
    )


@torch.no_grad()
def evaluate_symmetry(
    model: AlphaZeroNet,
    replay: ReplayBuffer,
    device: torch.device | str = "cpu",
    config: SymmetryProbeConfig = SymmetryProbeConfig(),
) -> dict:
    model.eval()
    model.to(device)
    selected = _fixed_probe_samples(replay, config.positions, config.seed)
    return {
        "empty_board": empty_board_metrics(model, device=device),
        "equivariance": equivariance_metrics(model, selected, device=device, batch_size=config.batch_size),
        "probe_positions": len(selected),
    }


@torch.no_grad()
def empty_board_metrics(model: AlphaZeroNet, device: torch.device | str = "cpu") -> dict:
    sample = empty_sample()
    logits, value = _model_outputs(model, [sample], device=device)
    probs = _masked_softmax(logits, [sample.legal_mask])[0]
    policy = {str(action): float(probs[action]) for action in range(NUM_ACTIONS)}
    return {
        "policy": policy,
        "value": float(value[0]),
        "groups": _group_metrics(probs),
    }


@torch.no_grad()
def equivariance_metrics(
    model: AlphaZeroNet,
    samples: Iterable[Sample],
    device: torch.device | str = "cpu",
    batch_size: int = 1024,
) -> dict:
    samples = list(samples)
    if not samples:
        return {
            "positions": 0,
            "mean_policy_l1": 0.0,
            "max_policy_abs": 0.0,
            "mean_value_std": 0.0,
            "max_value_std": 0.0,
        }

    l1_values: list[float] = []
    max_values: list[float] = []
    value_stds: list[float] = []
    for sample in samples:
        orbit = [transform_sample(sample, symmetry) for symmetry in range(8)]
        logits, values = _model_outputs(model, orbit, device=device, batch_size=batch_size)
        policies = _masked_softmax(logits, [item.legal_mask for item in orbit])
        base_policy = policies[0]
        for symmetry in range(1, 8):
            expected = transform_policy(base_policy, action_permutation(symmetry))
            diff = np.abs(policies[symmetry] - expected)
            l1_values.append(float(diff.sum()))
            max_values.append(float(diff.max()))
        value_stds.append(float(np.std(values)))

    return {
        "positions": len(samples),
        "mean_policy_l1": float(np.mean(l1_values)),
        "max_policy_abs": float(np.max(max_values)),
        "mean_value_std": float(np.mean(value_stds)),
        "max_value_std": float(np.max(value_stds)),
    }


def _fixed_probe_samples(replay: ReplayBuffer, count: int, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    count = min(count, len(replay.samples))
    return [replay.samples[index] for index in rng.sample(range(len(replay.samples)), count)]


def _group_metrics(policy: np.ndarray) -> dict[str, dict[str, float]]:
    payload = {}
    for name, actions in ACTION_GROUPS.items():
        values = np.array([policy[action] for action in actions], dtype=np.float64)
        payload[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "max_minus_min": float(values.max() - values.min()),
            "max_abs_from_mean": float(np.abs(values - values.mean()).max()),
        }
    return payload


def _model_outputs(
    model: AlphaZeroNet,
    samples: list[Sample],
    device: torch.device | str,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    logits_chunks = []
    value_chunks = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        inputs = encode_samples(chunk, device=device)
        logits, values = model(inputs)
        logits_chunks.append(logits.detach().cpu())
        value_chunks.append(values.detach().cpu())
    return (
        torch.cat(logits_chunks, dim=0).numpy(),
        torch.cat(value_chunks, dim=0).numpy(),
    )


def _masked_softmax(logits: np.ndarray, legal_masks: list[int]) -> np.ndarray:
    tensor = torch.as_tensor(logits, dtype=torch.float32)
    action_ids = torch.arange(NUM_ACTIONS, dtype=torch.int64)
    legal = torch.as_tensor(legal_masks, dtype=torch.int64)
    mask = ((legal[:, None] >> action_ids[None, :]) & 1).bool()
    empty_rows = ~mask.any(dim=1)
    if empty_rows.any():
        mask[empty_rows] = True
    masked = tensor.masked_fill(~mask, float("-inf"))
    return F.softmax(masked, dim=1).numpy().astype(np.float32)
