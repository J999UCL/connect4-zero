from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import json
import pickle
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from c4zero_train.checkpoint import load_checkpoint

INVALID_MOVE_VALUE = np.iinfo(np.int32).min
MATE_THRESHOLD = 90_000
ACTION_SIZE = 16
BOARD_CELLS = 64


@dataclass(frozen=True, slots=True)
class EvalSetMetrics:
    n_input: int
    n_evaluated: int
    n_dropped_unprobed: int
    n_dropped_incomplete: int
    optimality_rate: float
    mean_value_loss: float
    mate_find_rate: float
    blunder_rate: float
    n_mate: int
    n_safe: int
    action_counts: list[int]


@dataclass(frozen=True, slots=True)
class HeadToHeadResult:
    opponent: str
    games: int
    wins: int
    losses: int
    draws: int
    winrate: float


def load_eval_set(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"expected eval set list, got {type(records).__name__}")
    return records


def _canonical_bits(bb0: int, bb1: int, turn: int) -> tuple[int, int]:
    if turn == 0:
        return int(bb0), int(bb1)
    if turn == 1:
        return int(bb1), int(bb0)
    raise ValueError(f"turn must be 0 or 1, got {turn}")


def _planes_from_bits(current_bits: int, opponent_bits: int) -> np.ndarray:
    planes = np.zeros((2, 4, 4, 4), dtype=np.float32)
    for cell in range(BOARD_CELLS):
        z = cell // 16
        y = (cell % 16) // 4
        x = cell % 4
        bit = 1 << cell
        if current_bits & bit:
            planes[0, z, y, x] = 1.0
        if opponent_bits & bit:
            planes[1, z, y, x] = 1.0
    return planes


def _signed_obs_to_planes(obs: torch.Tensor) -> torch.Tensor:
    if obs.ndim != 2 or obs.shape[1] != BOARD_CELLS:
        raise ValueError(f"expected signed obs [B,64], got {tuple(obs.shape)}")
    current = (obs > 0).to(torch.float32)
    opponent = (obs < 0).to(torch.float32)
    return torch.stack((current, opponent), dim=1).reshape(obs.shape[0], 2, 4, 4, 4)


class Score4PolicyAdapter(nn.Module):
    """Adapt Gerard's signed [B,64] observation convention to c4zero planes."""

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.model = model
        self.device = torch.device(device)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        planes = _signed_obs_to_planes(obs.to(self.device))
        out = self.model(planes)
        if isinstance(out, tuple):
            return out[0], out[1]
        raise ValueError("c4zero policy adapter expected model to return (logits, value)")


def _legal_from_heights(heights: Iterable[int]) -> np.ndarray:
    values = tuple(int(h) for h in heights)
    if len(values) != ACTION_SIZE:
        raise ValueError(f"expected 16 heights, got {len(values)}")
    return np.asarray([height < 4 for height in values], dtype=bool)


def _features_from_records(
    records: list[dict[str, Any]],
    require_complete_legal_values: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    planes: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    move_values: list[np.ndarray] = []
    dropped_unprobed = 0
    dropped_incomplete = 0
    for record in records:
        current, opponent = _canonical_bits(record["bb0"], record["bb1"], record["turn"])
        legal = _legal_from_heights(record["heights"])
        values = np.asarray(record["oracle_move_values"], dtype=np.int64)
        if values.shape != (ACTION_SIZE,):
            raise ValueError(f"oracle_move_values must have shape (16,), got {values.shape}")
        legal_values_are_probed = values[legal] > INVALID_MOVE_VALUE // 2
        if not np.any(legal_values_are_probed):
            dropped_unprobed += 1
            if require_complete_legal_values:
                dropped_incomplete += 1
                continue
        if require_complete_legal_values and not np.all(legal_values_are_probed):
            dropped_incomplete += 1
            continue
        planes.append(_planes_from_bits(current, opponent))
        legal_masks.append(legal)
        move_values.append(values)
    if not planes:
        return (
            np.empty((0, 2, 4, 4, 4), dtype=np.float32),
            np.empty((0, ACTION_SIZE), dtype=bool),
            np.empty((0, ACTION_SIZE), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            dropped_unprobed,
            dropped_incomplete,
        )
    x = np.stack(planes).astype(np.float32)
    legal = np.stack(legal_masks).astype(bool)
    values = np.stack(move_values).astype(np.int64)
    best = np.where(legal, values, INVALID_MOVE_VALUE).max(axis=1)
    return x, legal, values, best, dropped_unprobed, dropped_incomplete


def _masked_actions_from_model(
    model: nn.Module,
    inputs: np.ndarray,
    legal: np.ndarray,
    device: str | torch.device,
    batch_size: int,
) -> np.ndarray:
    if inputs.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    device = torch.device(device)
    model = model.to(device).eval()
    actions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            end = min(start + batch_size, inputs.shape[0])
            batch = torch.as_tensor(inputs[start:end], dtype=torch.float32, device=device)
            logits, _values = model(batch)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e30, neginf=-1e30)
            legal_batch = torch.as_tensor(legal[start:end], dtype=torch.bool, device=device)
            logits = logits.masked_fill(~legal_batch, -1e30)
            actions.append(logits.argmax(dim=-1).cpu().numpy().astype(np.int64))
    return np.concatenate(actions, axis=0)


def evaluate_model_on_records(
    model: nn.Module,
    records: list[dict[str, Any]],
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 1024,
    eps: int = 1,
    require_complete_legal_values: bool = True,
) -> EvalSetMetrics:
    inputs, legal, move_values, best_values, dropped_unprobed, dropped_incomplete = _features_from_records(
        records,
        require_complete_legal_values,
    )
    actions = _masked_actions_from_model(model, inputs, legal, device, batch_size)
    if actions.size == 0:
        return EvalSetMetrics(
            n_input=len(records),
            n_evaluated=0,
            n_dropped_unprobed=dropped_unprobed,
            n_dropped_incomplete=dropped_incomplete,
            optimality_rate=0.0,
            mean_value_loss=0.0,
            mate_find_rate=0.0,
            blunder_rate=0.0,
            n_mate=0,
            n_safe=0,
            action_counts=[0] * ACTION_SIZE,
        )
    row = np.arange(actions.shape[0])
    chosen_values = move_values[row, actions]
    optimal = (best_values - chosen_values) <= eps
    mate_mask = best_values > MATE_THRESHOLD
    safe_mask = best_values > -MATE_THRESHOLD
    mate_find = (chosen_values > MATE_THRESHOLD) & mate_mask
    blunder = (chosen_values < -MATE_THRESHOLD) & safe_mask
    action_counts = np.bincount(actions, minlength=ACTION_SIZE).astype(int).tolist()
    return EvalSetMetrics(
        n_input=len(records),
        n_evaluated=int(actions.shape[0]),
        n_dropped_unprobed=dropped_unprobed,
        n_dropped_incomplete=dropped_incomplete,
        optimality_rate=float(optimal.mean()),
        mean_value_loss=float((best_values - chosen_values).mean()),
        mate_find_rate=float(mate_find.sum() / max(int(mate_mask.sum()), 1)),
        blunder_rate=float(blunder.sum() / max(int(safe_mask.sum()), 1)),
        n_mate=int(mate_mask.sum()),
        n_safe=int(safe_mask.sum()),
        action_counts=action_counts,
    )


def _load_model(checkpoint: Path | None, torchscript: Path | None, device: str | torch.device) -> nn.Module:
    if (checkpoint is None) == (torchscript is None):
        raise ValueError("provide exactly one of --checkpoint or --torchscript")
    if checkpoint is not None:
        model, _payload = load_checkpoint(checkpoint, device=device)
        return model
    assert torchscript is not None
    return torch.jit.load(str(torchscript), map_location=torch.device(device)).eval()


def evaluate_checkpoint_on_eval_set(
    checkpoint: str | Path,
    eval_set: str | Path,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 1024,
    eps: int = 1,
) -> EvalSetMetrics:
    model, _payload = load_checkpoint(checkpoint, device=device)
    return evaluate_model_on_records(
        model,
        load_eval_set(eval_set),
        device=device,
        batch_size=batch_size,
        eps=eps,
    )


def _import_score4(score4_repo: Path | None):
    if score4_repo is not None:
        sys.path.insert(0, str(score4_repo))
        target_release = score4_repo / "target" / "release"
        if target_release.exists():
            sys.path.insert(0, str(target_release))
    try:
        return importlib.import_module("score4")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "score4 is not importable. Build/install Gerard's Python bindings first, "
            "or run only the fixed eval-set mode."
        ) from exc


def _network_actions(policy: nn.Module, obs_np: np.ndarray, legal_np: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits, _value = policy(torch.as_tensor(obs_np, dtype=torch.float32))
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e30, neginf=-1e30)
        logits = logits.masked_fill(~torch.as_tensor(legal_np, dtype=torch.bool, device=logits.device), -1e30)
        actions = logits.argmax(dim=-1).cpu().numpy().astype(np.uint8)
    for index, action in enumerate(actions):
        if action >= ACTION_SIZE or not bool(legal_np[index, int(action)]):
            legal_actions = np.flatnonzero(legal_np[index])
            actions[index] = int(legal_actions[0]) if legal_actions.size else 0
    return actions


def evaluate_against_score4_ladder(
    model: nn.Module,
    *,
    score4_repo: str | Path | None,
    device: str | torch.device = "cpu",
    opponents: list[str],
    games: int,
    n_envs: int,
    seed: int,
) -> list[HeadToHeadResult]:
    score4 = _import_score4(Path(score4_repo) if score4_repo is not None else None)
    policy = Score4PolicyAdapter(model, device=device).eval()
    oracle = score4.Oracle(tt_mb=16)
    vec = score4.VecEnv(min(n_envs, games))
    rng = np.random.default_rng(seed)
    output: list[HeadToHeadResult] = []
    oracle_time_ms = {"oracle_d2": 2, "oracle_d4": 10, "oracle_d8": 30}
    for opponent in opponents:
        oracle.clear()
        vec.reset_all()
        agent_side = rng.integers(0, 2, size=vec.n).astype(np.int8)
        wins = losses = draws = completed = 0
        heuristic_seed = int(seed)
        while completed < games:
            obs = vec.obs_signed().astype(np.float32)
            legal = vec.legal_masks()
            turns = vec.turns()
            agent_mask = turns == agent_side
            actions = np.zeros(vec.n, dtype=np.uint8)
            if agent_mask.any():
                actions[agent_mask] = _network_actions(policy, obs[agent_mask], legal[agent_mask])
            opponent_mask = ~agent_mask
            if opponent_mask.any():
                env_indices = np.flatnonzero(opponent_mask).astype(np.uint64)
                if opponent.startswith("oracle_d"):
                    depth = int(opponent.split("d", maxsplit=1)[1])
                    actions[opponent_mask] = oracle.oracle_actions(
                        vec,
                        env_indices,
                        max_depth=depth,
                        time_ms=oracle_time_ms.get(opponent, 20),
                    )
                else:
                    actions[opponent_mask] = vec.heuristic_actions(opponent, env_indices, seed=heuristic_seed)
                    heuristic_seed += 1
            for index, action in enumerate(actions):
                if action >= ACTION_SIZE or not bool(legal[index, int(action)]):
                    legal_actions = np.flatnonzero(legal[index])
                    actions[index] = int(legal_actions[0]) if legal_actions.size else 0
            _obs, dones, winners = vec.step(actions)
            if not dones.any():
                continue
            for index in np.flatnonzero(dones):
                winner = int(winners[index])
                side = int(agent_side[index])
                if winner < 0:
                    draws += 1
                elif winner == side:
                    wins += 1
                else:
                    losses += 1
                completed += 1
                agent_side[index] = rng.integers(0, 2)
                if completed >= games:
                    break
        total = max(wins + losses + draws, 1)
        output.append(
            HeadToHeadResult(
                opponent=opponent,
                games=total,
                wins=wins,
                losses=losses,
                draws=draws,
                winrate=float(wins / total),
            )
        )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate c4zero checkpoints using Gerard's Score Four oracle assets.")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--checkpoint", type=Path)
    model_group.add_argument("--torchscript", type=Path)
    parser.add_argument("--eval-set", type=Path, help="Path to score-four-rl eval_set.pkl.")
    parser.add_argument("--score4-repo", type=Path, help="Optional score-four-rl checkout for live score4 bindings.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eps", type=int, default=1)
    parser.add_argument("--keep-incomplete", action="store_true", help="Keep rows where some legal moves have no oracle value.")
    parser.add_argument("--head-to-head", action="store_true")
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args(argv)

    model = _load_model(args.checkpoint, args.torchscript, args.device)
    result: dict[str, Any] = {}
    if args.eval_set is not None:
        metrics = evaluate_model_on_records(
            model,
            load_eval_set(args.eval_set),
            device=args.device,
            batch_size=args.batch_size,
            eps=args.eps,
            require_complete_legal_values=not args.keep_incomplete,
        )
        result["eval_set"] = asdict(metrics)
    if args.head_to_head:
        opponents = args.opponent or ["random", "win1", "block1", "center", "threat", "oracle_d2"]
        ladder = evaluate_against_score4_ladder(
            model,
            score4_repo=args.score4_repo,
            device=args.device,
            opponents=opponents,
            games=args.games,
            n_envs=args.n_envs,
            seed=args.seed,
        )
        result["head_to_head"] = [asdict(item) for item in ladder]
    if not result:
        raise ValueError("nothing to evaluate: pass --eval-set and/or --head-to-head")

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
