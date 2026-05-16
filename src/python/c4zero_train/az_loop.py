from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Iterable


@dataclass(frozen=True, slots=True)
class SelfPlayShard:
    index: int
    games: int
    seed: int
    out_dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / "manifest.json"

    @property
    def log_path(self) -> Path:
        return self.out_dir / "selfplay.log"


@dataclass(frozen=True, slots=True)
class CommandResult:
    command: list[str]
    returncode: int
    elapsed_sec: float
    log_path: Path


def split_games(total_games: int, processes: int) -> list[int]:
    if total_games <= 0:
        raise ValueError("total_games must be positive")
    if processes <= 0:
        raise ValueError("processes must be positive")
    active = min(total_games, processes)
    base = total_games // active
    remainder = total_games % active
    return [base + (1 if index < remainder else 0) for index in range(active)]


def selfplay_shards(round_dir: Path, total_games: int, processes: int, seed: int) -> list[SelfPlayShard]:
    return [
        SelfPlayShard(index=index, games=games, seed=seed + index, out_dir=round_dir / f"selfplay-{index:02d}")
        for index, games in enumerate(split_games(total_games, processes))
    ]


def load_manifest_list(paths: Iterable[str | Path], list_files: Iterable[str | Path]) -> list[str]:
    manifests: list[str] = [str(Path(path)) for path in paths]
    for list_file in list_files:
        list_path = Path(list_file)
        if not list_path.exists():
            raise FileNotFoundError(f"manifest list does not exist: {list_path}")
        for line in list_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                manifests.append(str(Path(stripped)))
    return manifests


def write_manifest_list(path: Path, manifests: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{manifest}\n" for manifest in manifests), encoding="utf-8")


def parse_key_value_summary(summary: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for token in summary.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key] = value
    return parsed


def _quote(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def _set_status(run_root: Path, text: str) -> None:
    (run_root / "status.txt").write_text(text.rstrip() + "\n", encoding="utf-8")


def run_command(command: list[str], log_path: Path, supervisor_log: Path) -> CommandResult:
    started = time.perf_counter()
    _append_line(supervisor_log, f"command_start log={log_path} command={_quote(command)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {_quote(command)}\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - started
    _append_line(
        supervisor_log,
        f"command_done returncode={completed.returncode} elapsed_sec={elapsed:.3f} log={log_path}",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {_quote(command)}")
    return CommandResult(command=command, returncode=completed.returncode, elapsed_sec=elapsed, log_path=log_path)


def run_parallel_selfplay(
    *,
    c4zero_bin: Path,
    model_path: Path,
    device: str,
    simulations: int,
    game_workers: int,
    search_threads: int,
    inference_batch_size: int,
    inference_max_wait_us: int,
    value_mode: str,
    virtual_loss: float,
    shards: list[SelfPlayShard],
    supervisor_log: Path,
) -> list[str]:
    processes: list[tuple[SelfPlayShard, subprocess.Popen]] = []
    started = time.perf_counter()
    for shard in shards:
        shard.out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(c4zero_bin),
            "selfplay",
            "--model",
            str(model_path),
            "--device",
            device,
            "--games",
            str(shard.games),
            "--simulations",
            str(simulations),
            "--game-workers",
            str(game_workers),
            "--search-threads",
            str(search_threads),
            "--inference-batch-size",
            str(inference_batch_size),
            "--inference-max-wait-us",
            str(inference_max_wait_us),
            "--virtual-loss",
            str(virtual_loss),
            "--value-mode",
            value_mode,
            "--seed",
            str(shard.seed),
            "--out",
            str(shard.out_dir),
        ]
        _append_line(supervisor_log, f"selfplay_start shard={shard.index} games={shard.games} command={_quote(command)}")
        log = shard.log_path.open("w", encoding="utf-8")
        log.write(f"$ {_quote(command)}\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        processes.append((shard, process))

    failed: list[tuple[SelfPlayShard, int]] = []
    for shard, process in processes:
        returncode = process.wait()
        _append_line(supervisor_log, f"selfplay_done shard={shard.index} returncode={returncode} log={shard.log_path}")
        if returncode != 0:
            failed.append((shard, returncode))
    elapsed = time.perf_counter() - started
    if failed:
        details = ", ".join(f"shard={shard.index} rc={returncode}" for shard, returncode in failed)
        raise RuntimeError(f"self-play shard failure(s): {details}")

    manifests = [str(shard.manifest_path) for shard in shards]
    missing = [manifest for manifest in manifests if not Path(manifest).exists()]
    if missing:
        raise FileNotFoundError(f"self-play shard manifest(s) missing: {missing}")
    _append_line(
        supervisor_log,
        f"selfplay_all_done shards={len(shards)} games={sum(shard.games for shard in shards)} elapsed_sec={elapsed:.3f}",
    )
    return manifests


def run_arena(
    *,
    c4zero_bin: Path,
    model_a: Path,
    model_b: Path | None,
    bot_b: str | None,
    device: str,
    games: int,
    simulations: int,
    search_threads: int,
    arena_workers: int,
    opening_count: int,
    opening_plies: int,
    games_per_opening: int,
    promotion_threshold: float,
    seed: int,
    log_path: Path,
    supervisor_log: Path,
) -> dict[str, str]:
    if (model_b is None) == (bot_b is None):
        raise ValueError("exactly one of model_b or bot_b must be provided")
    command = [
        str(c4zero_bin),
        "arena",
        "--model-a",
        str(model_a),
        "--device",
        device,
        "--games",
        str(games),
        "--simulations",
        str(simulations),
        "--search-threads",
        str(search_threads),
        "--arena-workers",
        str(arena_workers),
        "--opening-count",
        str(opening_count),
        "--opening-plies",
        str(opening_plies),
        "--games-per-opening",
        str(games_per_opening),
        "--promotion-threshold",
        str(promotion_threshold),
        "--no-root-noise",
        "--seed",
        str(seed),
    ]
    if model_b is not None:
        command.extend(["--model-b", str(model_b)])
    else:
        command.extend(["--bot-b", str(bot_b)])
    run_command(command, log_path, supervisor_log)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    summary = parse_key_value_summary(lines[-1] if lines else "")
    if "model_a_score_rate" not in summary:
        raise RuntimeError(f"arena summary did not contain model_a_score_rate: {log_path}")
    return summary


def _csv_append(path: Path, header: str, row: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(header.rstrip() + "\n", encoding="utf-8")
    _append_line(path, row)


def _train_command(args: argparse.Namespace, manifests: list[str], current_checkpoint: Path, round_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "c4zero_train.cli",
        "--resume",
        str(current_checkpoint),
        "--reset-optimizer",
        "--steps",
        str(args.train_steps),
        "--out",
        str(round_dir / "checkpoint"),
        "--device",
        args.train_device,
        "--replay-games",
        args.replay_games,
        "--learning-rate",
        str(args.learning_rate),
        "--momentum",
        str(args.momentum),
        "--weight-decay",
        str(args.weight_decay),
        "--policy-weight",
        str(args.policy_weight),
        "--value-weight",
        str(args.value_weight),
        "--symmetry-mode",
        args.symmetry_mode,
        "--batch-size",
        str(args.batch_size),
        "--replay-sampling",
        args.replay_sampling,
        "--recent-games",
        str(args.recent_games),
        "--recent-fraction",
        str(args.recent_fraction),
        "--seed",
        str(args.seed + args.current_round * 1000 + 17),
        "--log-every-steps",
        str(args.log_every_steps),
    ]
    for manifest in manifests:
        command.extend(["--manifest", manifest])
    return command


def _jsonable_config(args: argparse.Namespace) -> dict:
    config = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, list):
            config[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            config[key] = value
    return config


def _read_checkpoint_metrics(checkpoint_dir: Path) -> dict:
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8")).get("metrics", {})


def loop_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AlphaZero rounds with parallel self-play shards.")
    parser.add_argument("--c4zero-bin", type=Path, required=True)
    parser.add_argument("--start-checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--first-round", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--prior-manifest", action="append", default=[])
    parser.add_argument("--prior-manifest-file", action="append", default=[])
    parser.add_argument("--games-per-round", type=int, default=4000)
    parser.add_argument("--selfplay-processes", type=int, default=8)
    parser.add_argument("--selfplay-game-workers", type=int, default=4)
    parser.add_argument("--selfplay-search-threads", type=int, default=2)
    parser.add_argument("--selfplay-device", default="cuda")
    parser.add_argument("--selfplay-simulations", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--inference-max-wait-us", type=int, default=2000)
    parser.add_argument("--virtual-loss", type=float, default=1.0)
    parser.add_argument("--value-mode", choices=["zero", "model"], default="model")
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--train-steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--symmetry-mode", choices=["none", "random", "orbit"], default="orbit")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-games", default="all")
    parser.add_argument("--replay-sampling", choices=["uniform", "recent-mix"], default="recent-mix")
    parser.add_argument("--recent-games", type=int, default=4000)
    parser.add_argument("--recent-fraction", type=float, default=0.75)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--arena-device", default="cuda")
    parser.add_argument("--arena-games", type=int, default=64)
    parser.add_argument("--arena-simulations", type=int, default=256)
    parser.add_argument("--arena-search-threads", type=int, default=2)
    parser.add_argument("--arena-workers", type=int, default=4)
    parser.add_argument("--opening-count", type=int, default=16)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--games-per-opening", type=int, default=4)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument("--minimax-bot", action="append")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.minimax_bot is None:
        args.minimax_bot = ["minimax3", "minimax5"]

    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.games_per_round <= 0:
        raise ValueError("--games-per-round must be positive")

    args.run_root.mkdir(parents=True, exist_ok=True)
    args.data_root.mkdir(parents=True, exist_ok=True)
    supervisor_log = args.run_root / "supervisor.log"
    manifests = load_manifest_list(args.prior_manifest, args.prior_manifest_file)
    write_manifest_list(args.run_root / "manifests_so_far.txt", manifests)

    current_checkpoint = args.start_checkpoint
    current_round = args.first_round - 1
    config_snapshot = _jsonable_config(args)
    (args.run_root / "loop_config.json").write_text(json.dumps(config_snapshot, indent=2, sort_keys=True), encoding="utf-8")

    for round_index in range(args.first_round, args.first_round + args.rounds):
        args.current_round = round_index
        round_dir = args.run_root / f"round-{round_index:02d}"
        round_data_dir = args.data_root / f"round-{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        _set_status(args.run_root, f"round={round_index}/{args.first_round + args.rounds - 1} stage=selfplay current_round={current_round}")
        shards = selfplay_shards(
            round_data_dir,
            total_games=args.games_per_round,
            processes=args.selfplay_processes,
            seed=args.seed + round_index * 1000,
        )
        write_manifest_list(round_dir / "selfplay_manifest_targets.txt", [str(shard.manifest_path) for shard in shards])

        if args.dry_run:
            _append_line(supervisor_log, f"dry_run round={round_index} shards={len(shards)}")
            new_manifests = [str(shard.manifest_path) for shard in shards]
        else:
            new_manifests = run_parallel_selfplay(
                c4zero_bin=args.c4zero_bin,
                model_path=current_checkpoint / "inference.ts",
                device=args.selfplay_device,
                simulations=args.selfplay_simulations,
                game_workers=args.selfplay_game_workers,
                search_threads=args.selfplay_search_threads,
                inference_batch_size=args.inference_batch_size,
                inference_max_wait_us=args.inference_max_wait_us,
                value_mode=args.value_mode,
                virtual_loss=args.virtual_loss,
                shards=shards,
                supervisor_log=supervisor_log,
            )

        manifests.extend(new_manifests)
        write_manifest_list(round_dir / "train_manifests.txt", manifests)
        write_manifest_list(args.run_root / "manifests_so_far.txt", manifests)

        _set_status(args.run_root, f"round={round_index}/{args.first_round + args.rounds - 1} stage=train current_round={current_round}")
        train_command = _train_command(args, manifests, current_checkpoint, round_dir)
        if args.dry_run:
            (round_dir / "train.log").write_text(f"$ {_quote(train_command)}\n", encoding="utf-8")
        else:
            run_command(train_command, round_dir / "train.log", supervisor_log)

        candidate_checkpoint = round_dir / "checkpoint"
        metrics = _read_checkpoint_metrics(candidate_checkpoint)
        _csv_append(
            args.run_root / "train_summary.csv",
            "round,checkpoint,learning_rate,replay_sampling,recent_games,recent_fraction,symmetry_mode,base_batch_size,effective_batch_size,steps,last_loss,last_policy_loss,last_value_loss",
            ",".join(
                [
                    str(round_index),
                    str(candidate_checkpoint),
                    str(args.learning_rate),
                    args.replay_sampling,
                    str(args.recent_games),
                    str(args.recent_fraction),
                    args.symmetry_mode,
                    str(metrics.get("base_batch_size", args.batch_size)),
                    str(metrics.get("effective_batch_size", args.batch_size * (8 if args.symmetry_mode == "orbit" else 1))),
                    str(args.train_steps),
                    str(metrics.get("last_loss", "")),
                    str(metrics.get("last_policy_loss", "")),
                    str(metrics.get("last_value_loss", "")),
                ]
            ),
        )

        _set_status(args.run_root, f"round={round_index}/{args.first_round + args.rounds - 1} stage=arena current_round={current_round}")
        if args.dry_run:
            arena_summary = {"model_a_score_rate": "0.0", "model_a_wins": "0", "model_b_wins": "0", "draws": "0"}
        else:
            arena_summary = run_arena(
                c4zero_bin=args.c4zero_bin,
                model_a=candidate_checkpoint / "inference.ts",
                model_b=current_checkpoint / "inference.ts",
                bot_b=None,
                device=args.arena_device,
                games=args.arena_games,
                simulations=args.arena_simulations,
                search_threads=args.arena_search_threads,
                arena_workers=args.arena_workers,
                opening_count=args.opening_count,
                opening_plies=args.opening_plies,
                games_per_opening=args.games_per_opening,
                promotion_threshold=args.promotion_threshold,
                seed=args.seed + round_index * 1000 + 101,
                log_path=round_dir / "arena_vs_current.txt",
                supervisor_log=supervisor_log,
            )
        score = float(arena_summary["model_a_score_rate"])
        promoted = score >= args.promotion_threshold
        _csv_append(
            args.run_root / "promotion_log.csv",
            "round,current_round,current_checkpoint,candidate_checkpoint,selfplay_manifests,arena_score,model_a_wins,model_b_wins,draws,promoted",
            ",".join(
                [
                    str(round_index),
                    str(current_round),
                    str(current_checkpoint),
                    str(candidate_checkpoint),
                    ";".join(new_manifests),
                    str(score),
                    arena_summary.get("model_a_wins", ""),
                    arena_summary.get("model_b_wins", ""),
                    arena_summary.get("draws", ""),
                    "1" if promoted else "0",
                ]
            ),
        )

        _set_status(args.run_root, f"round={round_index}/{args.first_round + args.rounds - 1} stage=minimax current_round={current_round}")
        for offset, bot in enumerate(args.minimax_bot):
            if args.dry_run:
                continue
            summary = run_arena(
                c4zero_bin=args.c4zero_bin,
                model_a=candidate_checkpoint / "inference.ts",
                model_b=None,
                bot_b=bot,
                device=args.arena_device,
                games=args.arena_games,
                simulations=args.arena_simulations,
                search_threads=args.arena_search_threads,
                arena_workers=args.arena_workers,
                opening_count=args.opening_count,
                opening_plies=args.opening_plies,
                games_per_opening=args.games_per_opening,
                promotion_threshold=args.promotion_threshold,
                seed=args.seed + round_index * 1000 + 200 + offset,
                log_path=round_dir / f"arena_vs_{bot}_candidate.txt",
                supervisor_log=supervisor_log,
            )
            _csv_append(
                args.run_root / "minimax_eval.csv",
                "round,subject,checkpoint,bot,games,score,model_wins,bot_wins,draws,avg_plies,root_noise,arena_workers,opening_count,opening_plies,games_per_opening",
                ",".join(
                    [
                        str(round_index),
                        "candidate",
                        str(candidate_checkpoint),
                        bot,
                        summary.get("games", ""),
                        summary.get("model_a_score_rate", ""),
                        summary.get("model_a_wins", ""),
                        summary.get("model_b_wins", ""),
                        summary.get("draws", ""),
                        summary.get("avg_plies", ""),
                        summary.get("root_noise", ""),
                        summary.get("arena_workers", ""),
                        summary.get("opening_count", ""),
                        summary.get("opening_plies", ""),
                        summary.get("games_per_opening", ""),
                    ]
                ),
            )

        if promoted:
            current_checkpoint = candidate_checkpoint
            current_round = round_index
        _append_line(
            supervisor_log,
            f"round_done round={round_index} promoted={int(promoted)} current_round={current_round} score={score}",
        )

    _set_status(args.run_root, f"done last_round={args.first_round + args.rounds - 1} current_round={current_round}")
    return 0


if __name__ == "__main__":
    raise SystemExit(loop_main())
