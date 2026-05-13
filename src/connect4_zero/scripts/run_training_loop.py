"""Run repeated self-play, training, and arena evaluation rounds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from connect4_zero.scripts._common import configure_logging, log_config, log_environment, resolve_device, timestamp_slug


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repeated AlphaZero self-play -> train -> arena rounds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--initial-checkpoint", type=Path, required=True, help="Checkpoint used to start round 1.")
    parser.add_argument("--run-root", type=Path, required=True, help="Directory for loop logs and train/eval artifacts.")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory for generated self-play data.")
    parser.add_argument("--rounds", type=int, default=4, help="Number of training rounds.")
    parser.add_argument("--run-prefix", default=None, help="Optional run id prefix. Defaults to timestamped loop id.")
    parser.add_argument("--device", default="auto", help="Device passed to child scripts.")
    parser.add_argument("--games-per-round", type=int, default=4000, help="Self-play games per round.")
    parser.add_argument("--selfplay-batch-size", type=int, default=128, help="Self-play parallel games per worker.")
    parser.add_argument("--games-per-write", type=int, default=256, help="Self-play games per shard write.")
    parser.add_argument("--samples-per-shard", type=int, default=16384, help="Self-play samples per shard.")
    parser.add_argument("--selfplay-workers", type=str, default="12", help="Self-play worker count.")
    parser.add_argument("--worker-start-method", choices=("spawn", "forkserver", "fork"), default="spawn")
    parser.add_argument("--torch-threads-per-worker", type=int, default=1)
    parser.add_argument("--simulations-per-root", type=int, default=128, help="Self-play PUCT simulations per root.")
    parser.add_argument("--max-leaf-batch-size", type=int, default=256, help="Self-play PUCT leaf batch size.")
    parser.add_argument("--puct-inference-batch-size", type=int, default=4096, help="Self-play inference chunk size.")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--action-temperature", type=float, default=1.0)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--max-plies", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None, help="Base seed; round index is added.")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--train-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--arena-games", type=int, default=128)
    parser.add_argument("--arena-batch-size", type=int, default=32)
    parser.add_argument("--arena-simulations-per-root", type=int, default=64)
    parser.add_argument("--arena-max-leaf-batch-size", type=int, default=128)
    parser.add_argument("--arena-inference-batch-size", type=int, default=4096)
    parser.add_argument("--force", action="store_true", help="Overwrite per-round self-play data dirs.")
    parser.add_argument("--dry-run", action="store_true", help="Write the command plan but do not run child scripts.")
    parser.add_argument("--quiet", action="store_true", help="Reduce loop stdout logging; child scripts remain verbose.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = build_parser().parse_args(argv)
    _validate_args(args)
    run_id = args.run_prefix or f"puct-loop-{timestamp_slug()}"
    loop_run_root = args.run_root / run_id
    loop_data_root = args.data_root / run_id
    loop_run_root.mkdir(parents=True, exist_ok=True)
    loop_data_root.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(loop_run_root / "logs", name="run_training_loop", verbose=not args.quiet)
    device = resolve_device(args.device)
    log_environment(logger, device)
    log_config(logger, "training loop config", vars(args) | {"resolved_run_id": run_id})

    current_checkpoint = args.initial_checkpoint
    round_summaries: list[dict[str, object]] = []
    plan: list[dict[str, object]] = []

    for round_index in range(1, args.rounds + 1):
        round_id = f"round-{round_index:02d}"
        round_run = loop_run_root / round_id
        round_data = loop_data_root / round_id / "selfplay"
        train_out = round_run / "train"
        arena_out = round_run / "arena"
        round_seed = None if args.seed is None else args.seed + round_index

        generate_cmd = _generate_cmd(args, current_checkpoint, round_data, round_seed)
        train_cmd = _train_cmd(args, current_checkpoint, round_data / "manifest.jsonl", train_out)
        arena_cmd_template = _arena_cmd_template(args, current_checkpoint, arena_out, round_seed)
        plan.append(
            {
                "round": round_index,
                "baseline_checkpoint": str(current_checkpoint),
                "selfplay": generate_cmd,
                "train": train_cmd,
                "arena_template": arena_cmd_template,
            }
        )

        logger.info("round_start round=%s baseline_checkpoint=%s", round_index, current_checkpoint)
        if args.dry_run:
            continue

        _run_command(generate_cmd, logger, label=f"{round_id}.selfplay")
        _run_command(train_cmd, logger, label=f"{round_id}.train")
        next_checkpoint = _latest_checkpoint(train_out)
        arena_cmd = arena_cmd_template + ["--candidate-checkpoint", str(next_checkpoint)]
        _run_command(arena_cmd, logger, label=f"{round_id}.arena")

        arena_summary_path = arena_out / "arena-summary.json"
        round_summary = {
            "round": round_index,
            "baseline_checkpoint": str(current_checkpoint),
            "candidate_checkpoint": str(next_checkpoint),
            "selfplay_manifest": str(round_data / "manifest.jsonl"),
            "train_out": str(train_out),
            "arena_summary": str(arena_summary_path),
        }
        if arena_summary_path.exists():
            round_summary["arena"] = json.loads(arena_summary_path.read_text(encoding="utf-8"))
        round_summaries.append(round_summary)
        _write_json(loop_run_root / "loop-summary.json", {"run_id": run_id, "rounds": round_summaries})
        logger.info("round_done round=%s candidate_checkpoint=%s arena_summary=%s", round_index, next_checkpoint, arena_summary_path)
        current_checkpoint = next_checkpoint

    if args.dry_run:
        _write_json(loop_run_root / "loop-plan.json", {"run_id": run_id, "rounds": plan})
        logger.info("dry_run=true plan=%s", loop_run_root / "loop-plan.json")
    else:
        _write_json(loop_run_root / "loop-summary.json", {"run_id": run_id, "rounds": round_summaries})
        logger.info("loop_done run_id=%s rounds=%s summary=%s", run_id, args.rounds, loop_run_root / "loop-summary.json")
    return 0


def _generate_cmd(args: argparse.Namespace, checkpoint: Path, out: Path, seed: int | None) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "connect4_zero.scripts.generate_selfplay",
        "--backend",
        "puct",
        "--device",
        args.device,
        "--checkpoint",
        str(checkpoint),
        "--games",
        str(args.games_per_round),
        "--batch-size",
        str(args.selfplay_batch_size),
        "--games-per-write",
        str(args.games_per_write),
        "--samples-per-shard",
        str(args.samples_per_shard),
        "--simulations-per-root",
        str(args.simulations_per_root),
        "--max-leaf-batch-size",
        str(args.max_leaf_batch_size),
        "--puct-inference-batch-size",
        str(args.puct_inference_batch_size),
        "--puct-inference-mode",
        "worker",
        "--c-puct",
        str(args.c_puct),
        "--add-root-noise",
        "--root-dirichlet-alpha",
        str(args.root_dirichlet_alpha),
        "--root-exploration-fraction",
        str(args.root_exploration_fraction),
        "--action-temperature",
        str(args.action_temperature),
        "--policy-temperature",
        str(args.policy_temperature),
        "--max-plies",
        str(args.max_plies),
        "--num-workers",
        str(args.selfplay_workers),
        "--worker-start-method",
        args.worker_start_method,
        "--torch-threads-per-worker",
        str(args.torch_threads_per_worker),
        "--out",
        str(out),
        "--force",
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if not args.force:
        cmd.remove("--force")
    return cmd


def _train_cmd(args: argparse.Namespace, checkpoint: Path, manifest: Path, out: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "connect4_zero.scripts.train_resnet",
        "--device",
        args.device,
        "--resume",
        str(checkpoint),
        "--manifest",
        str(manifest),
        "--out",
        str(out),
        "--epochs",
        str(args.train_epochs),
        "--max-steps",
        str(args.train_steps),
        "--batch-size",
        str(args.train_batch_size),
        "--num-workers",
        str(args.train_workers),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--apply-symmetries",
    ]


def _arena_cmd_template(
    args: argparse.Namespace,
    baseline_checkpoint: Path,
    out: Path,
    seed: int | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "connect4_zero.scripts.arena_eval",
        "--device",
        args.device,
        "--baseline-checkpoint",
        str(baseline_checkpoint),
        "--games",
        str(args.arena_games),
        "--batch-size",
        str(args.arena_batch_size),
        "--simulations-per-root",
        str(args.arena_simulations_per_root),
        "--max-leaf-batch-size",
        str(args.arena_max_leaf_batch_size),
        "--inference-batch-size",
        str(args.arena_inference_batch_size),
        "--max-plies",
        str(args.max_plies),
        "--out",
        str(out),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


def _run_command(command: list[str], logger, label: str) -> None:
    logger.info("command_start label=%s command=%s", label, _quote_command(command))
    result = subprocess.run(command, check=False)
    logger.info("command_done label=%s returncode=%s", label, result.returncode)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with return code {result.returncode}")


def _latest_checkpoint(train_out: Path) -> Path:
    checkpoints = sorted((train_out / "checkpoints").glob("checkpoint-step-*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found under {train_out / 'checkpoints'}")
    return checkpoints[-1]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _quote_command(command: list[str]) -> str:
    return " ".join(_shell_quote(part) for part in command)


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "._/:=-" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _validate_args(args: argparse.Namespace) -> None:
    if not args.initial_checkpoint.exists():
        raise FileNotFoundError(f"--initial-checkpoint not found: {args.initial_checkpoint}")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.games_per_round <= 0:
        raise ValueError("--games-per-round must be positive")
    if args.selfplay_batch_size <= 0:
        raise ValueError("--selfplay-batch-size must be positive")
    if args.train_steps <= 0:
        raise ValueError("--train-steps must be positive")
    if args.arena_games <= 0:
        raise ValueError("--arena-games must be positive")
    if args.arena_batch_size <= 0:
        raise ValueError("--arena-batch-size must be positive")


if __name__ == "__main__":
    raise SystemExit(main())
