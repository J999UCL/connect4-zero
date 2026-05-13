"""CLI for checkpoint-vs-checkpoint arena evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from connect4_zero.eval import ArenaConfig, evaluate_arena
from connect4_zero.scripts._common import configure_logging, log_config, log_environment, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate two ResNet checkpoints by deterministic PUCT arena play.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-checkpoint", type=Path, required=True, help="Checkpoint being evaluated.")
    parser.add_argument("--baseline-checkpoint", type=Path, required=True, help="Checkpoint to compare against.")
    parser.add_argument("--games", type=int, default=128, help="Arena games.")
    parser.add_argument("--batch-size", type=int, default=32, help="Parallel arena games.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--simulations-per-root", type=int, default=64, help="PUCT simulations per move.")
    parser.add_argument("--max-leaf-batch-size", type=int, default=128, help="PUCT leaf batch size.")
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant.")
    parser.add_argument("--policy-temperature", type=float, default=1.0, help="Visit policy temperature.")
    parser.add_argument("--inference-batch-size", type=int, default=4096, help="Neural inference chunk size.")
    parser.add_argument("--max-plies", type=int, default=64, help="Safety cap for plies per game.")
    parser.add_argument("--seed", type=int, default=None, help="Optional torch seed.")
    parser.add_argument("--no-alternate-starts", action="store_true", help="Let candidate start every game.")
    parser.add_argument("--out", type=Path, default=Path("runs/arena"), help="Output directory.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional explicit summary JSON path.")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging; file log remains detailed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = build_parser().parse_args(argv)
    _validate_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.out / "logs", name="arena_eval", verbose=not args.quiet)
    device = resolve_device(args.device)
    log_environment(logger, device)
    log_config(logger, "arena eval config", vars(args))

    summary = evaluate_arena(
        ArenaConfig(
            candidate_checkpoint=args.candidate_checkpoint,
            baseline_checkpoint=args.baseline_checkpoint,
            games=args.games,
            batch_size=args.batch_size,
            device=device,
            simulations_per_root=args.simulations_per_root,
            max_leaf_batch_size=args.max_leaf_batch_size,
            c_puct=args.c_puct,
            policy_temperature=args.policy_temperature,
            inference_batch_size=args.inference_batch_size,
            max_plies=args.max_plies,
            seed=args.seed,
            alternate_starts=not args.no_alternate_starts,
        ),
        logger=logger,
    )
    json_out = args.json_out if args.json_out is not None else args.out / "arena-summary.json"
    summary.write_json(json_out)
    logger.info("arena_summary=%s", json_out)
    logger.info(
        "arena_result games=%s candidate_wins=%s baseline_wins=%s draws=%s candidate_score_rate=%.4f avg_plies=%.2f games_per_sec=%.3f",
        summary.games,
        summary.candidate_wins,
        summary.baseline_wins,
        summary.draws,
        summary.candidate_score_rate,
        summary.avg_plies,
        summary.games_per_second,
    )
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if not args.candidate_checkpoint.exists():
        raise FileNotFoundError(f"--candidate-checkpoint not found: {args.candidate_checkpoint}")
    if not args.baseline_checkpoint.exists():
        raise FileNotFoundError(f"--baseline-checkpoint not found: {args.baseline_checkpoint}")
    if args.games <= 0:
        raise ValueError("--games must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.simulations_per_root <= 0:
        raise ValueError("--simulations-per-root must be positive")
    if args.max_leaf_batch_size <= 0:
        raise ValueError("--max-leaf-batch-size must be positive")
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be positive")
    if args.max_plies <= 0:
        raise ValueError("--max-plies must be positive")


if __name__ == "__main__":
    raise SystemExit(main())
