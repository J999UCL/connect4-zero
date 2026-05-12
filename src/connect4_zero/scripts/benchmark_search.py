"""Benchmark the batched deep MCTS search path."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.scripts._common import (
    configure_logging,
    format_seconds,
    log_config,
    log_cuda_memory,
    log_environment,
    resolve_device,
    sync_if_cuda,
)
from connect4_zero.search import BatchedTreeMCTS, TreeMCTSConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark batched deep MCTS throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--batch-size", type=int, default=128, help="Root positions per search call.")
    parser.add_argument("--iterations", type=int, default=10, help="Measured iterations.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations.")
    parser.add_argument("--simulations-per-root", type=int, default=128, help="Full MCTS simulations per root.")
    parser.add_argument("--max-leaf-batch-size", type=int, default=4096, help="Selected leaves evaluated per rollout batch.")
    parser.add_argument("--rollouts-per-leaf", type=int, default=32, help="Random continuations per evaluated leaf.")
    parser.add_argument("--max-rollouts-per-chunk", type=int, default=262144, help="Largest rollout batch inside evaluator.")
    parser.add_argument("--exploration-constant", type=float, default=1.4, help="UCB exploration constant.")
    parser.add_argument("--virtual-loss", type=float, default=1.0, help="Virtual loss used while collecting leaf batches.")
    parser.add_argument("--policy-temperature", type=float, default=1.0, help="Visit-count policy temperature.")
    parser.add_argument("--max-plies", type=int, default=64, help="Safety cap for rollout plies.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/benchmarks"), help="Directory for benchmark logs.")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging; file log remains detailed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = build_parser().parse_args(argv)
    _validate_args(args)

    logger = configure_logging(args.log_dir, name="benchmark_search", verbose=not args.quiet)
    device = resolve_device(args.device)
    log_environment(logger, device)
    log_config(logger, "batched search benchmark config", vars(args))

    config = TreeMCTSConfig(
        simulations_per_root=args.simulations_per_root,
        max_leaf_batch_size=args.max_leaf_batch_size,
        rollouts_per_leaf=args.rollouts_per_leaf,
        exploration_constant=args.exploration_constant,
        virtual_loss=args.virtual_loss,
        policy_temperature=args.policy_temperature,
        rollout_device=device,
        seed=args.seed,
        max_rollout_steps=args.max_plies,
        max_rollouts_per_chunk=args.max_rollouts_per_chunk,
    )
    search = BatchedTreeMCTS(config)
    root_device = torch.device("cpu") if device.type in ("cuda", "mps") else device
    logger.info("search.tree_device=%s", root_device)
    logger.info("search.rollout_device=%s", device)
    roots = Connect4x4x4Batch(batch_size=args.batch_size, device=root_device)

    logger.info("warmup_start iterations=%s", args.warmup)
    for index in range(args.warmup):
        started_at = time.perf_counter()
        result = search.search_batch(roots)
        sync_if_cuda(device)
        duration = time.perf_counter() - started_at
        logger.info(
            "warmup_iteration=%s duration=%s visits=%s leaf_evals=%s terminal_evals=%s "
            "leaf_batches=%s max_leaf_batch=%s policy_mean_sum=%.6f",
            index,
            format_seconds(duration),
            int(result.visit_counts.sum().detach().cpu().item()),
            search.last_leaf_evaluations,
            search.last_terminal_evaluations,
            len(search.last_leaf_batch_sizes),
            max(search.last_leaf_batch_sizes) if search.last_leaf_batch_sizes else 0,
            result.policy.sum(dim=1).mean().detach().cpu().item(),
        )
        log_cuda_memory(logger, prefix="cuda.warmup")

    logger.info("benchmark_start iterations=%s", args.iterations)
    durations: list[float] = []
    total_visits = 0
    total_roots = 0
    total_rollouts = 0
    benchmark_started_at = time.perf_counter()

    for index in range(args.iterations):
        started_at = time.perf_counter()
        result = search.search_batch(roots)
        sync_if_cuda(device)
        duration = time.perf_counter() - started_at
        visits = int(result.visit_counts.sum().detach().cpu().item())
        rollouts = visits * args.rollouts_per_leaf
        durations.append(duration)
        total_visits += visits
        total_roots += args.batch_size
        total_rollouts += rollouts
        logger.info(
            "iteration=%s duration=%s roots=%s visits=%s leaf_evals=%s terminal_evals=%s "
            "leaf_batches=%s max_leaf_batch=%s rollout_games_est=%s roots_per_sec=%.2f "
            "visits_per_sec=%.1f rollout_games_per_sec_est=%.1f root_value_mean=%.4f",
            index,
            format_seconds(duration),
            args.batch_size,
            visits,
            search.last_leaf_evaluations,
            search.last_terminal_evaluations,
            len(search.last_leaf_batch_sizes),
            max(search.last_leaf_batch_sizes) if search.last_leaf_batch_sizes else 0,
            rollouts,
            args.batch_size / duration if duration > 0 else 0.0,
            visits / duration if duration > 0 else 0.0,
            rollouts / duration if duration > 0 else 0.0,
            result.root_values.mean().detach().cpu().item(),
        )
        log_cuda_memory(logger, prefix="cuda.iteration")

    elapsed = time.perf_counter() - benchmark_started_at
    mean_duration = sum(durations) / len(durations)
    logger.info("benchmark_complete elapsed=%s", format_seconds(elapsed))
    logger.info("summary.iterations=%s", args.iterations)
    logger.info("summary.mean_iteration_duration=%s", format_seconds(mean_duration))
    logger.info("summary.total_roots=%s", total_roots)
    logger.info("summary.total_visits=%s", total_visits)
    logger.info("summary.total_rollout_games_est=%s", total_rollouts)
    logger.info("summary.roots_per_sec=%.2f", total_roots / elapsed if elapsed > 0 else 0.0)
    logger.info("summary.visits_per_sec=%.1f", total_visits / elapsed if elapsed > 0 else 0.0)
    logger.info("summary.rollout_games_per_sec_est=%.1f", total_rollouts / elapsed if elapsed > 0 else 0.0)
    log_cuda_memory(logger, prefix="cuda.final")
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.simulations_per_root <= 0:
        raise ValueError("--simulations-per-root must be positive")
    if args.max_leaf_batch_size <= 0:
        raise ValueError("--max-leaf-batch-size must be positive")
    if args.rollouts_per_leaf <= 0:
        raise ValueError("--rollouts-per-leaf must be positive")


if __name__ == "__main__":
    raise SystemExit(main())
