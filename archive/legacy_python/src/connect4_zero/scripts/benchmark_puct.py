"""Benchmark the AlphaZero PUCT search path with a ResNet evaluator."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.model import Connect4ResNet3D, ResNet3DConfig, count_parameters, load_checkpoint
from connect4_zero.scripts._common import (
    configure_logging,
    format_seconds,
    log_config,
    log_cuda_memory,
    log_environment,
    resolve_device,
    sync_if_cuda,
)
from connect4_zero.search import BatchedPUCTMCTS, NeuralPolicyValueEvaluator, PUCTMCTSConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark batched PUCT MCTS with neural policy/value evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="auto", help="Model device: auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--batch-size", type=int, default=32, help="Root positions per search call.")
    parser.add_argument("--iterations", type=int, default=5, help="Measured iterations.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations.")
    parser.add_argument("--simulations-per-root", type=int, default=128, help="PUCT simulations per root.")
    parser.add_argument("--max-leaf-batch-size", type=int, default=256, help="Selected leaves per neural batch.")
    parser.add_argument("--inference-batch-size", type=int, default=4096, help="Model inference chunk size.")
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant.")
    parser.add_argument("--policy-temperature", type=float, default=1.0, help="Visit-count policy temperature.")
    parser.add_argument("--max-selection-depth", type=int, default=64, help="Safety cap for tree selection depth.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional ResNet checkpoint.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for root noise.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/benchmarks"), help="Directory for logs.")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging; file log remains detailed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = build_parser().parse_args(argv)
    _validate_args(args)
    logger = configure_logging(args.log_dir, name="benchmark_puct", verbose=not args.quiet)
    device = resolve_device(args.device)
    log_environment(logger, device)
    log_config(logger, "PUCT benchmark config", vars(args))

    model = _load_or_create_model(args.checkpoint, device)
    logger.info("model.parameters=%s", count_parameters(model))
    evaluator = NeuralPolicyValueEvaluator(model, device=device, inference_batch_size=args.inference_batch_size)
    search = BatchedPUCTMCTS(
        evaluator=evaluator,
        config=PUCTMCTSConfig(
            simulations_per_root=args.simulations_per_root,
            max_leaf_batch_size=args.max_leaf_batch_size,
            c_puct=args.c_puct,
            policy_temperature=args.policy_temperature,
            max_selection_depth=args.max_selection_depth,
            seed=args.seed,
        ),
    )
    tree_device = torch.device("cpu") if device.type in ("cuda", "mps") else device
    roots = Connect4x4x4Batch(args.batch_size, device=tree_device)

    logger.info("warmup_start iterations=%s", args.warmup)
    for index in range(args.warmup):
        started_at = time.perf_counter()
        result = search.search_batch(roots)
        sync_if_cuda(device)
        duration = time.perf_counter() - started_at
        logger.info(
            "warmup_iteration=%s duration=%s visits=%s leaf_evals=%s max_depth=%s policy_mean_sum=%.6f",
            index,
            format_seconds(duration),
            int(result.visit_counts.sum().detach().cpu().item()),
            search.last_leaf_evaluations,
            max(tree.max_depth for tree in search.last_trees),
            result.policy.sum(dim=1).mean().detach().cpu().item(),
        )
        log_cuda_memory(logger, prefix="cuda.warmup")

    durations: list[float] = []
    total_roots = 0
    total_visits = 0
    started_at = time.perf_counter()
    logger.info("benchmark_start iterations=%s", args.iterations)
    for index in range(args.iterations):
        iteration_started_at = time.perf_counter()
        result = search.search_batch(roots)
        sync_if_cuda(device)
        duration = time.perf_counter() - iteration_started_at
        visits = int(result.visit_counts.sum().detach().cpu().item())
        total_roots += args.batch_size
        total_visits += visits
        durations.append(duration)
        logger.info(
            "iteration=%s duration=%s roots=%s visits=%s leaf_evals=%s terminal_evals=%s "
            "leaf_batches=%s max_leaf_batch=%s expanded_children=%s max_depth=%s "
            "roots_per_sec=%.2f visits_per_sec=%.1f timing_prepare=%.4f timing_select=%.4f "
            "timing_expand=%.4f timing_leaf_eval=%.4f timing_backprop=%.4f timing_build=%.4f",
            index,
            format_seconds(duration),
            args.batch_size,
            visits,
            search.last_leaf_evaluations,
            search.last_terminal_evaluations,
            len(search.last_leaf_batch_sizes),
            max(search.last_leaf_batch_sizes) if search.last_leaf_batch_sizes else 0,
            search.last_expanded_children,
            max(tree.max_depth for tree in search.last_trees),
            args.batch_size / duration if duration > 0 else 0.0,
            visits / duration if duration > 0 else 0.0,
            search.last_timing_seconds["prepare_trees"],
            search.last_timing_seconds["select"],
            search.last_timing_seconds["expand"],
            search.last_timing_seconds["leaf_eval"],
            search.last_timing_seconds["backprop"],
            search.last_timing_seconds["build_result"],
        )
        log_cuda_memory(logger, prefix="cuda.iteration")

    elapsed = time.perf_counter() - started_at
    mean_duration = sum(durations) / len(durations)
    logger.info("benchmark_complete elapsed=%s", format_seconds(elapsed))
    logger.info("summary.iterations=%s", args.iterations)
    logger.info("summary.mean_iteration_duration=%s", format_seconds(mean_duration))
    logger.info("summary.total_roots=%s", total_roots)
    logger.info("summary.total_visits=%s", total_visits)
    logger.info("summary.roots_per_sec=%.2f", total_roots / elapsed if elapsed > 0 else 0.0)
    logger.info("summary.visits_per_sec=%.1f", total_visits / elapsed if elapsed > 0 else 0.0)
    log_cuda_memory(logger, prefix="cuda.final")
    return 0


def _load_or_create_model(checkpoint: Path | None, device: torch.device) -> Connect4ResNet3D:
    if checkpoint is None:
        model = Connect4ResNet3D(ResNet3DConfig())
    else:
        model = load_checkpoint(checkpoint, map_location=device).model
    model.eval()
    return model.to(device)


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
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be positive")


if __name__ == "__main__":
    raise SystemExit(main())
