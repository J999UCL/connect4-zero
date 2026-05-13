"""Generate AlphaZero-style self-play shards with loud progress logging."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from connect4_zero.data import SelfPlayConfig, SelfPlayDataset, SelfPlayGenerator, SelfPlayShardWriter
from connect4_zero.model import Connect4ResNet3D, ResNet3DConfig, load_checkpoint
from connect4_zero.scripts._common import (
    SelfPlayProgressLogger,
    configure_logging,
    directory_size_bytes,
    format_seconds,
    git_branch,
    git_commit,
    human_bytes,
    log_config,
    log_cuda_memory,
    log_environment,
    maybe_empty_output_dir,
    resolve_device,
    sync_if_cuda,
    timestamp_slug,
)
from connect4_zero.search import BatchedPUCTMCTS, BatchedTreeMCTS, NeuralPolicyValueEvaluator, PUCTMCTSConfig, TreeMCTSConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate safetensor self-play shards using batched deep MCTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--games", type=int, required=True, help="Number of complete games to generate.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for manifest, shards, and logs.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--batch-size", type=int, default=128, help="Parallel games per self-play batch.")
    parser.add_argument("--games-per-write", type=int, default=512, help="Complete games generated before each write.")
    parser.add_argument("--samples-per-shard", type=int, default=16384, help="Samples per safetensor shard.")
    parser.add_argument("--backend", choices=("rollout", "puct"), default="rollout", help="Search backend.")
    parser.add_argument("--simulations-per-root", type=int, default=128, help="Full MCTS simulations per root.")
    parser.add_argument("--max-leaf-batch-size", type=int, default=4096, help="Selected leaves evaluated per rollout batch.")
    parser.add_argument("--rollouts-per-leaf", type=int, default=32, help="Random continuations per evaluated leaf.")
    parser.add_argument("--max-rollouts-per-chunk", type=int, default=262144, help="Largest rollout batch inside evaluator.")
    parser.add_argument("--exploration-constant", type=float, default=1.4, help="UCB exploration constant.")
    parser.add_argument("--virtual-loss", type=float, default=1.0, help="Virtual loss used while collecting leaf batches.")
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant for neural search.")
    parser.add_argument("--puct-inference-batch-size", type=int, default=4096, help="Model inference chunk size for PUCT.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional ResNet checkpoint for PUCT search.")
    parser.add_argument("--add-root-noise", action="store_true", help="Add AlphaZero Dirichlet root noise for PUCT self-play.")
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3, help="PUCT root Dirichlet alpha.")
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25, help="PUCT root noise mixture fraction.")
    parser.add_argument("--policy-temperature", type=float, default=1.0, help="Visit-count policy temperature.")
    parser.add_argument("--action-temperature", type=float, default=1.0, help="Self-play action sampling temperature.")
    parser.add_argument("--max-plies", type=int, default=64, help="Safety cap for plies per game.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for rollouts and action sampling.")
    parser.add_argument("--append", action="store_true", help="Append shards to an existing output directory.")
    parser.add_argument("--force", action="store_true", help="Delete existing output directory before writing.")
    parser.add_argument("--no-verify-load", action="store_true", help="Skip loading the generated manifest at the end.")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging; file log remains detailed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = build_parser().parse_args(argv)
    _validate_args(args)

    maybe_empty_output_dir(args.out, append=args.append, force=args.force)
    args.out.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.out / "logs", name="generate_selfplay", verbose=not args.quiet)
    started_at = time.perf_counter()

    device = resolve_device(args.device)
    log_environment(logger, device)
    log_config(logger, "self-play generation config", vars(args))
    tree_device = torch.device("cpu") if device.type in ("cuda", "mps") else device
    logger.info("search.backend=%s", args.backend)
    logger.info("search.tree_device=%s", tree_device)
    logger.info("search.evaluator_device=%s", device)

    search = _build_search(args, device)
    self_play_config = SelfPlayConfig(
        batch_size=args.batch_size,
        device=tree_device,
        action_temperature=args.action_temperature,
        max_plies=args.max_plies,
        seed=args.seed,
    )
    generator = SelfPlayGenerator(search=search, config=self_play_config)
    writer = SelfPlayShardWriter(
        output_dir=args.out,
        samples_per_shard=args.samples_per_shard,
        metadata=_metadata(args, device, tree_device),
    )

    logger.info("generation_start games=%s output=%s", args.games, args.out)
    total_samples = 0
    games_completed = 0
    chunk_index = 0
    progress_logger = SelfPlayProgressLogger(logger, device=device)

    while games_completed < args.games:
        chunk_games = min(args.games_per_write, args.games - games_completed)
        logger.info(
            "chunk_start index=%s chunk_games=%s games_completed=%s/%s",
            chunk_index,
            chunk_games,
            games_completed,
            args.games,
        )
        chunk_started_at = time.perf_counter()
        samples = generator.generate_with_progress(chunk_games, progress_callback=progress_logger)
        sync_if_cuda(device)
        generation_seconds = time.perf_counter() - chunk_started_at
        logger.info(
            "chunk_generated index=%s samples=%s generation_duration=%s samples_per_sec=%.1f",
            chunk_index,
            samples.num_samples,
            format_seconds(generation_seconds),
            samples.num_samples / generation_seconds if generation_seconds > 0 else 0.0,
        )

        write_started_at = time.perf_counter()
        writer.write(samples)
        write_seconds = time.perf_counter() - write_started_at
        total_samples += samples.num_samples
        games_completed += chunk_games
        logger.info(
            "chunk_written index=%s write_duration=%s games_completed=%s/%s total_samples=%s output_size=%s",
            chunk_index,
            format_seconds(write_seconds),
            games_completed,
            args.games,
            total_samples,
            human_bytes(directory_size_bytes(args.out)),
        )
        log_cuda_memory(logger, prefix="cuda.after_chunk_write")
        del samples
        if device.type == "cuda":
            torch.cuda.empty_cache()
        chunk_index += 1

    elapsed = time.perf_counter() - started_at
    logger.info("generation_complete elapsed=%s games=%s samples=%s", format_seconds(elapsed), args.games, total_samples)
    logger.info("throughput.games_per_sec=%.3f", args.games / elapsed if elapsed > 0 else 0.0)
    logger.info("throughput.samples_per_sec=%.1f", total_samples / elapsed if elapsed > 0 else 0.0)
    logger.info("manifest=%s", args.out / "manifest.jsonl")
    logger.info("output_size=%s", human_bytes(directory_size_bytes(args.out)))

    if not args.no_verify_load:
        _verify_dataset(args.out / "manifest.jsonl", logger)

    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.games <= 0:
        raise ValueError("--games must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.games_per_write <= 0:
        raise ValueError("--games-per-write must be positive")
    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be positive")
    if args.simulations_per_root <= 0:
        raise ValueError("--simulations-per-root must be positive")
    if args.max_leaf_batch_size <= 0:
        raise ValueError("--max-leaf-batch-size must be positive")
    if args.rollouts_per_leaf <= 0:
        raise ValueError("--rollouts-per-leaf must be positive")
    if args.puct_inference_batch_size <= 0:
        raise ValueError("--puct-inference-batch-size must be positive")
    if args.force and args.append:
        raise ValueError("--force and --append are mutually exclusive")


def _build_search(args: argparse.Namespace, device: torch.device):
    if args.backend == "rollout":
        search_config = TreeMCTSConfig(
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
        return BatchedTreeMCTS(search_config)

    model = _load_or_create_model(args.checkpoint, device)
    evaluator = NeuralPolicyValueEvaluator(
        model=model,
        device=device,
        inference_batch_size=args.puct_inference_batch_size,
    )
    search_config = PUCTMCTSConfig(
        simulations_per_root=args.simulations_per_root,
        max_leaf_batch_size=args.max_leaf_batch_size,
        c_puct=args.c_puct,
        policy_temperature=args.policy_temperature,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
        add_root_noise=args.add_root_noise,
        max_selection_depth=args.max_plies,
        seed=args.seed,
    )
    return BatchedPUCTMCTS(evaluator=evaluator, config=search_config)


def _load_or_create_model(checkpoint: Path | None, device: torch.device) -> Connect4ResNet3D:
    if checkpoint is None:
        model = Connect4ResNet3D(ResNet3DConfig())
        model.eval()
        return model.to(device)
    state = load_checkpoint(checkpoint, map_location=device)
    state.model.eval()
    return state.model.to(device)


def _metadata(args: argparse.Namespace, device: torch.device, tree_device: torch.device) -> dict[str, str]:
    values: dict[str, Any] = {
        "format": "connect4_zero.selfplay.v1",
        "created_at_utc": timestamp_slug(),
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "device": str(device),
        "rollout_device": str(device),
        "backend": args.backend,
        "tree_device": str(tree_device),
        "games_requested": args.games,
        "batch_size": args.batch_size,
        "games_per_write": args.games_per_write,
        "simulations_per_root": args.simulations_per_root,
        "max_leaf_batch_size": args.max_leaf_batch_size,
        "rollouts_per_leaf": args.rollouts_per_leaf,
        "max_rollouts_per_chunk": args.max_rollouts_per_chunk,
        "exploration_constant": args.exploration_constant,
        "virtual_loss": args.virtual_loss,
        "c_puct": args.c_puct,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "add_root_noise": args.add_root_noise,
        "root_dirichlet_alpha": args.root_dirichlet_alpha,
        "root_exploration_fraction": args.root_exploration_fraction,
        "policy_temperature": args.policy_temperature,
        "action_temperature": args.action_temperature,
        "max_plies": args.max_plies,
        "seed": args.seed,
    }
    return {key: json.dumps(value) if isinstance(value, (dict, list)) else str(value) for key, value in values.items()}


def _verify_dataset(manifest_path: Path, logger) -> None:
    logger.info("verify_load_start manifest=%s", manifest_path)
    dataset = SelfPlayDataset(manifest_path, apply_symmetries=False)
    first = dataset[0]
    last = dataset[-1]
    logger.info("verify_load_samples=%s", len(dataset))
    logger.info("verify_first input_shape=%s policy_sum=%.6f value=%.1f", tuple(first["input"].shape), first["policy"].sum().item(), first["value"].item())
    logger.info("verify_last input_shape=%s policy_sum=%.6f value=%.1f", tuple(last["input"].shape), last["policy"].sum().item(), last["value"].item())
    logger.info("verify_load_done")


if __name__ == "__main__":
    raise SystemExit(main())
