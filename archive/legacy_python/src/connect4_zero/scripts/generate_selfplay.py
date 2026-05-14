"""Generate AlphaZero-style self-play shards with loud progress logging."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
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
from connect4_zero.search import (
    BatchedPUCTMCTS,
    BatchedTreeMCTS,
    NeuralPolicyValueEvaluator,
    PUCTMCTSConfig,
    SharedInferenceClientEvaluator,
    TreeMCTSConfig,
    run_policy_value_server,
)


_SHARED_INFERENCE_REQUEST_QUEUE: Any | None = None
_SHARED_INFERENCE_RESPONSE_QUEUES: list[Any] | None = None
_SHARED_INFERENCE_RESPONSE_TIMEOUT_SECONDS = 600.0


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
    parser.add_argument(
        "--puct-inference-mode",
        choices=("auto", "worker", "server"),
        default="auto",
        help="PUCT inference ownership. auto uses one shared server for CUDA multiprocessing.",
    )
    parser.add_argument(
        "--puct-server-max-batch-size",
        type=int,
        default=4096,
        help="Maximum neural states evaluated together by the shared PUCT inference server.",
    )
    parser.add_argument(
        "--puct-server-batch-timeout-ms",
        type=float,
        default=5.0,
        help="Milliseconds the shared inference server waits to coalesce worker requests.",
    )
    parser.add_argument(
        "--puct-server-response-timeout",
        type=float,
        default=600.0,
        help="Seconds a worker waits for a shared inference response before failing.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional ResNet checkpoint for PUCT search.")
    parser.add_argument("--add-root-noise", action="store_true", help="Add AlphaZero Dirichlet root noise for PUCT self-play.")
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3, help="PUCT root Dirichlet alpha.")
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25, help="PUCT root noise mixture fraction.")
    parser.add_argument("--policy-temperature", type=float, default=1.0, help="Visit-count policy temperature.")
    parser.add_argument("--action-temperature", type=float, default=1.0, help="Self-play action sampling temperature.")
    parser.add_argument("--max-plies", type=int, default=64, help="Safety cap for plies per game.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for rollouts and action sampling.")
    parser.add_argument(
        "--num-workers",
        default="1",
        help="Self-play actor processes. Use an integer, or 'auto'/'all' for the maximum safe worker count.",
    )
    parser.add_argument(
        "--worker-start-method",
        choices=("spawn", "forkserver", "fork"),
        default="spawn",
        help="Multiprocessing start method. Use spawn for CUDA runs.",
    )
    parser.add_argument(
        "--torch-threads-per-worker",
        type=int,
        default=1,
        help="torch CPU threads allocated inside each self-play worker.",
    )
    parser.add_argument(
        "--cuda-worker-memory-mib",
        type=int,
        default=768,
        help="Estimated CUDA memory per worker used to cap --num-workers auto.",
    )
    parser.add_argument(
        "--cuda-memory-reserve-mib",
        type=int,
        default=2048,
        help="CUDA memory left unused when resolving --num-workers auto.",
    )
    parser.add_argument(
        "--worker-stdout",
        action="store_true",
        help="Mirror every worker log to stdout. By default worker detail goes to worker log files only.",
    )
    parser.add_argument("--append", action="store_true", help="Append shards to an existing output directory.")
    parser.add_argument("--force", action="store_true", help="Delete existing output directory before writing.")
    parser.add_argument("--no-verify-load", action="store_true", help="Skip loading the generated manifest at the end.")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging; file log remains detailed.")
    return parser


@dataclass(frozen=True)
class GenerationSummary:
    games: int
    samples: int
    elapsed_seconds: float


@dataclass(frozen=True)
class WorkerResult:
    worker_index: int
    games: int
    samples: int
    elapsed_seconds: float
    output_dir: str
    manifest_path: str


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

    num_workers = _resolve_num_workers(args, device, logger)
    if num_workers == 1:
        summary = _generate_serial(args, logger, device, started_at=started_at)
    else:
        summary = _generate_parallel(args, logger, device, num_workers=num_workers, started_at=started_at)

    elapsed = time.perf_counter() - started_at
    logger.info("generation_complete elapsed=%s games=%s samples=%s", format_seconds(elapsed), args.games, summary.samples)
    logger.info("throughput.games_per_sec=%.3f", args.games / elapsed if elapsed > 0 else 0.0)
    logger.info("throughput.samples_per_sec=%.1f", summary.samples / elapsed if elapsed > 0 else 0.0)
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
    if args.puct_server_max_batch_size <= 0:
        raise ValueError("--puct-server-max-batch-size must be positive")
    if args.puct_server_batch_timeout_ms < 0:
        raise ValueError("--puct-server-batch-timeout-ms must be non-negative")
    if args.puct_server_response_timeout <= 0:
        raise ValueError("--puct-server-response-timeout must be positive")
    if args.torch_threads_per_worker <= 0:
        raise ValueError("--torch-threads-per-worker must be positive")
    if args.cuda_worker_memory_mib <= 0:
        raise ValueError("--cuda-worker-memory-mib must be positive")
    if args.cuda_memory_reserve_mib < 0:
        raise ValueError("--cuda-memory-reserve-mib must be non-negative")
    _parse_num_workers(args.num_workers)
    if args.force and args.append:
        raise ValueError("--force and --append are mutually exclusive")


def _generate_serial(
    args: argparse.Namespace,
    logger,
    device: torch.device,
    started_at: float,
) -> GenerationSummary:
    tree_device = torch.device("cpu") if device.type in ("cuda", "mps") else device
    logger.info("multiprocessing.enabled=False")
    logger.info("multiprocessing.num_workers=1")
    logger.info("search.backend=%s", args.backend)
    logger.info("search.tree_device=%s", tree_device)
    logger.info("search.evaluator_device=%s", device)
    if args.backend == "puct" and args.puct_inference_mode == "server":
        raise ValueError("--puct-inference-mode server requires --num-workers > 1")

    search = _build_search(args, device)
    generator = _build_generator(args, search, tree_device)
    writer = SelfPlayShardWriter(
        output_dir=args.out,
        samples_per_shard=args.samples_per_shard,
        metadata=_metadata(args, device, tree_device, extra={"num_workers": "1"}),
    )
    return _generate_chunks(
        args=args,
        logger=logger,
        generator=generator,
        writer=writer,
        device=device,
        output_dir=args.out,
        started_at=started_at,
    )


def _generate_parallel(
    args: argparse.Namespace,
    logger,
    device: torch.device,
    num_workers: int,
    started_at: float,
) -> GenerationSummary:
    if args.append:
        raise ValueError("--append is not supported with --num-workers > 1")
    worker_specs = _partition_games(args.games, num_workers)
    worker_root = args.out / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    logger.info("multiprocessing.enabled=True")
    logger.info("multiprocessing.num_workers=%s", num_workers)
    logger.info("multiprocessing.worker_root=%s", worker_root)
    logger.info("multiprocessing.start_method=%s", args.worker_start_method)
    logger.info("multiprocessing.torch_threads_per_worker=%s", args.torch_threads_per_worker)
    inference_mode = _resolve_puct_inference_mode(args, device, num_workers, logger)

    payloads = []
    for worker_index, worker_games in enumerate(worker_specs):
        worker_args = vars(args).copy()
        worker_args["games"] = worker_games
        worker_args["out"] = worker_root / f"worker-{worker_index:03d}"
        worker_args["append"] = False
        worker_args["force"] = True
        worker_args["quiet"] = True
        worker_args["num_workers"] = "1"
        worker_args["seed"] = _worker_seed(args.seed, worker_index)
        worker_args["_worker_index"] = worker_index
        worker_args["_resolved_puct_inference_mode"] = inference_mode
        worker_args["_requested_device"] = str(device)
        payloads.append({"worker_index": worker_index, "args": worker_args})

    results: list[WorkerResult] = []
    context = mp.get_context(args.worker_start_method)
    server_process: mp.Process | None = None
    request_queue = None
    response_queues = None
    if inference_mode == "server":
        request_queue = context.Queue(maxsize=max(4, num_workers * 4))
        response_queues = [context.Queue(maxsize=2) for _ in range(num_workers)]
        server_log_path = args.out / "logs" / f"puct_inference_server-{timestamp_slug()}.log"
        server_process = context.Process(
            target=run_policy_value_server,
            args=(
                request_queue,
                response_queues,
                str(args.checkpoint) if args.checkpoint is not None else None,
                str(device),
                args.puct_inference_batch_size,
                args.puct_server_max_batch_size,
                args.puct_server_batch_timeout_ms / 1000.0,
                str(server_log_path),
            ),
            name="connect4-puct-inference-server",
        )
        server_process.start()
        logger.info("puct_inference_server.pid=%s", server_process.pid)
        logger.info("puct_inference_server.log=%s", server_log_path)

    try:
        executor_kwargs: dict[str, Any] = {"max_workers": num_workers, "mp_context": context}
        if inference_mode == "server":
            executor_kwargs["initializer"] = _init_shared_inference_client
            executor_kwargs["initargs"] = (
                request_queue,
                response_queues,
                args.puct_server_response_timeout,
            )
        with concurrent.futures.ProcessPoolExecutor(**executor_kwargs) as executor:
            futures = [executor.submit(_run_worker, payload) for payload in payloads]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                logger.info(
                    "worker_complete index=%s games=%s samples=%s duration=%s output=%s",
                    result.worker_index,
                    result.games,
                    result.samples,
                    format_seconds(result.elapsed_seconds),
                    result.output_dir,
                )
    finally:
        if server_process is not None:
            assert request_queue is not None
            request_queue.put(None)
            server_process.join(timeout=30)
            if server_process.is_alive():
                logger.warning("puct_inference_server.terminate pid=%s", server_process.pid)
                server_process.terminate()
                server_process.join(timeout=10)

    if server_process is not None and server_process.exitcode not in (0, None):
        raise RuntimeError(f"PUCT inference server exited with code {server_process.exitcode}")

    results.sort(key=lambda result: result.worker_index)
    total_samples = _merge_worker_manifests(args.out, results, logger)
    elapsed = time.perf_counter() - started_at
    logger.info(
        "parallel_generation_complete workers=%s games=%s samples=%s elapsed=%s",
        num_workers,
        sum(result.games for result in results),
        total_samples,
        format_seconds(elapsed),
    )
    if device.type == "cuda":
        log_cuda_memory(logger, prefix="cuda.after_parallel_generation")
    return GenerationSummary(games=args.games, samples=total_samples, elapsed_seconds=elapsed)


def _generate_chunks(
    args: argparse.Namespace,
    logger,
    generator: SelfPlayGenerator,
    writer: SelfPlayShardWriter,
    device: torch.device,
    output_dir: Path,
    started_at: float,
) -> GenerationSummary:
    logger.info("generation_start games=%s output=%s", args.games, output_dir)
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
            human_bytes(directory_size_bytes(output_dir)),
        )
        log_cuda_memory(logger, prefix="cuda.after_chunk_write")
        del samples
        if device.type == "cuda":
            torch.cuda.empty_cache()
        chunk_index += 1

    elapsed = time.perf_counter() - started_at
    return GenerationSummary(games=args.games, samples=total_samples, elapsed_seconds=elapsed)


def _build_generator(args: argparse.Namespace, search, tree_device: torch.device) -> SelfPlayGenerator:
    self_play_config = SelfPlayConfig(
        batch_size=args.batch_size,
        device=tree_device,
        action_temperature=args.action_temperature,
        max_plies=args.max_plies,
        seed=args.seed,
    )
    return SelfPlayGenerator(search=search, config=self_play_config)


def _build_search(args: argparse.Namespace, device: torch.device, worker_index: int | None = None):
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

    inference_mode = getattr(args, "_resolved_puct_inference_mode", args.puct_inference_mode)
    if inference_mode == "server":
        if worker_index is None:
            raise RuntimeError("shared PUCT inference requires a worker_index")
        evaluator = _build_shared_inference_evaluator(worker_index)
    else:
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


def _run_worker(payload: dict[str, Any]) -> WorkerResult:
    worker_index = int(payload["worker_index"])
    args = argparse.Namespace(**payload["args"])
    args.out = Path(args.out)
    _set_torch_worker_threads(args.torch_threads_per_worker)
    maybe_empty_output_dir(args.out, append=False, force=True)
    args.out.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(
        args.out / "logs",
        name=f"generate_selfplay_worker_{worker_index:03d}",
        verbose=not args.quiet,
    )
    if not args.worker_stdout:
        _remove_stdout_logging(logger)
    started_at = time.perf_counter()
    requested_device = torch.device(getattr(args, "_requested_device", args.device))
    inference_mode = getattr(args, "_resolved_puct_inference_mode", args.puct_inference_mode)
    device = torch.device("cpu") if inference_mode == "server" else resolve_device(args.device)
    tree_device = torch.device("cpu") if device.type in ("cuda", "mps") else device
    logger.info("worker.index=%s", worker_index)
    logger.info("worker.games=%s", args.games)
    logger.info("worker.seed=%s", args.seed)
    logger.info("worker.torch_threads=%s", args.torch_threads_per_worker)
    logger.info("worker.requested_device=%s", requested_device)
    logger.info("worker.puct_inference_mode=%s", inference_mode)
    log_environment(logger, device)
    log_config(logger, "self-play worker config", vars(args))
    logger.info("search.backend=%s", args.backend)
    logger.info("search.tree_device=%s", tree_device)
    logger.info("search.evaluator_device=%s", requested_device if inference_mode == "server" else device)
    search = _build_search(args, device, worker_index=worker_index)
    generator = _build_generator(args, search, tree_device)
    writer = SelfPlayShardWriter(
        output_dir=args.out,
        samples_per_shard=args.samples_per_shard,
        metadata=_metadata(
            args,
            device,
            tree_device,
            extra={
                "num_workers": "1",
                "worker_index": str(worker_index),
                "worker_games": str(args.games),
            },
        ),
    )
    summary = _generate_chunks(
        args=args,
        logger=logger,
        generator=generator,
        writer=writer,
        device=device,
        output_dir=args.out,
        started_at=started_at,
    )
    logger.info(
        "worker_generation_complete index=%s games=%s samples=%s elapsed=%s",
        worker_index,
        summary.games,
        summary.samples,
        format_seconds(summary.elapsed_seconds),
    )
    return WorkerResult(
        worker_index=worker_index,
        games=summary.games,
        samples=summary.samples,
        elapsed_seconds=summary.elapsed_seconds,
        output_dir=str(args.out),
        manifest_path=str(args.out / "manifest.jsonl"),
    )


def _load_or_create_model(checkpoint: Path | None, device: torch.device) -> Connect4ResNet3D:
    if checkpoint is None:
        model = Connect4ResNet3D(ResNet3DConfig())
        model.eval()
        return model.to(device)
    state = load_checkpoint(checkpoint, map_location=device)
    state.model.eval()
    return state.model.to(device)


def _metadata(
    args: argparse.Namespace,
    device: torch.device,
    tree_device: torch.device,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
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
        "puct_inference_batch_size": args.puct_inference_batch_size,
        "puct_inference_mode": getattr(args, "_resolved_puct_inference_mode", args.puct_inference_mode),
        "puct_server_max_batch_size": args.puct_server_max_batch_size,
        "puct_server_batch_timeout_ms": args.puct_server_batch_timeout_ms,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "add_root_noise": args.add_root_noise,
        "root_dirichlet_alpha": args.root_dirichlet_alpha,
        "root_exploration_fraction": args.root_exploration_fraction,
        "policy_temperature": args.policy_temperature,
        "action_temperature": args.action_temperature,
        "max_plies": args.max_plies,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "worker_start_method": args.worker_start_method,
        "torch_threads_per_worker": args.torch_threads_per_worker,
        "worker_stdout": args.worker_stdout,
    }
    if extra is not None:
        values.update(extra)
    return {key: json.dumps(value) if isinstance(value, (dict, list)) else str(value) for key, value in values.items()}


def _resolve_puct_inference_mode(
    args: argparse.Namespace,
    device: torch.device,
    num_workers: int,
    logger,
) -> str:
    if args.backend != "puct":
        logger.info("puct_inference_mode=none")
        return "none"
    if args.puct_inference_mode == "worker":
        logger.info("puct_inference_mode=worker")
        return "worker"
    if args.puct_inference_mode == "server":
        logger.info("puct_inference_mode=server")
        return "server"

    resolved = "server" if device.type == "cuda" and num_workers > 1 else "worker"
    logger.info("puct_inference_mode=auto")
    logger.info("puct_inference_mode.resolved=%s", resolved)
    return resolved


def _init_shared_inference_client(
    request_queue: Any,
    response_queues: list[Any],
    response_timeout_seconds: float,
) -> None:
    global _SHARED_INFERENCE_REQUEST_QUEUE
    global _SHARED_INFERENCE_RESPONSE_QUEUES
    global _SHARED_INFERENCE_RESPONSE_TIMEOUT_SECONDS
    _SHARED_INFERENCE_REQUEST_QUEUE = request_queue
    _SHARED_INFERENCE_RESPONSE_QUEUES = response_queues
    _SHARED_INFERENCE_RESPONSE_TIMEOUT_SECONDS = float(response_timeout_seconds)


def _build_shared_inference_evaluator(worker_index: int) -> SharedInferenceClientEvaluator:
    if _SHARED_INFERENCE_REQUEST_QUEUE is None or _SHARED_INFERENCE_RESPONSE_QUEUES is None:
        raise RuntimeError("shared inference queues were not initialized in this worker")
    if worker_index < 0 or worker_index >= len(_SHARED_INFERENCE_RESPONSE_QUEUES):
        raise RuntimeError(f"worker_index {worker_index} has no inference response queue")
    return SharedInferenceClientEvaluator(
        request_queue=_SHARED_INFERENCE_REQUEST_QUEUE,
        response_queue=_SHARED_INFERENCE_RESPONSE_QUEUES[worker_index],
        response_index=worker_index,
        response_timeout_seconds=_SHARED_INFERENCE_RESPONSE_TIMEOUT_SECONDS,
    )


def _parse_num_workers(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in ("auto", "all", "max"):
        return None
    try:
        workers = int(normalized)
    except ValueError as exc:
        raise ValueError("--num-workers must be a positive integer, 'auto', or 'all'") from exc
    if workers <= 0:
        raise ValueError("--num-workers must be positive")
    return workers


def _resolve_num_workers(args: argparse.Namespace, device: torch.device, logger) -> int:
    requested = _parse_num_workers(args.num_workers)
    if requested is not None:
        workers = min(requested, args.games)
        logger.info("multiprocessing.requested_workers=%s", requested)
        logger.info("multiprocessing.resolved_workers=%s", workers)
        return workers

    cpu_workers = os.cpu_count() or 1
    workers = min(cpu_workers, args.games)
    logger.info("multiprocessing.auto.cpu_count=%s", cpu_workers)
    if device.type == "cuda":
        cuda_workers = _resolve_cuda_worker_cap(args, device, logger)
        workers = min(workers, cuda_workers)
    workers = max(1, workers)
    logger.info("multiprocessing.requested_workers=auto")
    logger.info("multiprocessing.resolved_workers=%s", workers)
    return workers


def _resolve_cuda_worker_cap(args: argparse.Namespace, device: torch.device, logger) -> int:
    if not torch.cuda.is_available():
        return 1
    index = device.index if device.index is not None else torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    free_mib = free_bytes // 1024**2
    total_mib = total_bytes // 1024**2
    usable_mib = max(0, int(free_mib) - args.cuda_memory_reserve_mib)
    cap = max(1, usable_mib // args.cuda_worker_memory_mib)
    logger.info("multiprocessing.auto.cuda_total_mib=%s", total_mib)
    logger.info("multiprocessing.auto.cuda_free_mib=%s", free_mib)
    logger.info("multiprocessing.auto.cuda_worker_memory_mib=%s", args.cuda_worker_memory_mib)
    logger.info("multiprocessing.auto.cuda_memory_reserve_mib=%s", args.cuda_memory_reserve_mib)
    logger.info("multiprocessing.auto.cuda_worker_cap=%s", cap)
    return int(cap)


def _partition_games(num_games: int, num_workers: int) -> list[int]:
    base, remainder = divmod(num_games, num_workers)
    return [base + (1 if index < remainder else 0) for index in range(num_workers) if base + (1 if index < remainder else 0) > 0]


def _worker_seed(seed: int | None, worker_index: int) -> int | None:
    if seed is None:
        return None
    return int(seed) + worker_index * 1009


def _set_torch_worker_threads(num_threads: int) -> None:
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_threads)
    except RuntimeError:
        pass


def _remove_stdout_logging(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)


def _merge_worker_manifests(output_dir: Path, results: list[WorkerResult], logger) -> int:
    manifest_path = output_dir / "manifest.jsonl"
    total_samples = 0
    with manifest_path.open("w", encoding="utf-8") as merged:
        for result in results:
            worker_dir = Path(result.output_dir)
            worker_manifest = Path(result.manifest_path)
            for line in worker_manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                record["shard"] = (worker_dir.relative_to(output_dir) / record["shard"]).as_posix()
                metadata = dict(record.get("metadata", {}))
                metadata["merged_worker_index"] = str(result.worker_index)
                metadata["merged_worker_games"] = str(result.games)
                record["metadata"] = metadata
                total_samples += int(record["num_samples"])
                merged.write(json.dumps(record, sort_keys=True))
                merged.write("\n")
    logger.info("merged_manifest=%s", manifest_path)
    logger.info("merged_manifest.workers=%s", len(results))
    logger.info("merged_manifest.samples=%s", total_samples)
    return total_samples


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
