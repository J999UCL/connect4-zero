"""Shared helpers for verbose command-line scripts."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

import torch


def configure_logging(log_dir: Path, name: str, verbose: bool = True) -> logging.Logger:
    """Configure stdout and file logging with immediate flushing."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler.setFormatter(formatter)

    log_path = log_dir / f"{name}-{timestamp_slug()}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info("log_file=%s", log_path)
    return logger


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device, but torch.cuda.is_available() is false")
        if resolved.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("requested MPS device, but torch.backends.mps.is_available() is false")
        return resolved

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_environment(logger: logging.Logger, device: torch.device) -> None:
    logger.info("environment.python=%s", sys.version.replace("\n", " "))
    logger.info("environment.platform=%s", platform.platform())
    logger.info("environment.cwd=%s", Path.cwd())
    logger.info("environment.device=%s", device)
    logger.info("environment.torch=%s", torch.__version__)
    logger.info("environment.cuda_available=%s", torch.cuda.is_available())
    logger.info("environment.mps_available=%s", torch.backends.mps.is_available())
    logger.info("environment.git_commit=%s", git_commit())
    logger.info("environment.git_branch=%s", git_branch())
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "RL_DATA",
        "RL_RUNS",
        "UV_CACHE_DIR",
        "PIP_CACHE_DIR",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
    ):
        if name in os.environ:
            logger.info("environment.%s=%s", name, os.environ[name])

    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        logger.info("cuda.index=%s", index)
        logger.info("cuda.name=%s", props.name)
        logger.info("cuda.total_memory_gb=%.2f", props.total_memory / 1024**3)
        log_cuda_memory(logger, prefix="cuda.initial")
        log_nvidia_smi(logger)


def log_config(logger: logging.Logger, title: str, values: Mapping[str, object]) -> None:
    logger.info("%s", "=" * 88)
    logger.info("%s", title)
    logger.info("%s", "=" * 88)
    for key in sorted(values):
        logger.info("config.%s=%s", key, values[key])


def log_cuda_memory(logger: logging.Logger, prefix: str = "cuda") -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    logger.info(
        "%s.memory allocated_mb=%.1f reserved_mb=%.1f max_allocated_mb=%.1f",
        prefix,
        allocated,
        reserved,
        max_allocated,
    )


def log_nvidia_smi(logger: logging.Logger) -> None:
    if shutil.which("nvidia-smi") is None:
        logger.info("nvidia_smi=not_found")
        return
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning("nvidia_smi_failed=%s", result.stderr.strip())
        return
    for line in result.stdout.strip().splitlines():
        logger.info("nvidia_smi.gpu=%s", line)


def git_commit() -> str:
    return _git_output(["git", "rev-parse", "--short", "HEAD"])


def git_branch() -> str:
    return _git_output(["git", "branch", "--show-current"])


def _git_output(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {remainder:.0f}s"


class RateTracker:
    """Track elapsed time and rates for script progress logging."""

    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.last_at = self.started_at

    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    def mark(self) -> tuple[float, float]:
        now = time.perf_counter()
        since_last = now - self.last_at
        self.last_at = now
        return now - self.started_at, since_last


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0


def maybe_empty_output_dir(path: Path, append: bool, force: bool) -> None:
    manifest = path / "manifest.jsonl"
    if not manifest.exists():
        return
    if append:
        return
    if force:
        shutil.rmtree(path)
        return
    raise FileExistsError(
        f"{path} already contains manifest.jsonl; pass --append to continue or --force to delete it"
    )


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class SelfPlayProgressLogger:
    """Structured callback for ``SelfPlayGenerator`` progress events."""

    def __init__(self, logger: logging.Logger, device: torch.device) -> None:
        self.logger = logger
        self.device = device
        self.tracker = RateTracker()
        self._search_start_by_ply: dict[int, float] = {}

    def __call__(self, event: str, payload: Mapping[str, float | int | str]) -> None:
        now = time.perf_counter()
        if event == "batch_start":
            self.logger.info(
                "batch_start index=%s games=%s completed=%s/%s elapsed=%s",
                payload["batch_index"],
                payload["batch_games"],
                payload["games_completed"],
                payload["games_total"],
                format_seconds(self.tracker.elapsed()),
            )
            log_cuda_memory(self.logger, prefix="cuda.batch_start")
            return

        if event == "ply_search_start":
            ply = int(payload["ply"])
            self._search_start_by_ply[ply] = now
            self.logger.info(
                "ply_search_start ply=%s active_games=%s finished_games=%s elapsed=%s",
                ply,
                payload["active_games"],
                payload["finished_games"],
                format_seconds(self.tracker.elapsed()),
            )
            return

        if event == "ply_search_end":
            sync_if_cuda(self.device)
            now = time.perf_counter()
            ply = int(payload["ply"])
            duration = now - self._search_start_by_ply.get(ply, now)
            visits = int(payload["total_visits"])
            visits_per_sec = visits / duration if duration > 0 else 0.0
            self.logger.info(
                "ply_search_end ply=%s active_games=%s duration=%s total_visits=%s visits_per_sec=%.1f "
                "leaf_evals=%s terminal_evals=%s leaf_batches=%s max_leaf_batch=%s "
                "tree_reuse_hits=%s tree_fresh_roots=%s "
                "expanded_children=%s expansion_batches=%s max_expansion_batch=%s "
                "timing_prepare=%.4f timing_select=%.4f timing_expand=%.4f timing_rollout=%.4f "
                "timing_backprop=%.4f timing_build=%.4f mean_root_value=%.4f mean_policy_entropy=%.4f",
                ply,
                payload["active_games"],
                format_seconds(duration),
                visits,
                visits_per_sec,
                payload.get("leaf_evaluations", "na"),
                payload.get("terminal_evaluations", "na"),
                payload.get("leaf_batches", "na"),
                payload.get("max_leaf_batch", "na"),
                payload.get("tree_reuse_hits", "na"),
                payload.get("tree_fresh_roots", "na"),
                payload.get("expanded_children", "na"),
                payload.get("expansion_batches", "na"),
                payload.get("max_expansion_batch", "na"),
                payload.get("timing_prepare", -1.0),
                payload.get("timing_select", -1.0),
                payload.get("timing_expand", -1.0),
                payload.get("timing_rollout", -1.0),
                payload.get("timing_backprop", -1.0),
                payload.get("timing_build", -1.0),
                payload["mean_root_value"],
                payload["mean_policy_entropy"],
            )
            log_cuda_memory(self.logger, prefix="cuda.ply_search_end")
            return

        if event == "ply_end":
            self.logger.info(
                "ply_end ply=%s active_games=%s wins=%s draws=%s newly_done=%s still_active=%s samples_so_far=%s",
                payload["ply"],
                payload["active_games"],
                payload["wins"],
                payload["draws"],
                payload["newly_done"],
                payload["still_active"],
                payload["samples_so_far"],
            )
            return

        if event == "batch_end":
            elapsed, since_last = self.tracker.mark()
            games_done = int(payload["games_completed"])
            games_total = int(payload["games_total"])
            samples = int(payload["batch_samples"])
            self.logger.info(
                "batch_end index=%s games=%s samples=%s completed=%s/%s batch_duration=%s elapsed=%s "
                "games_per_sec_total=%.3f samples_per_sec_batch=%.1f",
                payload["batch_index"],
                payload["batch_games"],
                samples,
                games_done,
                games_total,
                format_seconds(since_last),
                format_seconds(elapsed),
                games_done / elapsed if elapsed > 0 else 0.0,
                samples / since_last if since_last > 0 else 0.0,
            )
            log_cuda_memory(self.logger, prefix="cuda.batch_end")
