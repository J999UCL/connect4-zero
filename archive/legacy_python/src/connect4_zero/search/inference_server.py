"""Shared neural inference process for CPU self-play workers."""

from __future__ import annotations

import logging
import queue
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.model import Connect4ResNet3D, ResNet3DConfig, load_checkpoint
from connect4_zero.search.neural_evaluator import NeuralPolicyValueEvaluator
from connect4_zero.search.types import PolicyValueBatch


@dataclass(frozen=True)
class InferenceRequest:
    """One synchronous policy/value request from a CPU worker."""

    request_id: str
    response_index: int
    board: torch.Tensor
    heights: torch.Tensor
    done: torch.Tensor
    outcome: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.board.shape[0])


@dataclass(frozen=True)
class InferenceResponse:
    """Policy/value response routed back to a CPU worker."""

    request_id: str
    priors: torch.Tensor | None = None
    values: torch.Tensor | None = None
    error: str | None = None


class SharedInferenceClientEvaluator:
    """Queue-backed evaluator proxy used inside CPU self-play workers."""

    def __init__(
        self,
        request_queue: Any,
        response_queue: Any,
        response_index: int,
        response_timeout_seconds: float = 600.0,
    ) -> None:
        if response_timeout_seconds <= 0:
            raise ValueError("response_timeout_seconds must be positive")
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.response_index = int(response_index)
        self.response_timeout_seconds = float(response_timeout_seconds)
        self._counter = 0

    def evaluate_batch(self, states: Connect4x4x4Batch) -> PolicyValueBatch:
        """Send CPU states to the server and wait for priors/values."""
        request_id = f"{self.response_index}:{self._counter}"
        self._counter += 1
        cpu_states = states if states.device.type == "cpu" else states.to("cpu")
        request = InferenceRequest(
            request_id=request_id,
            response_index=self.response_index,
            board=cpu_states.board.detach().cpu().clone(),
            heights=cpu_states.heights.detach().cpu().clone(),
            done=cpu_states.done.detach().cpu().clone(),
            outcome=cpu_states.outcome.detach().cpu().clone(),
        )
        self.request_queue.put(request)

        try:
            response: InferenceResponse = self.response_queue.get(timeout=self.response_timeout_seconds)
        except queue.Empty as exc:
            raise TimeoutError(f"timed out waiting for inference response {request_id}") from exc

        if response.request_id != request_id:
            raise RuntimeError(f"received inference response {response.request_id}, expected {request_id}")
        if response.error is not None:
            raise RuntimeError(response.error)
        if response.priors is None or response.values is None:
            raise RuntimeError(f"inference response {request_id} did not include tensors")

        target_device = states.device
        return PolicyValueBatch(
            priors=response.priors.to(device=target_device, dtype=torch.float32),
            values=response.values.to(device=target_device, dtype=torch.float32),
        )


def run_policy_value_server(
    request_queue: Any,
    response_queues: Sequence[Any],
    checkpoint: str | None,
    device: str,
    inference_batch_size: int,
    max_batch_size: int,
    batch_timeout_seconds: float,
    log_path: str | None = None,
) -> None:
    """Own the neural net on one device and serve batched inference requests."""
    logger = _server_logger(log_path)
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if batch_timeout_seconds < 0:
        raise ValueError("batch_timeout_seconds must be non-negative")

    torch_device = torch.device(device)
    logger.info(
        "inference_server_start device=%s checkpoint=%s inference_batch_size=%s max_batch_size=%s batch_timeout=%.4f",
        torch_device,
        checkpoint,
        inference_batch_size,
        max_batch_size,
        batch_timeout_seconds,
    )
    model = _load_or_create_model(Path(checkpoint) if checkpoint is not None else None, torch_device)
    evaluator = NeuralPolicyValueEvaluator(
        model=model,
        device=torch_device,
        inference_batch_size=inference_batch_size,
    )

    num_batches = 0
    num_requests = 0
    num_states = 0
    max_seen_batch = 0
    started_at = time.perf_counter()

    while True:
        item = request_queue.get()
        if item is None:
            break
        requests = [item]
        total_states = item.batch_size
        stop_after_batch = False
        deadline = time.perf_counter() + batch_timeout_seconds

        while total_states < max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                next_item = request_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if next_item is None:
                stop_after_batch = True
                break
            requests.append(next_item)
            total_states += next_item.batch_size

        try:
            _evaluate_requests(evaluator, requests, response_queues)
        except Exception:
            error = traceback.format_exc()
            logger.error("inference_server_batch_failed requests=%s states=%s\n%s", len(requests), total_states, error)
            _send_error_responses(requests, response_queues, error)
            raise

        num_batches += 1
        num_requests += len(requests)
        num_states += total_states
        max_seen_batch = max(max_seen_batch, total_states)
        if num_batches == 1 or num_batches % 100 == 0:
            elapsed = time.perf_counter() - started_at
            logger.info(
                "inference_server_progress batches=%s requests=%s states=%s max_batch=%s states_per_sec=%.1f",
                num_batches,
                num_requests,
                num_states,
                max_seen_batch,
                num_states / elapsed if elapsed > 0 else 0.0,
            )
        if stop_after_batch:
            break

    elapsed = time.perf_counter() - started_at
    logger.info(
        "inference_server_stop batches=%s requests=%s states=%s max_batch=%s elapsed=%.2fs states_per_sec=%.1f",
        num_batches,
        num_requests,
        num_states,
        max_seen_batch,
        elapsed,
        num_states / elapsed if elapsed > 0 else 0.0,
    )


def _evaluate_requests(
    evaluator: NeuralPolicyValueEvaluator,
    requests: Sequence[InferenceRequest],
    response_queues: Sequence[Any],
) -> None:
    states = _make_batch(requests)
    result = evaluator.evaluate_batch(states)
    priors = result.priors.detach().cpu()
    values = result.values.detach().cpu()

    offset = 0
    for request in requests:
        end = offset + request.batch_size
        response_queues[request.response_index].put(
            InferenceResponse(
                request_id=request.request_id,
                priors=priors[offset:end].clone(),
                values=values[offset:end].clone(),
            )
        )
        offset = end


def _send_error_responses(
    requests: Sequence[InferenceRequest],
    response_queues: Sequence[Any],
    error: str,
) -> None:
    for request in requests:
        response_queues[request.response_index].put(
            InferenceResponse(request_id=request.request_id, error=error)
        )


def _make_batch(requests: Sequence[InferenceRequest]) -> Connect4x4x4Batch:
    states = Connect4x4x4Batch(sum(request.batch_size for request in requests), device="cpu")
    states.board = torch.cat([request.board for request in requests], dim=0).clone()
    states.heights = torch.cat([request.heights for request in requests], dim=0).clone()
    states.done = torch.cat([request.done for request in requests], dim=0).clone()
    states.outcome = torch.cat([request.outcome for request in requests], dim=0).clone()
    return states


def _load_or_create_model(checkpoint: Path | None, device: torch.device) -> Connect4ResNet3D:
    if checkpoint is None:
        model = Connect4ResNet3D(ResNet3DConfig())
    else:
        model = load_checkpoint(checkpoint, map_location=device).model
    model.eval()
    return model.to(device)


def _server_logger(log_path: str | None) -> logging.Logger:
    logger = logging.getLogger(f"connect4_zero.inference_server.{id(log_path)}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if log_path is None:
        logger.addHandler(logging.NullHandler())
        return logger
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
