import multiprocessing as mp
from pathlib import Path

import torch

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.search import SharedInferenceClientEvaluator, run_policy_value_server


def test_shared_inference_server_returns_legal_priors(tmp_path: Path) -> None:
    context = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    request_queue = context.Queue()
    response_queues = [context.Queue()]
    server = context.Process(
        target=run_policy_value_server,
        args=(
            request_queue,
            response_queues,
            None,
            "cpu",
            4,
            8,
            0.01,
            str(tmp_path / "inference_server.log"),
        ),
    )
    server.start()
    try:
        evaluator = SharedInferenceClientEvaluator(
            request_queue=request_queue,
            response_queue=response_queues[0],
            response_index=0,
            response_timeout_seconds=30.0,
        )
        states = Connect4x4x4Batch(batch_size=2)
        states.heights[:, 0, 0] = 4

        result = evaluator.evaluate_batch(states)

        assert result.priors.shape == (2, 16)
        assert result.values.shape == (2,)
        assert torch.all(result.priors[:, 0].eq(0))
        assert torch.allclose(result.priors.sum(dim=1), torch.ones(2), atol=1e-6)
        assert torch.all(result.values.ge(-1))
        assert torch.all(result.values.le(1))
    finally:
        request_queue.put(None)
        server.join(timeout=10)
        if server.is_alive():
            server.terminate()
            server.join(timeout=10)

    assert server.exitcode == 0
