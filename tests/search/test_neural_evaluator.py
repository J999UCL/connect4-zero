import torch
from torch import nn

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE
from connect4_zero.model import Connect4ResNet3D
from connect4_zero.search import NeuralPolicyValueEvaluator


class CountingModel(Connect4ResNet3D):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor):
        self.calls += 1
        batch = x.shape[0]
        logits = torch.arange(ACTION_SIZE, dtype=torch.float32, device=x.device).repeat(batch, 1)
        values = torch.full((batch,), 0.25, dtype=torch.float32, device=x.device)
        return logits, values


def test_neural_evaluator_masks_illegal_actions_and_normalizes_priors() -> None:
    model = CountingModel()
    evaluator = NeuralPolicyValueEvaluator(model, device="cpu")
    states = Connect4x4x4Batch(batch_size=2)
    states.heights[:, 0, 0] = 4

    result = evaluator.evaluate_batch(states)

    assert result.priors.shape == (2, ACTION_SIZE)
    assert result.values.shape == (2,)
    assert torch.all(result.priors[:, 0].eq(0))
    assert torch.allclose(result.priors.sum(dim=1), torch.ones(2))
    assert model.calls == 1


def test_neural_evaluator_chunked_inference_matches_full_batch() -> None:
    states = Connect4x4x4Batch(batch_size=5)
    full = NeuralPolicyValueEvaluator(CountingModel(), device="cpu", inference_batch_size=16)
    chunked = NeuralPolicyValueEvaluator(CountingModel(), device="cpu", inference_batch_size=2)

    full_result = full.evaluate_batch(states)
    chunked_result = chunked.evaluate_batch(states)

    assert torch.allclose(full_result.priors, chunked_result.priors)
    assert torch.allclose(full_result.values, chunked_result.values)


def test_neural_evaluator_bypasses_terminal_states() -> None:
    model = CountingModel()
    evaluator = NeuralPolicyValueEvaluator(model, device="cpu")
    states = Connect4x4x4Batch(batch_size=2)
    states.done[:] = True
    states.outcome[0] = 1
    states.outcome[1] = 0

    result = evaluator.evaluate_batch(states)

    assert model.calls == 0
    assert torch.all(result.priors.eq(0))
    assert result.values.tolist() == [-1.0, 0.0]
