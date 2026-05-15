import numpy as np
import torch

from c4zero_tools.datasets import Sample
from c4zero_train.replay import ReplayBuffer
from c4zero_train.symmetry_metrics import SymmetryProbeConfig, empty_board_metrics, evaluate_symmetry


class FakeModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor, drifting_values: bool = False):
        super().__init__()
        self.register_buffer("logits", logits.to(torch.float32))
        self.drifting_values = drifting_values

    def forward(self, x):
        batch = x.shape[0]
        logits = self.logits.repeat(batch, 1)
        if self.drifting_values:
            values = torch.linspace(-0.5, 0.5, batch, dtype=torch.float32, device=x.device)
        else:
            values = torch.zeros(batch, dtype=torch.float32, device=x.device)
        return logits.to(x.device), values


def sample() -> Sample:
    policy = np.full(16, 1.0 / 16.0, dtype=np.float32)
    visits = np.ones(16, dtype=np.uint32)
    return Sample(
        current_bits=(1 << 0) | (1 << 5),
        opponent_bits=1 << 10,
        heights=tuple([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        ply=3,
        game_id=0,
        legal_mask=0xFFFF,
        action=5,
        policy=policy,
        visit_counts=visits,
        value=1.0,
    )


def test_empty_board_metric_detects_asymmetric_corner_policy():
    logits = torch.zeros(1, 16)
    logits[0, 0] = 6.0
    metrics = empty_board_metrics(FakeModel(logits))

    assert metrics["groups"]["corners"]["max_minus_min"] > 0.5


def test_empty_board_metric_is_zero_for_uniform_logits():
    metrics = empty_board_metrics(FakeModel(torch.zeros(1, 16)))

    assert metrics["groups"]["corners"]["max_minus_min"] == 0.0
    assert metrics["groups"]["edges"]["max_minus_min"] == 0.0
    assert metrics["groups"]["centers"]["max_minus_min"] == 0.0


def test_equivariance_metric_detects_policy_and_value_drift():
    logits = torch.zeros(1, 16)
    logits[0, 0] = 5.0
    replay = ReplayBuffer([sample()])

    metrics = evaluate_symmetry(
        FakeModel(logits, drifting_values=True),
        replay,
        config=SymmetryProbeConfig(positions=1, seed=1, batch_size=8),
    )

    assert metrics["equivariance"]["mean_policy_l1"] > 0.1
    assert metrics["equivariance"]["mean_value_std"] > 0.0
