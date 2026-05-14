import numpy as np
import torch

from c4zero_tools.datasets import Sample
from c4zero_train.checkpoint import load_checkpoint, save_checkpoint
from c4zero_train.export import export_torchscript_model
from c4zero_train.model import create_model
from c4zero_train.trainer import TrainConfig, make_optimizer, make_scheduler, train_step


def samples():
    policy = np.zeros(16, dtype=np.float32)
    policy[0] = 0.75
    policy[5] = 0.25
    visits = np.zeros(16, dtype=np.uint32)
    visits[0] = 3
    visits[5] = 1
    return [
        Sample(1, 2, tuple([0] * 16), 2, 0, 0xFFFF, 0, policy, visits, 1.0),
        Sample(4, 8, tuple([0] * 16), 3, 0, 0xFFFF, 5, policy, visits, -1.0),
    ]


def test_train_step_changes_weights_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(1)
    model = create_model("tiny")
    optimizer = make_optimizer(model, TrainConfig(batch_size=2, learning_rate=0.01))
    scheduler = make_scheduler(optimizer)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    breakdown = train_step(model, optimizer, samples())
    assert breakdown.total > 0.0
    assert breakdown.policy > 0.0
    assert breakdown.value >= 0.0
    assert any(not torch.equal(before[name], parameter.detach()) for name, parameter in model.named_parameters())

    save_checkpoint(tmp_path, model, optimizer, scheduler, step=1, epoch=0, replay_manifests=["manifest.json"])
    loaded, payload = load_checkpoint(tmp_path)
    assert payload["step"] == 1
    model.eval()
    loaded.eval()
    x = torch.zeros((1, 2, 4, 4, 4), dtype=torch.float32)
    with torch.no_grad():
        expected = model(x)
        actual = loaded(x)
    assert torch.allclose(expected[0], actual[0])
    assert torch.allclose(expected[1], actual[1])


def test_torchscript_export_matches_eager(tmp_path):
    torch.manual_seed(2)
    model = create_model("tiny").eval()
    path = tmp_path / "inference.ts"
    export_torchscript_model(model, path)
    scripted = torch.jit.load(str(path)).eval()
    x = torch.randn((2, 2, 4, 4, 4), dtype=torch.float32)
    with torch.no_grad():
        eager = model(x)
        traced = scripted(x)
    assert torch.allclose(eager[0], traced[0], atol=1e-6, rtol=1e-5)
    assert torch.allclose(eager[1], traced[1], atol=1e-6, rtol=1e-5)
