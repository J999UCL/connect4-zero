import json

import numpy as np
import torch
import pytest

from c4zero_tools.datasets import Sample
from c4zero_train.checkpoint import load_checkpoint, restore_optimizer_and_scheduler, save_checkpoint
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
    assert breakdown.optimized_total == breakdown.total
    assert breakdown.l2_regularization > 0.0
    assert breakdown.paper_total_loss > breakdown.total
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

    loaded_optimizer = make_optimizer(loaded, TrainConfig(batch_size=2, learning_rate=0.01))
    loaded_scheduler = make_scheduler(loaded_optimizer)
    restore_optimizer_and_scheduler(payload, loaded_optimizer, loaded_scheduler)
    assert loaded_optimizer.state_dict()["state"]
    assert loaded_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]


def test_checkpoint_rejects_mutated_version_metadata(tmp_path):
    model = create_model("tiny")
    optimizer = make_optimizer(model, TrainConfig(batch_size=2, learning_rate=0.01))
    scheduler = make_scheduler(optimizer)
    save_checkpoint(tmp_path, model, optimizer, scheduler, step=1, epoch=0, replay_manifests=[])
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["version"]["encoder_version"] = "999.0.0"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="encoder_version mismatch"):
        load_checkpoint(tmp_path)


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
