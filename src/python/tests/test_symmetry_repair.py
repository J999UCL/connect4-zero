import numpy as np

from c4zero_tools.datasets import Sample
from c4zero_train.checkpoint import save_checkpoint
from c4zero_train.model import create_model
from c4zero_train.replay import ReplayBuffer
import c4zero_train.symmetry_repair as repair
from c4zero_train.trainer import TrainConfig, make_optimizer


def sample() -> Sample:
    policy = np.full(16, 1.0 / 16.0, dtype=np.float32)
    visits = np.ones(16, dtype=np.uint32)
    return Sample(
        current_bits=1,
        opponent_bits=2,
        heights=tuple([1, 1] + [0] * 14),
        ply=2,
        game_id=0,
        legal_mask=0xFFFF,
        action=0,
        policy=policy,
        visit_counts=visits,
        value=0.0,
    )


def test_symmetry_repair_smoke_writes_eval_artifacts(monkeypatch, tmp_path):
    model = create_model("tiny")
    optimizer = make_optimizer(model, TrainConfig(batch_size=1, learning_rate=0.01))
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, optimizer, scheduler=None, step=7, epoch=0, replay_manifests=[])

    monkeypatch.setattr(
        repair.ReplayBuffer,
        "from_manifests",
        lambda _manifests, replay_games=None: ReplayBuffer([sample()]),
    )

    out = tmp_path / "repair"
    repair.repair_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            "manifest.json",
            "--out",
            str(out),
            "--steps",
            "2",
            "--eval-every",
            "1",
            "--base-batch-size",
            "1",
            "--probe-positions",
            "1",
            "--probe-batch-size",
            "8",
            "--device",
            "cpu",
        ]
    )

    assert (out / "repair_config.json").exists()
    assert (out / "repair_metrics.jsonl").exists()
    for step in (0, 1, 2):
        step_dir = out / f"step-{step:06d}"
        assert (step_dir / "checkpoint" / "model_state.pt").exists()
        assert (step_dir / "checkpoint" / "inference.ts").exists()
        assert (step_dir / "symmetry_metrics.json").exists()
        assert (step_dir / "stage0_probe.json").exists()
