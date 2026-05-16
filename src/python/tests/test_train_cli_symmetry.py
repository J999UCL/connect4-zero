import numpy as np

from c4zero_tools.datasets import Sample
import c4zero_train.cli as train_cli
from c4zero_train.losses import LossBreakdown


def sample() -> Sample:
    return Sample(
        current_bits=0,
        opponent_bits=0,
        heights=tuple([0] * 16),
        ply=0,
        game_id=0,
        legal_mask=0xFFFF,
        action=0,
        policy=np.full(16, 1.0 / 16.0, dtype=np.float32),
        visit_counts=np.ones(16, dtype=np.uint32),
        value=0.0,
    )


class FakeReplay:
    def __init__(self):
        self.samples = [sample()] * 8
        self.sample_batch_calls = []
        self.sample_orbit_batch_calls = []
        self.train_step_calls = []

    def sample_batch(self, batch_size, rng, augment_symmetries=False, sampling_config=None):
        self.sample_batch_calls.append((batch_size, augment_symmetries, sampling_config.mode))
        return [sample()] * batch_size

    def sample_orbit_batch(self, base_batch_size, rng, sampling_config=None):
        self.sample_orbit_batch_calls.append((base_batch_size, sampling_config.mode, sampling_config.recent_games, sampling_config.recent_fraction))
        return [sample()] * (base_batch_size * 8)

    def metadata(self):
        return {"num_samples": len(self.samples), "num_games": 1}

    def sampling_metadata(self, sampling_config):
        return {"replay_sampling": sampling_config.mode}


def patch_training(monkeypatch, replay):
    monkeypatch.setattr(train_cli.ReplayBuffer, "from_manifests", lambda _manifests, _replay_games: replay)

    def fake_train_step(_model, optimizer, _samples, **kwargs):
        replay.train_step_calls.append(
            {
                "learning_rate": optimizer.param_groups[0]["lr"],
                "policy_weight": kwargs["policy_weight"],
                "value_weight": kwargs["value_weight"],
            }
        )
        return LossBreakdown(
            total=1.0,
            policy=0.7,
            value=0.3,
            l2_regularization=0.1,
            paper_total_loss=1.1,
            optimized_total=1.0,
        )

    monkeypatch.setattr(train_cli, "train_step", fake_train_step)
    monkeypatch.setattr(train_cli, "make_scheduler", lambda _optimizer: None)
    monkeypatch.setattr(train_cli, "save_checkpoint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_cli, "export_torchscript_model", lambda *_args, **_kwargs: None)


def test_train_cli_random_symmetry_flag_reaches_replay(monkeypatch, tmp_path):
    replay = FakeReplay()
    patch_training(monkeypatch, replay)

    train_cli.train_main(
        [
            "--preset",
            "tiny",
            "--manifest",
            "manifest.json",
            "--steps",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--augment-symmetries",
        ]
    )

    assert replay.sample_batch_calls == [(8, True, "uniform")]
    assert replay.sample_orbit_batch_calls == []


def test_train_cli_orbit_symmetry_mode_uses_orbit_batch(monkeypatch, tmp_path):
    replay = FakeReplay()
    patch_training(monkeypatch, replay)

    train_cli.train_main(
        [
            "--preset",
            "tiny",
            "--manifest",
            "manifest.json",
            "--steps",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--batch-size",
            "2",
            "--symmetry-mode",
            "orbit",
        ]
    )

    assert replay.sample_batch_calls == []
    assert replay.sample_orbit_batch_calls == [(2, "uniform", 4_000, 0.75)]


def test_train_cli_recent_mix_reaches_orbit_replay(monkeypatch, tmp_path):
    replay = FakeReplay()
    patch_training(monkeypatch, replay)

    train_cli.train_main(
        [
            "--preset",
            "tiny",
            "--manifest",
            "manifest.json",
            "--steps",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--batch-size",
            "2",
            "--symmetry-mode",
            "orbit",
            "--replay-sampling",
            "recent-mix",
            "--recent-games",
            "123",
            "--recent-fraction",
            "0.8",
        ]
    )

    assert replay.sample_batch_calls == []
    assert replay.sample_orbit_batch_calls == [(2, "recent-mix", 123, 0.8)]


def test_train_cli_learning_rate_reaches_optimizer(monkeypatch, tmp_path):
    replay = FakeReplay()
    patch_training(monkeypatch, replay)

    train_cli.train_main(
        [
            "--preset",
            "tiny",
            "--manifest",
            "manifest.json",
            "--steps",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--learning-rate",
            "0.01",
            "--policy-weight",
            "0.9",
            "--value-weight",
            "0.4",
        ]
    )

    assert replay.train_step_calls == [
        {"learning_rate": 0.01, "policy_weight": 0.9, "value_weight": 0.4}
    ]
