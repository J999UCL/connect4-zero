from pathlib import Path

import numpy as np
import torch

from c4az.arena import ArenaConfig, evaluate_arena
from c4az.cli import train_main
from c4az.data import ReplayBuffer, SelfPlayDataset, SelfPlaySample, write_dataset
from c4az.game import ACTION_SIZE, initial_position, symmetry_action_permutation
from c4az.mcts import UniformEvaluator
from c4az.network import create_model
from c4az.selfplay import SelfPlayConfig, generate_self_play_games
from c4az.train import AlphaZeroLoss, load_checkpoint, save_checkpoint, train_step


def test_shard_round_trip_and_symmetry_augmentation(tmp_path: Path) -> None:
    position = initial_position().play(0).play(5)
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[3] = 1.0
    visits = np.zeros(ACTION_SIZE, dtype=np.uint32)
    visits[3] = 9
    sample = SelfPlaySample.from_position(
        position,
        policy=policy,
        value=1.0,
        visit_counts=visits,
        action=3,
        game_id=42,
    )

    manifest = write_dataset(tmp_path, [sample], metadata={"model_checkpoint": None})
    dataset = SelfPlayDataset(manifest, augment_symmetries=True)
    transformed = dataset[1]
    new_action = symmetry_action_permutation(1)[3]

    assert len(dataset) == 8
    assert transformed["policy"][new_action] == 1.0
    assert transformed["visit_counts"][new_action] == 9
    assert transformed["value"].item() == 1.0


def test_replay_buffer_samples_uniformly_from_most_recent_games(tmp_path: Path) -> None:
    manifests = []
    for round_index, action in enumerate((0, 1, 2)):
        position = initial_position()
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        policy[action] = 1.0
        visits = np.zeros(ACTION_SIZE, dtype=np.uint32)
        visits[action] = 3
        sample = SelfPlaySample.from_position(
            position,
            policy=policy,
            value=1.0,
            visit_counts=visits,
            action=action,
            game_id=0,
        )
        manifests.append(write_dataset(tmp_path / f"round-{round_index}" / "data", [sample], metadata={"model_checkpoint": None}))

    replay = ReplayBuffer(manifests, replay_games=2, augment_symmetries=False, seed=7)
    batch = replay.sample_batch(64)

    assert replay.num_games == 2
    assert replay.num_positions == 2
    assert replay.selected_games == ((1, 0), (2, 0))
    assert batch["input"].shape == (64, 2, 4, 4, 4)
    assert set(batch["action"].tolist()) <= {1, 2}


def test_train_cli_samples_from_replay_buffer_manifests(tmp_path: Path) -> None:
    manifests = []
    for round_index, value in enumerate((-1.0, 1.0)):
        position = initial_position()
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        policy[round_index] = 1.0
        visits = np.zeros(ACTION_SIZE, dtype=np.uint32)
        visits[round_index] = 5
        sample = SelfPlaySample.from_position(
            position,
            policy=policy,
            value=value,
            visit_counts=visits,
            action=round_index,
            game_id=round_index,
        )
        manifests.append(write_dataset(tmp_path / f"round-{round_index}" / "data", [sample], metadata={"model_checkpoint": None}))

    output_dir = tmp_path / "train"
    train_main(
        [
            "--manifest",
            str(manifests[0]),
            "--manifest",
            str(manifests[1]),
            "--replay-games",
            "2",
            "--preset",
            "tiny",
            "--out",
            str(output_dir),
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--seed",
            "7",
        ]
    )

    assert (output_dir / "checkpoint.pt").exists()


def test_self_play_values_match_final_result_perspective() -> None:
    games = generate_self_play_games(UniformEvaluator(), SelfPlayConfig(games=1, simulations_per_move=2, seed=3))
    game = games[0]

    assert game.samples
    for sample in game.samples:
        flips = game.plies - sample.ply
        assert sample.value == game.terminal_value * ((-1.0) ** flips)
        assert np.isclose(sample.policy.sum(), 1.0)


def test_one_train_step_changes_weights_and_has_finite_losses(tmp_path: Path) -> None:
    position = initial_position()
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[0] = 1.0
    visits = np.ones(ACTION_SIZE, dtype=np.uint32)
    sample = SelfPlaySample.from_position(
        position,
        policy=policy,
        value=0.25,
        visit_counts=visits,
        action=0,
        game_id=0,
    )
    manifest = write_dataset(tmp_path / "data", [sample], metadata={"model_checkpoint": None})
    batch = SelfPlayDataset(manifest)[0]
    batch = {key: value.unsqueeze(0) if torch.is_tensor(value) and value.ndim > 0 else value.reshape(1) for key, value in batch.items()}
    model = create_model("tiny")
    before = model.policy_head[-1].bias.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    metrics = train_step(model, batch, optimizer, AlphaZeroLoss())

    assert np.isfinite(metrics["loss"])
    assert not torch.allclose(before, model.policy_head[-1].bias)


def test_checkpoint_save_load_for_training(tmp_path: Path) -> None:
    model = create_model("tiny")
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(path, model, optimizer, step=5, epoch=1, metrics={"loss": 2.0})
    loaded, payload = load_checkpoint(path, optimizer=optimizer)

    assert payload["step"] == 5
    assert payload["metrics"]["loss"] == 2.0
    assert type(loaded) is type(model)


def test_tiny_arena_smoke_runs_deterministically() -> None:
    result = evaluate_arena(
        UniformEvaluator(),
        UniformEvaluator(),
        ArenaConfig(games=2, simulations_per_move=2, seed=9),
    )

    assert result.games == 2
    assert result.wins + result.losses + result.draws == 2


def test_tiny_end_to_end_loop(tmp_path: Path) -> None:
    games = generate_self_play_games(UniformEvaluator(), SelfPlayConfig(games=2, simulations_per_move=2, seed=5))
    samples = [sample for game in games for sample in game.samples]
    manifest = write_dataset(tmp_path / "data", samples, metadata={"model_checkpoint": None})
    dataset = SelfPlayDataset(manifest)
    model = create_model("tiny")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = dataset[0]
    batch = {key: value.unsqueeze(0) if torch.is_tensor(value) and value.ndim > 0 else value.reshape(1) for key, value in batch.items()}

    train_step(model, batch, optimizer, AlphaZeroLoss())
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model, optimizer, step=1)
    loaded, _ = load_checkpoint(checkpoint)
    assert loaded.config.preset == "tiny"
