from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch

from c4zero_train.checkpoint import load_checkpoint, restore_optimizer_and_scheduler, save_checkpoint
from c4zero_train.export import export_checkpoint, export_torchscript_model
from c4zero_train.model import create_model, count_parameters
from c4zero_train.replay import ReplayBuffer, ReplayConfig
from c4zero_train.trainer import TrainConfig, make_optimizer, make_scheduler, sample_training_batch, train_step


def _resolve_symmetry_mode(symmetry_mode: str | None, augment_symmetries: bool) -> str:
    if symmetry_mode is None:
        return "random" if augment_symmetries else "none"
    if augment_symmetries and symmetry_mode != "random":
        raise ValueError("--augment-symmetries is equivalent to --symmetry-mode random")
    return symmetry_mode


def inspect_model_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect c4zero PyTorch model presets.")
    parser.add_argument("--preset", choices=["tiny", "small", "medium"], default="small")
    args = parser.parse_args(argv)
    model = create_model(args.preset)
    print(json.dumps({"preset": args.preset, "params": count_parameters(model), "config": asdict(model.config)}, indent=2))
    return 0


def export_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a PyTorch checkpoint or random preset to TorchScript.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--preset", choices=["tiny", "small", "medium"], default="small")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.checkpoint:
        path = export_checkpoint(args.checkpoint, args.out, device=args.device)
    else:
        model = create_model(args.preset)
        path = args.out
        export_torchscript_model(model, path, device=args.device)
    print(path)
    return 0


def train_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train c4zero PyTorch model from C++ self-play shards.")
    parser.add_argument("--preset", choices=["tiny", "small", "medium"])
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--replay-games", default=None, help="Replay window game count, or 'all' for curriculum data.")
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--augment-symmetries", action="store_true", help="Apply random 4x4 base-plane symmetries during replay sampling.")
    parser.add_argument("--symmetry-mode", choices=["none", "random", "orbit"])
    parser.add_argument("--batch-size", type=int, help="Training batch size; in orbit mode this is the base batch before 8-way expansion.")
    parser.add_argument("--replay-sampling", choices=["uniform", "recent-mix"], default="uniform")
    parser.add_argument("--recent-games", type=int, default=4_000)
    parser.add_argument("--recent-fraction", type=float, default=0.75)
    parser.add_argument("--reset-optimizer", action="store_true", help="Resume model weights but start a fresh optimizer/scheduler.")
    parser.add_argument("--log-every-steps", type=int, default=0)
    args = parser.parse_args(argv)
    symmetry_mode = _resolve_symmetry_mode(args.symmetry_mode, args.augment_symmetries)

    payload = None
    if args.resume is not None:
        model, payload = load_checkpoint(args.resume, device=args.device)
        if args.preset is not None and args.preset != model.config.preset:
            raise ValueError(f"--preset {args.preset} does not match resumed checkpoint preset {model.config.preset}")
        preset = model.config.preset
    else:
        preset = args.preset or "small"
        model = create_model(preset).to(args.device)

    replay_config = ReplayConfig.for_preset(preset)
    replay_games: int | str = replay_config.replay_games
    if args.replay_games is not None:
        replay_games = "all" if args.replay_games == "all" else int(args.replay_games)
    replay = ReplayBuffer.from_manifests(args.manifest, replay_games)
    configured_batch_size = args.batch_size if args.batch_size is not None else replay_config.batch_size
    train_config = TrainConfig(
        batch_size=min(configured_batch_size, len(replay.samples)),
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        augment_symmetries=args.augment_symmetries,
        symmetry_mode=symmetry_mode,
        replay_sampling=args.replay_sampling,
        recent_games=args.recent_games,
        recent_fraction=args.recent_fraction,
    )
    optimizer = make_optimizer(model, train_config)
    scheduler = make_scheduler(optimizer)
    start_step = 0
    start_epoch = 0
    if payload is not None and not args.reset_optimizer:
        restore_optimizer_and_scheduler(payload, optimizer, scheduler)
        start_step = int(payload["step"])
        start_epoch = int(payload["epoch"])
    losses = []
    rng = random.Random(train_config.seed)
    started_at = time.perf_counter()
    trained_samples = 0
    for local_step in range(1, args.steps + 1):
        samples = sample_training_batch(replay, train_config, rng)
        trained_samples += len(samples)
        loss = train_step(
            model,
            optimizer,
            samples,
            device=args.device,
            policy_weight=train_config.policy_weight,
            value_weight=train_config.value_weight,
        )
        losses.append(loss)
        if scheduler is not None:
            scheduler.step()
        if args.log_every_steps > 0 and (local_step == 1 or local_step % args.log_every_steps == 0 or local_step == args.steps):
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            print(
                json.dumps(
                    {
                        "kind": "train_progress",
                        "local_step": local_step,
                        "step": start_step + local_step,
                        "steps": args.steps,
                        "loss": loss.total,
                        "policy_loss": loss.policy,
                        "value_loss": loss.value,
                        "paper_total_loss": loss.paper_total_loss,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "base_batch_size": train_config.batch_size,
                        "effective_batch_size": len(samples),
                        "symmetry_mode": symmetry_mode,
                        "replay_sampling": args.replay_sampling,
                        "recent_games": args.recent_games,
                        "recent_fraction": args.recent_fraction,
                        "samples_per_sec": trained_samples / elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    final_step = start_step + args.steps
    metrics = {
        "last_loss": losses[-1].total,
        "last_optimized_loss": losses[-1].optimized_total,
        "last_paper_total_loss": losses[-1].paper_total_loss,
        "last_policy_loss": losses[-1].policy,
        "last_value_loss": losses[-1].value,
        "last_l2_regularization": losses[-1].l2_regularization,
        "policy_weight": args.policy_weight,
        "value_weight": args.value_weight,
        "learning_rate": train_config.learning_rate,
        "momentum": train_config.momentum,
        "weight_decay": train_config.weight_decay,
        "symmetry_augmentation": symmetry_mode != "none",
        "symmetry_mode": symmetry_mode,
        "base_batch_size": train_config.batch_size,
        "effective_batch_size": train_config.batch_size * (8 if symmetry_mode == "orbit" else 1),
        "replay_games": replay_games,
        **replay.sampling_metadata(train_config.replay_sampling_config()),
        **replay.metadata(),
    }
    save_checkpoint(args.out, model, optimizer, scheduler, step=final_step, epoch=start_epoch, replay_manifests=args.manifest, metrics=metrics)
    export_torchscript_model(model, args.out / "inference.ts", device=args.device)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(train_main())
