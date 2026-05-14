from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from c4zero_train.checkpoint import save_checkpoint
from c4zero_train.export import export_checkpoint, export_torchscript_model
from c4zero_train.model import create_model, count_parameters
from c4zero_train.replay import ReplayBuffer, ReplayConfig
from c4zero_train.trainer import TrainConfig, make_optimizer, make_scheduler, train_steps


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
    parser.add_argument("--preset", choices=["tiny", "small", "medium"], default="small")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    replay_config = ReplayConfig.for_preset(args.preset)
    replay = ReplayBuffer.from_manifests(args.manifest, replay_config.replay_games)
    model = create_model(args.preset).to(args.device)
    train_config = TrainConfig(batch_size=min(replay_config.batch_size, len(replay.samples)), seed=args.seed)
    optimizer = make_optimizer(model, train_config)
    scheduler = make_scheduler(optimizer)
    losses = train_steps(model, replay, optimizer, scheduler, train_config, args.steps, device=args.device)
    metrics = {
        "last_loss": losses[-1].total,
        "last_policy_loss": losses[-1].policy,
        "last_value_loss": losses[-1].value,
        **replay.metadata(),
    }
    save_checkpoint(args.out, model, optimizer, scheduler, step=args.steps, epoch=0, replay_manifests=args.manifest, metrics=metrics)
    export_torchscript_model(model, args.out / "inference.ts", device=args.device)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(train_main())
