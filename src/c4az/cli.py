from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from c4az.arena import ArenaConfig, evaluate_arena
from c4az.data import SelfPlayDataset, write_dataset
from c4az.mcts import TorchEvaluator
from c4az.network import create_model, count_parameters
from c4az.selfplay import SelfPlayConfig, generate_self_play_games
from c4az.train import AlphaZeroLoss, TrainerConfig, load_checkpoint, save_checkpoint, train_step


def selfplay_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate clean AlphaZero self-play data.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--preset", default="small", choices=("tiny", "small", "medium"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    model = create_model(args.preset)
    if args.checkpoint is not None:
        model, _ = load_checkpoint(args.checkpoint, model=model, map_location=args.device)
    evaluator = TorchEvaluator(model, device=args.device)
    games = generate_self_play_games(
        evaluator,
        SelfPlayConfig(games=args.games, simulations_per_move=args.simulations, seed=args.seed),
    )
    samples = [sample for game in games for sample in game.samples]
    manifest = write_dataset(
        args.out,
        samples,
        metadata={
            "model_checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "mcts_config": {"simulations_per_move": args.simulations},
            "temperature_config": {"cutoff_ply": 12},
        },
    )
    print(manifest)
    return 0


def train_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the clean AlphaZero model.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preset", default="small", choices=("tiny", "small", "medium"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model = create_model(args.preset).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    if args.resume is not None:
        model, payload = load_checkpoint(args.resume, model=model, optimizer=optimizer, map_location=device)
        start_step = int(payload["step"])
    dataset = SelfPlayDataset(args.manifest, augment_symmetries=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loss_fn = AlphaZeroLoss()
    step = start_step
    while step < start_step + args.steps:
        for batch in loader:
            step += 1
            metrics = train_step(model, batch, optimizer, loss_fn, device=device)
            print(json.dumps({"step": step, **metrics}))
            if step >= start_step + args.steps:
                break
    save_checkpoint(args.out / "checkpoint.pt", model, optimizer, step=step, metrics=metrics)
    return 0


def loop_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a small AlphaZero self-play/train loop.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preset", default="tiny", choices=("tiny", "small", "medium"))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    checkpoint = None
    for round_index in range(args.rounds):
        round_dir = args.out / f"round-{round_index:03d}"
        selfplay_args = [
            "--out",
            str(round_dir / "data"),
            "--preset",
            args.preset,
            "--games",
            str(args.games),
            "--simulations",
            str(args.simulations),
            "--device",
            args.device,
        ]
        if checkpoint is not None:
            selfplay_args.extend(["--checkpoint", str(checkpoint)])
        selfplay_main(selfplay_args)
        train_main(
            [
                "--manifest",
                str(round_dir / "data" / "manifest.json"),
                "--preset",
                args.preset,
                "--out",
                str(round_dir / "train"),
                "--steps",
                str(args.train_steps),
                "--device",
                args.device,
            ]
        )
        checkpoint = round_dir / "train" / "checkpoint.pt"
    return 0


def arena_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate two AlphaZero checkpoints.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    candidate, _ = load_checkpoint(args.candidate, map_location=args.device)
    baseline, _ = load_checkpoint(args.baseline, map_location=args.device)
    result = evaluate_arena(
        TorchEvaluator(candidate, device=args.device),
        TorchEvaluator(baseline, device=args.device),
        ArenaConfig(games=args.games, simulations_per_move=args.simulations),
    )
    print(json.dumps(result.__dict__))
    return 0


def inspect_checkpoint_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect an AlphaZero checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args(argv)
    model, payload = load_checkpoint(args.checkpoint, map_location="cpu")
    print(json.dumps({"config": payload["model_config"], "parameters": count_parameters(model), "step": payload["step"]}, indent=2))
    return 0
