from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch

from c4zero_supervised.data import Stage0Dataset
from c4zero_supervised.stage0_train import evaluate as evaluate_stage0
from c4zero_tools.version import current_version_info
from c4zero_train.checkpoint import load_checkpoint, save_checkpoint
from c4zero_train.export import export_torchscript_model
from c4zero_train.replay import ReplayBuffer
from c4zero_train.symmetry_metrics import SymmetryProbeConfig, evaluate_symmetry
from c4zero_train.trainer import TrainConfig, make_optimizer, train_step


def repair_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair checkpoint coordinate bias with all-8 symmetry orbit training.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--out", type=Path, default=Path("/tmp/thakwani/rl-runs/symmetry-repair-round21"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--base-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--probe-positions", type=int, default=256)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--stage0-manifest")
    parser.add_argument("--stage0-batch-size", type=int, default=2048)
    args = parser.parse_args(argv)

    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.base_batch_size <= 0:
        raise ValueError("--base-batch-size must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    model, payload = load_checkpoint(args.checkpoint, device=args.device)
    replay = ReplayBuffer.from_manifests(args.manifest, replay_games="all")
    stage0_dataset = Stage0Dataset.from_manifest(args.stage0_manifest) if args.stage0_manifest else None

    train_config = TrainConfig(
        batch_size=min(args.base_batch_size, len(replay.samples)),
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        symmetry_mode="orbit",
    )
    optimizer = make_optimizer(model, train_config)
    rng = random.Random(args.seed)
    global_start_step = int(payload["step"])
    metrics_path = args.out / "repair_metrics.jsonl"
    run_started = time.perf_counter()
    last_loss = None
    trained_samples = 0

    config_payload = {
        "checkpoint": str(args.checkpoint),
        "manifests": [str(path) for path in args.manifest],
        "out": str(args.out),
        "version": current_version_info(),
        "train_config": {
            **asdict(train_config),
            "base_batch_size": train_config.batch_size,
            "effective_batch_size": train_config.batch_size * 8,
            "replay_games": "all",
            "replay_metadata": replay.metadata(),
        },
        "eval": {
            "eval_every": args.eval_every,
            "probe_positions": args.probe_positions,
            "probe_batch_size": args.probe_batch_size,
            "stage0_manifest": args.stage0_manifest,
            "stage0_batch_size": args.stage0_batch_size,
        },
    }
    (args.out / "repair_config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")

    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def log_record(record: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    def evaluate_and_save(repair_step: int) -> None:
        step_dir = args.out / f"step-{repair_step:06d}"
        checkpoint_dir = step_dir / "checkpoint"
        symmetry = evaluate_symmetry(
            model,
            replay,
            device=args.device,
            config=SymmetryProbeConfig(
                positions=args.probe_positions,
                seed=args.seed + 10_001,
                batch_size=args.probe_batch_size,
            ),
        )
        if stage0_dataset is None:
            stage0_probe = {
                "skipped": True,
                "reason": "--stage0-manifest was not provided",
            }
        else:
            stage0_probe = {
                "skipped": False,
                "manifest": args.stage0_manifest,
                "metrics": evaluate_stage0(model, stage0_dataset, args.stage0_batch_size, torch.device(args.device)),
            }

        checkpoint_metrics = {
            "kind": "symmetry_repair_checkpoint",
            "repair_step": repair_step,
            "global_step": global_start_step + repair_step,
            "last_loss": None if last_loss is None else last_loss.total,
            "last_policy_loss": None if last_loss is None else last_loss.policy,
            "last_value_loss": None if last_loss is None else last_loss.value,
            "policy_weight": args.policy_weight,
            "value_weight": args.value_weight,
            "symmetry_mode": "orbit",
            "base_batch_size": train_config.batch_size,
            "effective_batch_size": train_config.batch_size * 8,
            "replay_games": "all",
            **replay.metadata(),
        }
        save_checkpoint(
            checkpoint_dir,
            model,
            optimizer,
            scheduler=None,
            step=global_start_step + repair_step,
            epoch=0,
            replay_manifests=args.manifest,
            metrics=checkpoint_metrics,
        )
        export_torchscript_model(model, checkpoint_dir / "inference.ts", device=args.device)
        write_json(step_dir / "symmetry_metrics.json", symmetry)
        write_json(step_dir / "stage0_probe.json", stage0_probe)
        log_record(
            {
                "kind": "repair_eval",
                "repair_step": repair_step,
                "global_step": global_start_step + repair_step,
                "empty_corner_max_minus_min": symmetry["empty_board"]["groups"]["corners"]["max_minus_min"],
                "empty_edge_max_minus_min": symmetry["empty_board"]["groups"]["edges"]["max_minus_min"],
                "empty_center_max_minus_min": symmetry["empty_board"]["groups"]["centers"]["max_minus_min"],
                "mean_policy_l1": symmetry["equivariance"]["mean_policy_l1"],
                "max_policy_abs": symmetry["equivariance"]["max_policy_abs"],
                "mean_value_std": symmetry["equivariance"]["mean_value_std"],
                "stage0_skipped": stage0_probe["skipped"],
            }
        )

    evaluate_and_save(0)
    for repair_step in range(1, args.steps + 1):
        samples = replay.sample_orbit_batch(train_config.batch_size, rng)
        trained_samples += len(samples)
        last_loss = train_step(
            model,
            optimizer,
            samples,
            device=args.device,
            policy_weight=args.policy_weight,
            value_weight=args.value_weight,
        )
        if args.log_every_steps > 0 and (repair_step == 1 or repair_step % args.log_every_steps == 0):
            elapsed = max(time.perf_counter() - run_started, 1e-9)
            log_record(
                {
                    "kind": "repair_train_progress",
                    "repair_step": repair_step,
                    "global_step": global_start_step + repair_step,
                    "steps": args.steps,
                    "loss": last_loss.total,
                    "policy_loss": last_loss.policy,
                    "value_loss": last_loss.value,
                    "paper_total_loss": last_loss.paper_total_loss,
                    "base_batch_size": train_config.batch_size,
                    "effective_batch_size": len(samples),
                    "samples_per_sec": trained_samples / elapsed,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )
        if repair_step % args.eval_every == 0 or repair_step == args.steps:
            evaluate_and_save(repair_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(repair_main())
