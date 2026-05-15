from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from c4zero_supervised.data import Stage0Dataset, write_dataset_summary
from c4zero_supervised.metrics import MetricTotals, batch_metrics, forward_policy_logits, policy_loss
from c4zero_tools.version import current_version_info
from c4zero_train.checkpoint import load_checkpoint, save_checkpoint
from c4zero_train.export import export_torchscript_model
from c4zero_train.model import AlphaZeroNet, create_model


@dataclass(frozen=True, slots=True)
class Stage0TrainConfig:
    preset: str
    train_manifest: str
    val_manifest: str | None
    batch_size: int
    epochs: int
    max_steps: int | None
    learning_rate: float
    momentum: float
    weight_decay: float
    log_every_steps: int
    eval_every_steps: int
    seed: int
    device: str
    freeze_value_head: bool


def make_optimizer(model: AlphaZeroNet, config: Stage0TrainConfig) -> torch.optim.SGD:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.SGD(
        parameters,
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def set_value_head_trainable(model: AlphaZeroNet, trainable: bool) -> None:
    for module in [model.value_conv, model.value_bn, model.value_fc1, model.value_fc2]:
        for parameter in module.parameters():
            parameter.requires_grad = trainable


@torch.no_grad()
def evaluate(
    model: AlphaZeroNet,
    dataset: Stage0Dataset,
    batch_size: int,
    device: torch.device | str,
) -> dict:
    model.eval()
    rng = np.random.default_rng(0)
    totals = MetricTotals()
    for indices in dataset.iter_epoch_indices(batch_size, rng, shuffle=False):
        batch = dataset.batch(indices, device=device)
        logits = forward_policy_logits(model, batch.inputs)
        batch_totals = batch_metrics(logits, batch, dataset.category_names)
        totals.update(batch_totals)
        for name, category_totals in batch_totals.category_totals.items():
            totals.category_totals.setdefault(name, MetricTotals()).update(category_totals)
    return totals.as_dict()


def train(
    model: AlphaZeroNet,
    train_dataset: Stage0Dataset,
    val_dataset: Stage0Dataset | None,
    config: Stage0TrainConfig,
    out_dir: Path,
    start_step: int = 0,
    optimizer_state: dict | None = None,
) -> tuple[torch.optim.Optimizer, list[dict], int]:
    device = torch.device(config.device)
    model.to(device)
    set_value_head_trainable(model, not config.freeze_value_head)
    optimizer = make_optimizer(model, config)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    rng = np.random.default_rng(config.seed)
    metrics_path = out_dir / "metrics.jsonl"
    history: list[dict] = []
    global_step = start_step
    run_start = time.perf_counter()
    epoch_start = run_start
    rolling_loss_sum = 0.0
    rolling_samples = 0
    steps_per_epoch = int(np.ceil(len(train_dataset) / config.batch_size))

    def cuda_memory() -> dict[str, float]:
        if device.type != "cuda":
            return {}
        return {
            "cuda_allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
            "cuda_reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
            "cuda_max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        }

    def log_record(record: dict, echo: bool = True) -> None:
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if echo:
            print(json.dumps(record, sort_keys=True), flush=True)

    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        epoch_samples = 0
        step_in_epoch = 0
        for indices in train_dataset.iter_epoch_indices(config.batch_size, rng, shuffle=True):
            if config.max_steps is not None and global_step >= start_step + config.max_steps:
                return optimizer, history, global_step
            batch_size = int(len(indices))
            model.train()
            batch = train_dataset.batch(indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_policy_logits(model, batch.inputs)
            loss = policy_loss(logits, batch.target_policy)
            loss.backward()
            optimizer.step()
            global_step += 1
            step_in_epoch += 1
            epoch_samples += batch_size
            rolling_loss_sum += float(loss.detach().cpu()) * batch_size
            rolling_samples += batch_size

            if config.log_every_steps > 0 and global_step % config.log_every_steps == 0:
                now = time.perf_counter()
                elapsed = max(now - run_start, 1e-9)
                epoch_elapsed = max(now - epoch_start, 1e-9)
                record = {
                    "kind": "train_progress",
                    "step": global_step,
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "steps_per_epoch": steps_per_epoch,
                    "epoch_progress": min(1.0, step_in_epoch / steps_per_epoch),
                    "samples_seen": (global_step - start_step) * config.batch_size,
                    "epoch_samples_seen": epoch_samples,
                    "latest_policy_loss": float(loss.detach().cpu()),
                    "rolling_policy_loss": rolling_loss_sum / max(rolling_samples, 1),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "steps_per_sec": (global_step - start_step) / elapsed,
                    "samples_per_sec": ((global_step - start_step) * config.batch_size) / elapsed,
                    "epoch_samples_per_sec": epoch_samples / epoch_elapsed,
                    **cuda_memory(),
                }
                log_record(record)

            if config.eval_every_steps > 0 and global_step % config.eval_every_steps == 0:
                record = {
                    "kind": "train_step",
                    "step": global_step,
                    "epoch": epoch,
                    "train_policy_loss": float(loss.detach().cpu()),
                    "elapsed_sec": time.perf_counter() - run_start,
                    **cuda_memory(),
                }
                if val_dataset is not None:
                    record["validation"] = evaluate(model, val_dataset, config.batch_size, device)
                log_record(record)

        record = {
            "kind": "epoch_end",
            "step": global_step,
            "epoch": epoch + 1,
            "epoch_elapsed_sec": time.perf_counter() - epoch_start,
            "elapsed_sec": time.perf_counter() - run_start,
            **cuda_memory(),
        }
        if val_dataset is not None:
            record["validation"] = evaluate(model, val_dataset, config.batch_size, device)
        log_record(record)

    return optimizer, history, global_step


def train_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Policy-only supervised Stage 0 curriculum training.")
    parser.add_argument("--preset", choices=["tiny", "small", "medium"], default="small")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--train-value-head", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    train_dataset = Stage0Dataset.from_manifest(args.train_manifest)
    val_dataset = Stage0Dataset.from_manifest(args.val_manifest) if args.val_manifest else None

    start_step = 0
    optimizer_state = None
    if args.resume is not None:
        model, payload = load_checkpoint(args.resume, device=args.device)
        if model.config.preset != args.preset:
            raise ValueError(f"--preset {args.preset} does not match resumed checkpoint preset {model.config.preset}")
        start_step = int(payload["step"])
        optimizer_state = payload.get("optimizer_state")
    else:
        model = create_model(args.preset)

    config = Stage0TrainConfig(
        preset=args.preset,
        train_manifest=str(Path(args.train_manifest)),
        val_manifest=str(Path(args.val_manifest)) if args.val_manifest else None,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        log_every_steps=args.log_every_steps,
        eval_every_steps=args.eval_every_steps,
        seed=args.seed,
        device=args.device,
        freeze_value_head=not args.train_value_head,
    )
    (args.out / "supervised_config.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "version": current_version_info(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_dataset_summary(
        args.out / "datasets.json",
        {
            name: dataset
            for name, dataset in [("train", train_dataset), ("validation", val_dataset)]
            if dataset is not None
        },
    )

    optimizer, history, final_step = train(
        model,
        train_dataset,
        val_dataset,
        config,
        args.out,
        start_step=start_step,
        optimizer_state=optimizer_state,
    )
    final_metrics = history[-1] if history else {}
    save_checkpoint(
        args.out,
        model,
        optimizer,
        scheduler=None,
        step=final_step,
        epoch=config.epochs,
        replay_manifests=[config.train_manifest],
        metrics={
            "supervised_stage": 0,
            "final": final_metrics,
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset) if val_dataset is not None else 0,
            "test_samples": 0,
        },
    )
    export_torchscript_model(model, args.out / "inference.ts", device=args.device)
    print(json.dumps(final_metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(train_main())
