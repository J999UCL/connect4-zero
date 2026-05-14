"""Training CLI skeleton for the 3D ResNet policy/value model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from connect4_zero.data import RustBinarySelfPlayDataset, SelfPlayDataset
from connect4_zero.model import Connect4ResNet3D, ResNet3DConfig, count_parameters, load_checkpoint, save_checkpoint
from connect4_zero.scripts._common import configure_logging, log_config, log_environment, resolve_device
from connect4_zero.train import AlphaZeroLoss, TrainerConfig, default_checkpoint_path, train_step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Connect4 Zero 3D ResNet from self-play shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Self-play manifest path.")
    parser.add_argument(
        "--dataset-format",
        choices=("auto", "safetensors", "rust-binary"),
        default="auto",
        help="Dataset format for --manifest.",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/train-resnet"), help="Output directory.")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional absolute max optimizer step; 0 means no cap.")
    parser.add_argument("--max-new-steps", type=int, default=0, help="Optional additional optimizer steps after resume; 0 means no cap.")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--grad-clip-norm", type=float, default=5.0, help="Gradient clipping norm.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--apply-symmetries", action="store_true", help="Use 8 D4 board symmetries.")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Checkpoint interval in optimizer steps.")
    parser.add_argument("--dry-run", action="store_true", help="Build objects and exit without training.")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging; file log remains detailed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = build_parser().parse_args(argv)
    _validate_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.out / "logs", name="train_resnet", verbose=not args.quiet)
    device = resolve_device(args.device)
    log_environment(logger, device)
    log_config(logger, "ResNet training config", vars(args))

    model = _load_or_create_model(args.resume, device)
    logger.info("model.parameters=%s", count_parameters(model))
    if args.dry_run:
        logger.info("dry_run=true; no training executed")
        return 0
    if args.manifest is None:
        raise ValueError("--manifest is required unless --dry-run is set")

    dataset = _load_dataset(args.manifest, args.dataset_format, apply_symmetries=args.apply_symmetries)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    trainer_config = TrainerConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        num_workers=args.num_workers,
        amp=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    resume_step = 0
    resume_epoch = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, model=model, optimizer=optimizer, map_location=device)
        resume_step = checkpoint.step
        resume_epoch = checkpoint.epoch
        logger.info("resume_loaded checkpoint=%s step=%s epoch=%s", args.resume, resume_step, resume_epoch)
    loss_fn = AlphaZeroLoss()
    step = resume_step
    target_step = _target_step(args.max_steps, args.max_new_steps, resume_step)

    logger.info("training_start samples=%s start_step=%s target_step=%s", len(dataset), step, target_step)
    for epoch in range(args.epochs):
        for batch in loader:
            metrics = train_step(model, batch, optimizer, loss_fn, device=device, config=trainer_config)
            step += 1
            logger.info(
                "step=%s epoch=%s loss=%.6f policy_loss=%.6f value_loss=%.6f grad_norm=%.4f",
                step,
                resume_epoch + epoch,
                metrics["loss"],
                metrics["policy_loss"],
                metrics["value_loss"],
                metrics["grad_norm"],
            )
            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                save_checkpoint(default_checkpoint_path(args.out, step), model, optimizer, step=step, epoch=resume_epoch + epoch, metrics=metrics)
            if target_step and step >= target_step:
                save_checkpoint(default_checkpoint_path(args.out, step), model, optimizer, step=step, epoch=resume_epoch + epoch, metrics=metrics)
                logger.info("training_complete reason=target_step step=%s", step)
                return 0

    save_checkpoint(default_checkpoint_path(args.out, step), model, optimizer, step=step, epoch=resume_epoch + args.epochs)
    logger.info("training_complete reason=epochs step=%s", step)
    return 0


def _load_or_create_model(checkpoint: Path | None, device: torch.device) -> Connect4ResNet3D:
    if checkpoint is None:
        model = Connect4ResNet3D(ResNet3DConfig())
    else:
        model = load_checkpoint(checkpoint, map_location=device).model
    return model.to(device)


def _load_dataset(
    manifest: Path,
    dataset_format: str,
    apply_symmetries: bool,
) -> SelfPlayDataset | RustBinarySelfPlayDataset:
    resolved_format = _resolve_dataset_format(manifest, dataset_format)
    if resolved_format == "safetensors":
        return SelfPlayDataset(manifest, apply_symmetries=apply_symmetries)
    if resolved_format == "rust-binary":
        return RustBinarySelfPlayDataset(manifest, apply_symmetries=apply_symmetries)
    raise ValueError(f"unsupported dataset format: {resolved_format}")


def _resolve_dataset_format(manifest: Path, dataset_format: str) -> str:
    if dataset_format != "auto":
        return dataset_format
    if manifest.suffix == ".jsonl":
        return "safetensors"
    if manifest.suffix == ".json":
        return "rust-binary"
    raise ValueError(f"cannot auto-detect dataset format from manifest suffix: {manifest}")


def _target_step(max_steps: int, max_new_steps: int, resume_step: int) -> int:
    if max_steps and max_new_steps:
        raise ValueError("use only one of --max-steps or --max-new-steps")
    if max_new_steps:
        return resume_step + max_new_steps
    return max_steps


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative")
    if args.max_new_steps < 0:
        raise ValueError("--max-new-steps must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")
    if args.grad_clip_norm <= 0:
        raise ValueError("--grad-clip-norm must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")


if __name__ == "__main__":
    raise SystemExit(main())
