from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from c4zero_supervised.data import Stage0Dataset
from c4zero_supervised.stage0_train import evaluate
from c4zero_train.checkpoint import load_checkpoint


def eval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a Stage 0 curriculum split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    model, payload = load_checkpoint(args.checkpoint, device=args.device)
    dataset = Stage0Dataset.from_manifest(args.manifest)
    metrics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload["step"]),
        "manifest": str(args.manifest),
        "dataset": dataset.describe(),
        "metrics": evaluate(model, dataset, args.batch_size, torch.device(args.device)),
    }
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(eval_main())
