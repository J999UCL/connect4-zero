"""Export the PyTorch 3D ResNet to the Rust ONNX inference contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from connect4_zero.model import Connect4ResNet3D
from connect4_zero.model.checkpoint import load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    if args.checkpoint is None:
        model = Connect4ResNet3D()
    else:
        model = load_checkpoint(args.checkpoint, map_location=device).model
    model.to(device)
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 2, 4, 4, 4), dtype=torch.float32, device=device)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={
            "input": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
