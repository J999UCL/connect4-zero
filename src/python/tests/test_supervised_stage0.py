import json

import numpy as np
import torch

from c4zero_supervised.data import HEADER, MAGIC, SAMPLE_DTYPE, Stage0Dataset
from c4zero_supervised.stage0_eval import eval_main
from c4zero_supervised.stage0_train import evaluate, train_main
from c4zero_tools.version import current_version_info
from c4zero_train.model import create_model


def write_stage0_fixture(root, count=16):
    (root / "shards").mkdir(parents=True)
    records = np.zeros(count, dtype=SAMPLE_DTYPE)
    for index in range(count):
        records["current_bits"][index] = 1 << (index % 16)
        records["opponent_bits"][index] = 0
        records["heights"][index] = np.zeros(16, dtype=np.uint8)
        records["ply"][index] = 1
        records["game_id"][index] = index
        records["legal_mask"][index] = 0xFFFF
        records["action"][index] = 0
        records["policy"][index, 0] = 1.0
        records["visit_counts"][index, 0] = 256
        records["value"][index] = 0.0
    shard = root / "shards" / "shard-000000.c4az"
    shard.write_bytes(HEADER.pack(MAGIC, 1, 0, count) + records.tobytes())
    category_counts = {
        "immediate_win": count // 2,
        "immediate_block": count - count // 2,
        "safe_move_vs_blunder": 0,
        "playable_vs_floating_threat": 0,
        "fork_create": 0,
        "fork_block": 0,
        "minimax_depth3_policy": 0,
    }
    manifest = {
        "schema_version": current_version_info()["dataset_schema_version"],
        "created_at": "2026-05-15T00:00:00Z",
        "num_games": count,
        "num_samples": count,
        "dataset_kind": "curriculum_stage0",
        "model_checkpoint": "none",
        "shard_paths": ["shards/shard-000000.c4az"],
        "config": {"category_counts": category_counts},
        "version": current_version_info(),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_stage0_dataset_loads_columnar_batches(tmp_path):
    manifest = write_stage0_fixture(tmp_path)
    dataset = Stage0Dataset.from_manifest(manifest)
    assert len(dataset) == 16
    assert dataset.describe()["category_counts"]["immediate_win"] == 8
    assert dataset.describe()["category_counts"]["immediate_block"] == 8

    batch = dataset.batch(np.array([0, 1, 2]), device="cpu")
    assert batch.inputs.shape == (3, 2, 4, 4, 4)
    assert batch.target_policy.shape == (3, 16)
    assert batch.legal_mask.tolist() == [0xFFFF, 0xFFFF, 0xFFFF]
    assert batch.inputs[0, 0, 0, 0, 0].item() == 1.0
    assert batch.inputs[1, 0, 0, 0, 1].item() == 1.0


def test_stage0_evaluate_reports_policy_metrics(tmp_path):
    manifest = write_stage0_fixture(tmp_path)
    dataset = Stage0Dataset.from_manifest(manifest)
    model = create_model("tiny")
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)

    metrics = evaluate(model, dataset, batch_size=4, device="cpu")
    assert metrics["samples"] == 16
    assert metrics["top1_target_accuracy"] == 1.0
    assert metrics["mean_illegal_probability"] == 0.0
    assert metrics["mean_target_probability"] == 1.0 / 16.0
    assert "by_category" in metrics


def test_stage0_train_and_eval_scripts_write_artifacts(tmp_path):
    train_manifest = write_stage0_fixture(tmp_path / "train", count=16)
    val_manifest = write_stage0_fixture(tmp_path / "val", count=8)
    out_dir = tmp_path / "out"

    train_main(
        [
            "--preset",
            "tiny",
            "--train-manifest",
            str(train_manifest),
            "--val-manifest",
            str(val_manifest),
            "--out",
            str(out_dir),
            "--batch-size",
            "4",
            "--epochs",
            "1",
            "--max-steps",
            "2",
            "--eval-every-steps",
            "1",
        ]
    )

    assert (out_dir / "model_state.pt").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "inference.ts").exists()
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir / "datasets.json").exists()

    eval_path = tmp_path / "eval.json"
    eval_main(
        [
            "--checkpoint",
            str(out_dir),
            "--manifest",
            str(val_manifest),
            "--batch-size",
            "4",
            "--out",
            str(eval_path),
        ]
    )
    assert json.loads(eval_path.read_text())["metrics"]["samples"] == 8
