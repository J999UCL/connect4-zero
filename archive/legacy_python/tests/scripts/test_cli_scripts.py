import json
from pathlib import Path

import numpy as np
import zstandard as zstd

from connect4_zero.data import SelfPlayDataset
from connect4_zero.model import Connect4ResNet3D, load_checkpoint, save_checkpoint
from connect4_zero.scripts.arena_eval import main as arena_eval_main
from connect4_zero.scripts.benchmark_search import main as benchmark_main
from connect4_zero.scripts.benchmark_puct import main as benchmark_puct_main
from connect4_zero.scripts.generate_selfplay import main as generate_main
from connect4_zero.scripts.run_training_loop import main as run_training_loop_main
from connect4_zero.scripts.train_resnet import main as train_resnet_main


def test_generate_selfplay_cli_writes_manifest_and_logs(tmp_path: Path) -> None:
    output_dir = tmp_path / "selfplay"

    exit_code = generate_main(
        [
            "--games",
            "1",
            "--out",
            str(output_dir),
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--games-per-write",
            "1",
            "--samples-per-shard",
            "4",
            "--simulations-per-root",
            "1",
            "--max-leaf-batch-size",
            "4",
            "--rollouts-per-leaf",
            "1",
            "--max-rollouts-per-chunk",
            "128",
            "--seed",
            "11",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "manifest.jsonl").exists()
    assert list((output_dir / "shards").glob("*.safetensors"))
    assert list((output_dir / "logs").glob("generate_selfplay-*.log"))


def test_benchmark_search_cli_writes_log(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"

    exit_code = benchmark_main(
        [
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--simulations-per-root",
            "1",
            "--max-leaf-batch-size",
            "4",
            "--rollouts-per-leaf",
            "1",
            "--max-rollouts-per-chunk",
            "128",
            "--seed",
            "11",
            "--log-dir",
            str(log_dir),
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert list(log_dir.glob("benchmark_search-*.log"))


def test_benchmark_puct_cli_writes_log(tmp_path: Path) -> None:
    log_dir = tmp_path / "puct_logs"

    exit_code = benchmark_puct_main(
        [
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--simulations-per-root",
            "1",
            "--max-leaf-batch-size",
            "1",
            "--inference-batch-size",
            "2",
            "--log-dir",
            str(log_dir),
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert list(log_dir.glob("benchmark_puct-*.log"))


def test_generate_selfplay_cli_accepts_puct_backend(tmp_path: Path) -> None:
    output_dir = tmp_path / "selfplay_puct"

    exit_code = generate_main(
        [
            "--backend",
            "puct",
            "--games",
            "1",
            "--out",
            str(output_dir),
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--games-per-write",
            "1",
            "--samples-per-shard",
            "128",
            "--simulations-per-root",
            "1",
            "--max-leaf-batch-size",
            "1",
            "--puct-inference-batch-size",
            "4",
            "--action-temperature",
            "0",
            "--seed",
            "11",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "manifest.jsonl").exists()
    assert list((output_dir / "shards").glob("*.safetensors"))


def test_generate_selfplay_cli_multiprocess_merges_worker_manifests(tmp_path: Path) -> None:
    output_dir = tmp_path / "selfplay_mp"

    exit_code = generate_main(
        [
            "--games",
            "2",
            "--out",
            str(output_dir),
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--games-per-write",
            "1",
            "--samples-per-shard",
            "128",
            "--simulations-per-root",
            "1",
            "--max-leaf-batch-size",
            "2",
            "--rollouts-per-leaf",
            "1",
            "--max-rollouts-per-chunk",
            "128",
            "--num-workers",
            "2",
            "--worker-start-method",
            "fork",
            "--seed",
            "11",
            "--quiet",
        ]
    )

    assert exit_code == 0
    manifest_path = output_dir / "manifest.jsonl"
    assert manifest_path.exists()
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert len(records) == 2
    assert all(record["shard"].startswith("workers/worker-") for record in records)
    assert list((output_dir / "workers").glob("worker-*/logs/generate_selfplay_worker_*.log"))
    dataset = SelfPlayDataset(manifest_path)
    assert len(dataset) == sum(record["num_samples"] for record in records)


def test_generate_selfplay_cli_puct_shared_inference_server(tmp_path: Path) -> None:
    output_dir = tmp_path / "selfplay_puct_server"

    exit_code = generate_main(
        [
            "--backend",
            "puct",
            "--games",
            "2",
            "--out",
            str(output_dir),
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--games-per-write",
            "1",
            "--samples-per-shard",
            "128",
            "--simulations-per-root",
            "1",
            "--max-leaf-batch-size",
            "1",
            "--puct-inference-batch-size",
            "4",
            "--puct-inference-mode",
            "server",
            "--puct-server-max-batch-size",
            "8",
            "--puct-server-batch-timeout-ms",
            "1",
            "--action-temperature",
            "0",
            "--num-workers",
            "2",
            "--worker-start-method",
            "fork",
            "--seed",
            "11",
            "--quiet",
        ]
    )

    assert exit_code == 0
    manifest_path = output_dir / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert len(records) == 2
    assert list((output_dir / "logs").glob("puct_inference_server-*.log"))


def test_train_resnet_cli_dry_run(tmp_path: Path) -> None:
    exit_code = train_resnet_main(
        [
            "--device",
            "cpu",
            "--out",
            str(tmp_path / "train"),
            "--dry-run",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert list((tmp_path / "train" / "logs").glob("train_resnet-*.log"))


def test_train_resnet_cli_accepts_rust_binary_dataset(tmp_path: Path) -> None:
    manifest = _write_tiny_rust_dataset(tmp_path / "rust_data")

    exit_code = train_resnet_main(
        [
            "--manifest",
            str(manifest),
            "--dataset-format",
            "rust-binary",
            "--device",
            "cpu",
            "--out",
            str(tmp_path / "train_rust"),
            "--batch-size",
            "1",
            "--max-new-steps",
            "1",
            "--num-workers",
            "0",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert list((tmp_path / "train_rust" / "checkpoints").glob("checkpoint-step-*.pt"))


def test_train_resnet_cli_resumes_with_max_new_steps(tmp_path: Path) -> None:
    manifest = _write_tiny_rust_dataset(tmp_path / "rust_data")
    resume = tmp_path / "resume.pt"
    save_checkpoint(resume, Connect4ResNet3D(), step=7, epoch=3)

    exit_code = train_resnet_main(
        [
            "--manifest",
            str(manifest),
            "--dataset-format",
            "rust-binary",
            "--resume",
            str(resume),
            "--device",
            "cpu",
            "--out",
            str(tmp_path / "train_resume"),
            "--batch-size",
            "1",
            "--max-new-steps",
            "1",
            "--num-workers",
            "0",
            "--quiet",
        ]
    )

    checkpoint = load_checkpoint(tmp_path / "train_resume" / "checkpoints" / "checkpoint-step-00000008.pt")
    assert exit_code == 0
    assert checkpoint.step == 8
    assert checkpoint.epoch == 3


def test_arena_eval_cli_writes_summary(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.pt"
    baseline = tmp_path / "baseline.pt"
    save_checkpoint(candidate, Connect4ResNet3D())
    save_checkpoint(baseline, Connect4ResNet3D())

    exit_code = arena_eval_main(
        [
            "--candidate-checkpoint",
            str(candidate),
            "--baseline-checkpoint",
            str(baseline),
            "--games",
            "2",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--simulations-per-root",
            "1",
            "--max-leaf-batch-size",
            "1",
            "--inference-batch-size",
            "4",
            "--opening-plies",
            "2",
            "--paired-openings",
            "--add-root-noise",
            "--action-temperature",
            "1.0",
            "--out",
            str(tmp_path / "arena"),
            "--quiet",
        ]
    )

    summary_path = tmp_path / "arena" / "arena-summary.json"
    summary = json.loads(summary_path.read_text())
    assert exit_code == 0
    assert summary["games"] == 2
    assert summary["opening_plies"] == 2
    assert summary["paired_openings"] is True
    assert summary["unique_openings"] == 1
    assert summary["add_root_noise"] is True
    assert summary["action_temperature"] == 1.0
    assert summary["candidate_wins"] + summary["baseline_wins"] + summary["draws"] == 2
    assert list((tmp_path / "arena" / "logs").glob("arena_eval-*.log"))


def test_run_training_loop_dry_run_writes_plan(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, Connect4ResNet3D())

    exit_code = run_training_loop_main(
        [
            "--initial-checkpoint",
            str(checkpoint),
            "--run-root",
            str(tmp_path / "runs"),
            "--data-root",
            str(tmp_path / "data"),
            "--run-prefix",
            "dry",
            "--rounds",
            "2",
            "--device",
            "cpu",
            "--games-per-round",
            "2",
            "--arena-games",
            "2",
            "--arena-opening-plies",
            "2",
            "--arena-paired-openings",
            "--arena-add-root-noise",
            "--arena-action-temperature",
            "1.0",
            "--dry-run",
            "--quiet",
        ]
    )

    plan_path = tmp_path / "runs" / "dry" / "loop-plan.json"
    plan = json.loads(plan_path.read_text())
    assert exit_code == 0
    assert plan["run_id"] == "dry"
    assert len(plan["rounds"]) == 2
    assert "--paired-openings" in plan["rounds"][0]["arena_template"]
    assert "--add-root-noise" in plan["rounds"][0]["arena_template"]
    assert list((tmp_path / "runs" / "dry" / "logs").glob("run_training_loop-*.log"))


def _write_tiny_rust_dataset(root: Path) -> Path:
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True)
    shard_path = shard_dir / "shard-000000.c4zst"

    payload = bytearray()
    payload.extend(b"C4ZRS001")
    payload.extend(np.array([1], dtype="<u8").tobytes())
    payload.extend(np.array([1], dtype="<u8").tobytes())
    payload.extend(np.array([0], dtype="<u8").tobytes())
    heights = np.zeros((1, 16), dtype="u1")
    heights[0, 0] = 1
    payload.extend(heights.tobytes())
    policy = np.zeros((1, 16), dtype="<f4")
    policy[0, 4] = 1.0
    payload.extend(policy.tobytes())
    payload.extend(np.array([1], dtype="i1").tobytes())
    visits = np.zeros((1, 16), dtype="<u4")
    visits[0, 4] = 9
    payload.extend(visits.tobytes())
    q_values = np.zeros((1, 16), dtype="<f4")
    q_values[0, 4] = 0.75
    payload.extend(q_values.tobytes())
    payload.extend(np.array([0xFFFE], dtype="<u2").tobytes())
    payload.extend(np.array([4], dtype="u1").tobytes())
    payload.extend(np.array([0], dtype="u1").tobytes())
    shard_path.write_bytes(zstd.ZstdCompressor().compress(bytes(payload)))

    manifest = {
        "format": "connect4_zero.rust.selfplay.v1",
        "format_version": 1,
        "samples": 1,
        "generator": "test",
        "model_path": None,
        "shards": [{"path": "shards/shard-000000.c4zst", "samples": 1, "compression": "zstd"}],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path
