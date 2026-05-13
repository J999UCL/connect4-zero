from pathlib import Path

from connect4_zero.scripts.benchmark_search import main as benchmark_main
from connect4_zero.scripts.benchmark_puct import main as benchmark_puct_main
from connect4_zero.scripts.generate_selfplay import main as generate_main
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
