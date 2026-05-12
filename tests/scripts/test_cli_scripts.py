from pathlib import Path

from connect4_zero.scripts.benchmark_search import main as benchmark_main
from connect4_zero.scripts.generate_selfplay import main as generate_main


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
            "--num-selection-waves",
            "0",
            "--leaves-per-root",
            "1",
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
            "--num-selection-waves",
            "0",
            "--leaves-per-root",
            "1",
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
