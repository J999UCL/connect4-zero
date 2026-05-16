from __future__ import annotations

from pathlib import Path

import pytest

from c4zero_train import az_loop


def test_split_games_evenly_across_active_processes():
    assert az_loop.split_games(4000, 8) == [500] * 8
    assert az_loop.split_games(1003, 8) == [126, 126, 126, 125, 125, 125, 125, 125]
    assert az_loop.split_games(3, 8) == [1, 1, 1]


@pytest.mark.parametrize("total_games,processes", [(0, 8), (10, 0)])
def test_split_games_rejects_invalid_inputs(total_games, processes):
    with pytest.raises(ValueError):
        az_loop.split_games(total_games, processes)


def test_selfplay_shards_have_unique_dirs_seeds_and_manifests(tmp_path: Path):
    shards = az_loop.selfplay_shards(tmp_path / "round-70", total_games=10, processes=4, seed=9000)

    assert [shard.games for shard in shards] == [3, 3, 2, 2]
    assert [shard.seed for shard in shards] == [9000, 9001, 9002, 9003]
    assert [shard.out_dir.name for shard in shards] == [
        "selfplay-00",
        "selfplay-01",
        "selfplay-02",
        "selfplay-03",
    ]
    assert [shard.manifest_path for shard in shards] == [
        tmp_path / "round-70" / "selfplay-00" / "manifest.json",
        tmp_path / "round-70" / "selfplay-01" / "manifest.json",
        tmp_path / "round-70" / "selfplay-02" / "manifest.json",
        tmp_path / "round-70" / "selfplay-03" / "manifest.json",
    ]


def test_load_manifest_list_combines_direct_and_file_entries(tmp_path: Path):
    list_file = tmp_path / "manifests.txt"
    list_file.write_text(
        "\n".join(
            [
                "# older replay",
                "/tmp/data/round-01/manifest.json",
                "",
                "/tmp/data/round-02/manifest.json",
            ]
        ),
        encoding="utf-8",
    )

    assert az_loop.load_manifest_list(["/tmp/data/round-00/manifest.json"], [list_file]) == [
        "/tmp/data/round-00/manifest.json",
        "/tmp/data/round-01/manifest.json",
        "/tmp/data/round-02/manifest.json",
    ]


def test_parse_key_value_summary_handles_arena_output():
    parsed = az_loop.parse_key_value_summary(
        "games=64 model_a_wins=41 model_b_wins=23 draws=0 "
        "model_a_score_rate=0.640625 avg_plies=33.7031 player_a=model:/tmp/a"
    )

    assert parsed["games"] == "64"
    assert parsed["model_a_score_rate"] == "0.640625"
    assert parsed["player_a"] == "model:/tmp/a"


def test_train_command_passes_every_manifest(tmp_path: Path):
    parser_args = [
        "--c4zero-bin",
        "/tmp/c4zero",
        "--start-checkpoint",
        "/tmp/checkpoint",
        "--run-root",
        str(tmp_path / "run"),
        "--data-root",
        str(tmp_path / "data"),
        "--first-round",
        "70",
        "--rounds",
        "1",
        "--dry-run",
    ]
    # Exercise the real parser by using loop_main dry-run side effects instead of
    # duplicating parser construction in the test.
    az_loop.loop_main(parser_args)

    train_log = tmp_path / "run" / "round-70" / "train.log"
    text = train_log.read_text(encoding="utf-8")
    assert "--manifest" in text
    assert "selfplay-00/manifest.json" in text
    assert "selfplay-07/manifest.json" in text
