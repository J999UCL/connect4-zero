from __future__ import annotations

from pathlib import Path

from c4az.trace_demo import run_demo
from c4az.trace_tools import write_index, write_summary


def test_trace_demo_runs_deterministic_puct_search() -> None:
    summary = run_demo(simulations=6)

    assert summary.ply == 8
    assert summary.root_visits == 6
    assert sum(summary.trace_event_counts.values()) > 0
    assert summary.trace_event_counts["evaluate"] >= 1
    assert summary.evaluator_calls >= 1
    assert summary.positions_evaluated >= 1
    assert all(0 <= row["action"] < 16 for row in summary.top_actions)


def test_trace_index_and_summary_are_written(tmp_path: Path) -> None:
    summary_path = write_summary(tmp_path, simulations=2)
    index_path = write_index(
        tmp_path,
        {
            "viztracer": str(tmp_path / "viztracer" / "puct_demo.html"),
            "lectrace": str(tmp_path / "lectrace_site" / "index.html"),
            "summary": str(summary_path),
        },
    )

    assert summary_path.exists()
    assert index_path.exists()
    assert "VizTracer timeline" in index_path.read_text(encoding="utf-8")
