from io import StringIO

import pytest
import torch

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE, CURRENT_PLAYER
from connect4_zero.scripts.play_terminal import TerminalPlaySession, main
from connect4_zero.search import BatchedSearchResult, BatchedTreeMCTS, SearchTree, TreeMCTSConfig


class RecordingEvaluator:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.batch_sizes: list[int] = []

    def evaluate_batch(self, states: Connect4x4x4Batch) -> torch.Tensor:
        self.batch_sizes.append(states.batch_size)
        return torch.full((states.batch_size,), self.value, dtype=torch.float32, device=states.device)


def make_search(simulations: int = 8) -> BatchedTreeMCTS:
    return BatchedTreeMCTS(
        TreeMCTSConfig(
            simulations_per_root=simulations,
            max_leaf_batch_size=64,
            rollouts_per_leaf=1,
            rollout_device="cpu",
        ),
        evaluator=RecordingEvaluator(),
    )


def test_human_legal_move_updates_state() -> None:
    session = TerminalPlaySession(search=make_search())

    move = session.human_move(5)

    assert move.actor == "human"
    assert move.action == 5
    assert not move.won
    assert not move.draw
    assert session.engine_player == "bot"
    assert session.game.heights[0, 1, 1].item() == 1


def test_human_out_of_range_move_is_rejected_without_mutation() -> None:
    session = TerminalPlaySession(search=make_search())
    before = session.game.clone()
    before_tree = session.current_tree

    with pytest.raises(ValueError, match="between 0 and 15"):
        session.human_move(99)

    assert torch.equal(session.game.board, before.board)
    assert torch.equal(session.game.heights, before.heights)
    assert session.current_tree is before_tree
    assert session.engine_player == "human"


def test_human_full_column_move_is_rejected_without_mutation() -> None:
    session = TerminalPlaySession(search=make_search())
    session.game.heights[0, 0, 0] = 4
    before = session.game.clone()

    with pytest.raises(ValueError, match="not legal"):
        session.human_move(0)

    assert torch.equal(session.game.board, before.board)
    assert torch.equal(session.game.heights, before.heights)
    assert session.engine_player == "human"


def test_human_terminal_win_message_state() -> None:
    session = TerminalPlaySession(search=make_search())
    session.game.board[:, 0, 0, 0:3] = CURRENT_PLAYER
    session.game.heights[:, 0, 0] = 3

    move = session.human_move(0)

    assert move.won
    assert session.is_terminal
    assert session.winner == "human"
    assert "human won" in session.render_status()


def test_bot_move_uses_batched_tree_mcts_result_and_picks_legal_action() -> None:
    search = make_search(simulations=4)
    session = TerminalPlaySession(search=search, human_starts=False)

    move = session.bot_move()

    assert isinstance(session.search, BatchedTreeMCTS)
    assert move.actor == "bot"
    assert move.action == session.last_bot_action
    assert 0 <= move.action < 16
    assert move.action != 0
    assert search.last_trees
    assert session.last_bot_summary is not None
    assert session.last_bot_summary.visit_counts.sum().item() == 4
    assert session.engine_player == "human"


def test_bot_action_selection_breaks_visit_ties_by_q_then_center() -> None:
    session = TerminalPlaySession(search=make_search())
    visit_counts = torch.zeros((1, ACTION_SIZE), dtype=torch.float32)
    visit_counts[0, [0, 5, 10]] = 7
    q_values = torch.zeros((1, ACTION_SIZE), dtype=torch.float32)
    q_values[0, 5] = 0.25
    q_values[0, 10] = 0.25
    result = BatchedSearchResult(
        visit_counts=visit_counts,
        policy=visit_counts / visit_counts.sum(dim=1, keepdim=True),
        q_values=q_values,
        root_values=torch.zeros(1),
    )

    assert session._choose_bot_action(result) == 5


def test_subtree_reuse_advances_after_bot_and_human_moves() -> None:
    session = TerminalPlaySession(search=make_search(simulations=24), human_starts=False)

    bot_move = session.bot_move()

    assert bot_move.tree_reused
    assert session.current_tree is not None
    reusable_human_actions = [
        action
        for action, child in enumerate(session.current_tree.root.children)
        if child is not None
    ]
    assert reusable_human_actions

    human_move = session.human_move(reusable_human_actions[0])

    assert human_move.tree_reused
    assert session.current_tree is not None
    assert session.search.last_reused_roots == 0


def test_subtree_reuse_falls_back_when_child_was_not_expanded() -> None:
    session = TerminalPlaySession(search=make_search())
    session.current_tree = SearchTree.from_state(session.game.clone())

    move = session.human_move(0)

    assert not move.tree_reused
    assert session.current_tree is None


def test_play_terminal_cli_quit_smoke() -> None:
    output = StringIO()

    exit_code = main(
        [
            "--device",
            "cpu",
            "--simulations-per-root",
            "1",
            "--max-leaf-batch-size",
            "4",
            "--rollouts-per-leaf",
            "1",
            "--max-rollouts-per-chunk",
            "128",
        ],
        input_fn=lambda _: "quit",
        output=output,
    )

    assert exit_code == 0
    assert "4x4x4 Connect Four" in output.getvalue()
