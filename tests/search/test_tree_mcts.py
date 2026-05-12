import torch
import pytest

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE, BOARD_SIZE, CURRENT_PLAYER
from connect4_zero.search import BatchedTreeMCTS, SearchTree, TreeMCTSConfig


class RecordingEvaluator:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.batch_sizes: list[int] = []

    def evaluate_batch(self, states: Connect4x4x4Batch) -> torch.Tensor:
        assert not bool(states.done.any().item())
        self.batch_sizes.append(states.batch_size)
        return torch.full((states.batch_size,), self.value, dtype=torch.float32, device=states.device)


def immediate_win_root(batch_size: int = 1) -> Connect4x4x4Batch:
    game = Connect4x4x4Batch(batch_size=batch_size)
    game.board[:, 0, 0, 0:3] = CURRENT_PLAYER
    game.heights[:, 0, 0] = 3
    return game


def test_tree_root_has_correct_legal_actions() -> None:
    tree = SearchTree.from_state(Connect4x4x4Batch(batch_size=1))

    assert tree.root.legal_actions == tuple(range(ACTION_SIZE))
    assert tree.root.visits == 0
    assert tree.root.children == [None for _ in range(ACTION_SIZE)]


def test_expanded_child_state_matches_engine_step() -> None:
    root = Connect4x4x4Batch(batch_size=1)
    tree = SearchTree.from_state(root)

    child = tree.expand_child(tree.root, 5)
    expected = root.clone()
    expected.step(torch.tensor([5], dtype=torch.long))

    assert torch.equal(child.state.board, expected.board)
    assert torch.equal(child.state.heights, expected.heights)
    assert torch.equal(child.state.done, expected.done)
    assert torch.equal(child.state.outcome, expected.outcome)


def test_terminal_child_is_marked_with_exact_value() -> None:
    tree = SearchTree.from_state(immediate_win_root())

    child = tree.expand_child(tree.root, 0)

    assert child.terminal_value == -1.0
    assert child.is_terminal


def test_backprop_flips_values_across_depth_two_and_three() -> None:
    tree = SearchTree.from_state(Connect4x4x4Batch(batch_size=1))
    child = tree.expand_child(tree.root, 0)
    grandchild = tree.expand_child(child, 1)
    search = BatchedTreeMCTS(TreeMCTSConfig(simulations_per_root=1), evaluator=RecordingEvaluator())

    search._backpropagate([tree.root, child, grandchild], 0.75)

    assert grandchild.mean_value == pytest.approx(0.75)
    assert child.mean_value == pytest.approx(-0.75)
    assert tree.root.mean_value == pytest.approx(0.75)

    tree = SearchTree.from_state(Connect4x4x4Batch(batch_size=1))
    child = tree.expand_child(tree.root, 0)
    grandchild = tree.expand_child(child, 1)
    great_grandchild = tree.expand_child(grandchild, 2)

    search._backpropagate([tree.root, child, grandchild, great_grandchild], 0.25)

    assert great_grandchild.mean_value == pytest.approx(0.25)
    assert grandchild.mean_value == pytest.approx(-0.25)
    assert child.mean_value == pytest.approx(0.25)
    assert tree.root.mean_value == pytest.approx(-0.25)


def test_virtual_loss_diversifies_pending_selection_after_root_expansion() -> None:
    tree = SearchTree.from_state(Connect4x4x4Batch(batch_size=1))
    for action in tree.root.legal_actions:
        child = tree.expand_child(tree.root, action)
        child.visits = 1
    tree.root.visits = len(tree.root.legal_actions)
    search = BatchedTreeMCTS(
        TreeMCTSConfig(simulations_per_root=2, virtual_loss=1.0),
        evaluator=RecordingEvaluator(),
    )

    first = search._select_path(0, tree)
    search._apply_virtual_loss(first.path)
    second = search._select_path(0, tree)

    assert first.path[1].parent_action != second.path[1].parent_action


def test_search_batch_returns_deep_policy_statistics_for_many_roots() -> None:
    roots = Connect4x4x4Batch(batch_size=2)
    evaluator = RecordingEvaluator(value=0.0)
    search = BatchedTreeMCTS(
        TreeMCTSConfig(
            simulations_per_root=20,
            max_leaf_batch_size=8,
            rollouts_per_leaf=2,
            seed=7,
        ),
        evaluator=evaluator,
    )

    result = search.search_batch(roots)

    assert result.visit_counts.shape == (2, ACTION_SIZE)
    assert result.policy.shape == (2, ACTION_SIZE)
    assert result.q_values.shape == (2, ACTION_SIZE)
    assert result.root_values.shape == (2,)
    assert torch.all(result.visit_counts.sum(dim=1).eq(20))
    assert torch.allclose(result.policy.sum(dim=1), torch.ones(2))
    assert sum(evaluator.batch_sizes) == 40
    assert max(evaluator.batch_sizes) <= 8


def test_search_builds_nodes_deeper_than_root_children() -> None:
    evaluator = RecordingEvaluator(value=0.0)
    search = BatchedTreeMCTS(
        TreeMCTSConfig(
            simulations_per_root=24,
            max_leaf_batch_size=64,
            rollouts_per_leaf=1,
        ),
        evaluator=evaluator,
    )

    search.search_batch(Connect4x4x4Batch(batch_size=1))

    assert search.last_trees[0].max_depth > 1


def test_terminal_nodes_are_not_sent_to_rollout_evaluator() -> None:
    evaluator = RecordingEvaluator(value=0.0)
    search = BatchedTreeMCTS(
        TreeMCTSConfig(simulations_per_root=1, max_leaf_batch_size=4, rollouts_per_leaf=1),
        evaluator=evaluator,
    )

    result = search.search_batch(immediate_win_root())

    assert evaluator.batch_sizes == []
    assert search.last_terminal_evaluations == 1
    assert result.visit_counts[0, 0].item() == 1.0
    assert result.policy[0, 0].item() == 1.0
    assert result.q_values[0, 0].item() == pytest.approx(1.0)
    assert result.root_values[0].item() == pytest.approx(1.0)


def test_full_columns_receive_zero_policy_probability() -> None:
    roots = Connect4x4x4Batch(batch_size=2)
    roots.heights[:, 0, 0] = BOARD_SIZE
    search = BatchedTreeMCTS(
        TreeMCTSConfig(simulations_per_root=8, max_leaf_batch_size=8, rollouts_per_leaf=1),
        evaluator=RecordingEvaluator(),
    )

    result = search.search_batch(roots)

    assert torch.all(result.visit_counts[:, 0].eq(0))
    assert torch.all(result.policy[:, 0].eq(0))
    assert torch.allclose(result.policy.sum(dim=1), torch.ones(2))


def test_terminal_roots_return_zero_policy_and_terminal_value() -> None:
    roots = Connect4x4x4Batch(batch_size=2)
    roots.done[:] = True
    roots.outcome[0] = 1
    roots.outcome[1] = 0
    evaluator = RecordingEvaluator()
    search = BatchedTreeMCTS(
        TreeMCTSConfig(simulations_per_root=8, max_leaf_batch_size=8, rollouts_per_leaf=1),
        evaluator=evaluator,
    )

    result = search.search_batch(roots)

    assert evaluator.batch_sizes == []
    assert torch.all(result.policy.eq(0))
    assert result.root_values.tolist() == [-1.0, 0.0]
