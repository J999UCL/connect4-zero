import pytest
import torch

from connect4_zero.game import Connect4x4x4Batch
from connect4_zero.game.constants import ACTION_SIZE, BOARD_SIZE, CURRENT_PLAYER, OPPONENT_PLAYER
from connect4_zero.search import BatchedPUCTMCTS, PUCTMCTSConfig, PUCTSearchTree, PolicyValueBatch
from connect4_zero.search.puct_tree import PUCTNode


class FakePolicyValueEvaluator:
    def __init__(self, value: float = 0.0, preferred_action: int | None = None) -> None:
        self.value = value
        self.preferred_action = preferred_action
        self.batch_sizes: list[int] = []

    def evaluate_batch(self, states: Connect4x4x4Batch) -> PolicyValueBatch:
        self.batch_sizes.append(states.batch_size)
        legal = states.legal_mask()
        priors = legal.to(dtype=torch.float32)
        if self.preferred_action is not None:
            priors = priors * 0.01
            priors[:, self.preferred_action] = torch.where(
                legal[:, self.preferred_action],
                torch.ones(states.batch_size, dtype=torch.float32, device=states.device),
                torch.zeros(states.batch_size, dtype=torch.float32, device=states.device),
            )
        priors = priors / priors.sum(dim=1, keepdim=True).clamp_min(1.0)
        values = torch.full((states.batch_size,), self.value, dtype=torch.float32, device=states.device)
        return PolicyValueBatch(priors=priors, values=values)


def immediate_win_root() -> Connect4x4x4Batch:
    game = Connect4x4x4Batch(batch_size=1)
    game.board[:, 0, 0, 0:3] = CURRENT_PLAYER
    game.heights[:, 0, 0] = 3
    return game


def opponent_immediate_threat_root() -> Connect4x4x4Batch:
    game = Connect4x4x4Batch(batch_size=1)
    game.board[:, 0, 0, 0:3] = OPPONENT_PLAYER
    game.heights[:, 0, 0] = 3
    return game


def constrained_two_action_root() -> Connect4x4x4Batch:
    game = Connect4x4x4Batch(batch_size=1)
    game.heights[:] = BOARD_SIZE
    game.heights[:, 0, 0] = 0
    game.heights[:, 0, 1] = 0
    return game


def test_puct_visit_counts_sum_to_simulations_per_root() -> None:
    roots = Connect4x4x4Batch(batch_size=2)
    search = BatchedPUCTMCTS(
        evaluator=FakePolicyValueEvaluator(),
        config=PUCTMCTSConfig(simulations_per_root=12, max_leaf_batch_size=4),
    )

    result = search.search_batch(roots)

    assert result.visit_counts.shape == (2, ACTION_SIZE)
    assert torch.all(result.visit_counts.sum(dim=1).eq(12))
    assert torch.allclose(result.policy.sum(dim=1), torch.ones(2))
    assert search.last_new_visits_added == [12, 12]
    assert search.last_depth_histogram


def test_puct_backprop_flips_values_through_depth_five() -> None:
    state = Connect4x4x4Batch(batch_size=1)
    path = [
        PUCTNode(state=state.clone(), parent=None, parent_action=None, legal_actions=()),
    ]
    for depth in range(1, 6):
        path.append(
            PUCTNode(
                state=state.clone(),
                parent=path[-1],
                parent_action=0,
                legal_actions=(),
                depth=depth,
            )
        )
    search = BatchedPUCTMCTS(FakePolicyValueEvaluator(), PUCTMCTSConfig(simulations_per_root=1))

    search._backpropagate(path, 0.5)

    assert path[-1].mean_value == pytest.approx(0.5)
    assert path[-2].mean_value == pytest.approx(-0.5)
    assert path[-3].mean_value == pytest.approx(0.5)
    assert path[0].mean_value == pytest.approx(-0.5)


def test_puct_immediate_current_player_win_gets_dominant_visits() -> None:
    search = BatchedPUCTMCTS(
        evaluator=FakePolicyValueEvaluator(value=0.0),
        config=PUCTMCTSConfig(simulations_per_root=48, max_leaf_batch_size=1, c_puct=1.5),
    )

    result = search.search_batch(immediate_win_root())

    assert result.policy[0].argmax().item() == 0
    assert result.q_values[0, 0].item() == pytest.approx(1.0)
    assert result.visit_counts[0, 0].item() > result.visit_counts[0, 1:].max().item()


def test_puct_blocks_playable_opponent_immediate_win() -> None:
    search = BatchedPUCTMCTS(
        evaluator=FakePolicyValueEvaluator(value=0.0),
        config=PUCTMCTSConfig(simulations_per_root=160, max_leaf_batch_size=1, c_puct=1.5),
    )

    result = search.search_batch(opponent_immediate_threat_root())

    assert result.policy[0].argmax().item() == 0
    assert result.visit_counts[0, 0].item() > result.visit_counts[0, 1:].max().item()


def test_puct_does_not_treat_non_playable_three_as_immediate_threat() -> None:
    game = Connect4x4x4Batch(batch_size=1)
    game.board[:, 0:3, 0, 1] = OPPONENT_PLAYER
    game.board[:, 0:3, 0, 0] = OPPONENT_PLAYER
    game.heights[:, 0:3, 0] = 2

    candidate = game.clone()
    step = candidate.step(torch.tensor([12], dtype=torch.long))

    assert bool(step.legal[0].item())
    assert not bool(step.won[0].item())


def test_puct_follows_high_fake_prior_when_values_are_equal() -> None:
    search = BatchedPUCTMCTS(
        evaluator=FakePolicyValueEvaluator(value=0.0, preferred_action=7),
        config=PUCTMCTSConfig(simulations_per_root=32, max_leaf_batch_size=4, c_puct=1.5),
    )

    result = search.search_batch(Connect4x4x4Batch(batch_size=1))

    assert result.policy[0].argmax().item() == 7


def test_puct_root_dirichlet_noise_only_changes_when_enabled() -> None:
    roots = Connect4x4x4Batch(batch_size=1)
    no_noise = BatchedPUCTMCTS(
        evaluator=FakePolicyValueEvaluator(),
        config=PUCTMCTSConfig(simulations_per_root=1, add_root_noise=False, seed=1),
    )
    with_noise = BatchedPUCTMCTS(
        evaluator=FakePolicyValueEvaluator(),
        config=PUCTMCTSConfig(simulations_per_root=1, add_root_noise=True, seed=1),
    )

    no_noise.search_batch(roots)
    with_noise.search_batch(roots)

    assert not torch.allclose(no_noise.last_trees[0].root.child_priors, with_noise.last_trees[0].root.child_priors)
    assert torch.isclose(with_noise.last_trees[0].root.child_priors.sum(), torch.tensor(1.0))


def test_puct_subtree_reuse_adds_fresh_visits() -> None:
    root = Connect4x4x4Batch(batch_size=1)
    search = BatchedPUCTMCTS(
        evaluator=FakePolicyValueEvaluator(value=0.0),
        config=PUCTMCTSConfig(simulations_per_root=12, max_leaf_batch_size=4),
    )
    result = search.search_batch(root)
    action = int(result.policy[0].argmax().item())
    reused = search.advance_tree(search.last_trees[0], action)
    assert reused is not None
    previous_visits = reused.root.visits
    next_root = root.clone()
    next_root.step(torch.tensor([action], dtype=torch.long))

    search.search_batch_with_trees(next_root, [reused])

    assert search.last_reused_roots == 1
    assert search.last_new_visits_added == [12]
    assert search.last_trees[0].root.visits == previous_visits + 12


def test_puct_depth_reaches_five_on_constrained_branching_fixture() -> None:
    search = BatchedPUCTMCTS(
        evaluator=FakePolicyValueEvaluator(value=0.0),
        config=PUCTMCTSConfig(simulations_per_root=80, max_leaf_batch_size=1, c_puct=1.5),
    )

    search.search_batch(constrained_two_action_root())

    assert search.last_trees[0].max_depth >= 5
