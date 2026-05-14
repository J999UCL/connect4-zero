from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from c4az.game import ACTION_SIZE, Position, action_z_to_bit, initial_position
from c4az.mcts import MCTSConfig, NetworkOutput, PUCTMCTS, masked_softmax, visit_counts_to_policy


class FakeEvaluator:
    def __init__(self, *, default_value: float = 0.0) -> None:
        self.default_value = default_value
        self.values: dict[tuple[int, int, int], float] = {}
        self.logits: dict[tuple[int, int, int], np.ndarray] = {}
        self.calls = 0
        self.positions_evaluated = 0

    def set_value(self, position: Position, value: float) -> None:
        self.values[_key(position)] = value

    def set_prior(self, position: Position, action: int, strength: float = 8.0) -> None:
        logits = np.zeros(ACTION_SIZE, dtype=np.float32)
        logits[action] = strength
        self.logits[_key(position)] = logits

    def evaluate(self, positions: list[Position]) -> list[NetworkOutput]:
        self.calls += 1
        self.positions_evaluated += len(positions)
        outputs = []
        for position in positions:
            key = _key(position)
            outputs.append(
                NetworkOutput(
                    policy_logits=self.logits.get(key, np.zeros(ACTION_SIZE, dtype=np.float32)),
                    value=self.values.get(key, self.default_value),
                )
            )
        return outputs


def _key(position: Position) -> tuple[int, int, int]:
    return position.current, position.opponent, position.ply


def test_golden_ledger_for_first_two_simulations() -> None:
    evaluator = FakeEvaluator()
    root_position = initial_position()
    child0 = root_position.play(0)
    child1 = root_position.play(1)
    evaluator.set_value(child0, 0.25)
    evaluator.set_value(child1, -0.5)
    mcts = PUCTMCTS(evaluator, MCTSConfig(simulations_per_move=2, c_puct=1.5, trace=True))

    result = mcts.search(root_position)

    selected = [event["action"] for event in result.trace if event["event"] == "select"]
    assert selected == [0, 1]
    assert result.root.visits == 2
    assert result.root.children[0].visits == 1
    assert result.root.children[0].mean_value == 0.25
    assert result.root.children[1].mean_value == -0.5
    assert result.root.mean_value == 0.125
    assert result.visit_counts.sum() == 2


def test_terminal_child_bypasses_evaluator() -> None:
    current = sum(action_z_to_bit(action, 0) for action in (1, 2, 3))
    root = Position(current=current, heights=(0, 1, 1, 1, *([0] * 12)), ply=3)
    evaluator = FakeEvaluator()
    evaluator.set_prior(root, 0)
    mcts = PUCTMCTS(evaluator, MCTSConfig(simulations_per_move=1))

    result = mcts.search(root)

    assert evaluator.positions_evaluated == 1
    assert result.root.children[0].position.is_terminal
    assert result.q_values[0] == 1.0


def test_immediate_win_gets_dominant_visits() -> None:
    current = sum(action_z_to_bit(action, 0) for action in (1, 2, 3))
    root = Position(current=current, heights=(0, 1, 1, 1, *([0] * 12)), ply=3)
    mcts = PUCTMCTS(FakeEvaluator(), MCTSConfig(simulations_per_move=32))

    result = mcts.search(root)

    assert int(result.visit_counts.argmax()) == 0
    assert result.visit_counts[0] > 16


def test_immediate_opponent_win_is_blocked() -> None:
    opponent = sum(action_z_to_bit(action, 0) for action in (1, 2, 3))
    root = Position(opponent=opponent, heights=(0, 1, 1, 1, *([0] * 12)), ply=3)
    mcts = PUCTMCTS(FakeEvaluator(), MCTSConfig(simulations_per_move=96))

    result = mcts.search(root)

    assert int(result.visit_counts.argmax()) == 0
    assert result.visit_counts[0] > 32


def test_non_playable_gravity_threat_is_not_treated_as_immediate() -> None:
    opponent = sum(action_z_to_bit(action, 1) for action in (0, 1, 2))
    root = Position(opponent=opponent, heights=(2, 2, 2, 0, *([0] * 12)), ply=6)
    mcts = PUCTMCTS(FakeEvaluator(), MCTSConfig(simulations_per_move=32))

    result = mcts.search(root)

    assert result.root.children[3] is not None
    assert not result.root.children[3].position.is_terminal
    assert result.q_values[3] == 0.0


def test_high_priors_dominate_when_values_are_equal() -> None:
    root = initial_position()
    evaluator = FakeEvaluator()
    evaluator.set_prior(root, 5)
    mcts = PUCTMCTS(evaluator, MCTSConfig(simulations_per_move=32))

    result = mcts.search(root)

    assert int(result.visit_counts.argmax()) == 5


def test_high_values_can_overcome_priors() -> None:
    current = sum(action_z_to_bit(action, 0) for action in (0, 1, 3))
    root = Position(current=current, heights=(1, 1, 0, 1, *([0] * 12)), ply=3)
    evaluator = FakeEvaluator()
    evaluator.set_prior(root, 5, strength=1.5)
    mcts = PUCTMCTS(evaluator, MCTSConfig(simulations_per_move=128))

    result = mcts.search(root)

    assert int(result.visit_counts.argmax()) == 2


def test_temperature_policy_conversion() -> None:
    counts = np.array([10, 5, 0, *([0] * 13)], dtype=np.uint32)
    mask = (1 << 0) | (1 << 1) | (1 << 2)

    tau1 = visit_counts_to_policy(counts, 1.0, mask)
    tau0 = visit_counts_to_policy(counts, 0.0, mask)
    tau_half = visit_counts_to_policy(counts, 0.5, mask)

    assert np.allclose(tau1[:3], [2 / 3, 1 / 3, 0])
    assert tau0[0] == 1.0
    assert tau_half[0] > tau1[0]


def test_root_noise_changes_only_root_priors() -> None:
    root = initial_position()
    evaluator = FakeEvaluator()
    base = PUCTMCTS(evaluator, MCTSConfig(simulations_per_move=1, seed=7)).search(root, add_root_noise=False)
    noisy = PUCTMCTS(evaluator, MCTSConfig(simulations_per_move=1, seed=7)).search(root, add_root_noise=True)

    assert not np.allclose(base.root.priors, noisy.root.priors)
    assert np.isclose(noisy.root.priors.sum(), 1.0)


def test_subtree_reuse_preserves_child_visits() -> None:
    root = initial_position()
    mcts = PUCTMCTS(FakeEvaluator(), MCTSConfig(simulations_per_move=8))
    result = mcts.search(root)
    action = int(result.visit_counts.argmax())
    child_visits = result.root.children[action].visits

    child = mcts.advance_tree(result.root, action)

    assert child is not None
    assert child.parent is None
    assert child.visits == child_visits


def test_reference_mcts_matches_production_on_seeded_position() -> None:
    root = initial_position().play(0).play(5)
    evaluator = FakeEvaluator(default_value=0.1)
    evaluator.set_prior(root, 7, strength=2.0)
    config = MCTSConfig(simulations_per_move=12, c_puct=1.5)

    production = PUCTMCTS(evaluator, config).search(root)
    reference_visits = _reference_search(root, evaluator, config)

    assert production.visit_counts.tolist() == reference_visits.tolist()


@dataclass
class RefNode:
    position: Position
    priors: np.ndarray = field(default_factory=lambda: np.zeros(ACTION_SIZE, dtype=np.float32))
    legal_mask: int = 0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "RefNode"] = field(default_factory=dict)
    expanded: bool = False

    @property
    def mean(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def _reference_search(position: Position, evaluator: FakeEvaluator, config: MCTSConfig) -> np.ndarray:
    root = RefNode(position)
    _ref_expand(root, evaluator)
    for _ in range(config.simulations_per_move):
        path = [root]
        node = root
        while node.expanded and not node.position.is_terminal:
            action = _ref_select(node, config.c_puct)
            if action not in node.children:
                node.children[action] = RefNode(node.position.play(action))
            node = node.children[action]
            path.append(node)
            if not node.expanded:
                break
        value = node.position.terminal_value if node.position.is_terminal else _ref_expand(node, evaluator)
        for visited in reversed(path):
            visited.visits += 1
            visited.value_sum += value
            value = -value
    visits = np.zeros(ACTION_SIZE, dtype=np.uint32)
    for action, child in root.children.items():
        visits[action] = child.visits
    return visits


def _ref_expand(node: RefNode, evaluator: FakeEvaluator) -> float:
    output = evaluator.evaluate([node.position])[0]
    node.legal_mask = node.position.legal_mask()
    node.priors = _local_masked_softmax(output.policy_logits, node.legal_mask)
    node.expanded = True
    return output.value


def _ref_select(node: RefNode, c_puct: float) -> int:
    best_action = -1
    best_score = -math.inf
    total = sum(child.visits for child in node.children.values())
    for action in range(ACTION_SIZE):
        if not (node.legal_mask & (1 << action)):
            continue
        child = node.children.get(action)
        visits = child.visits if child is not None else 0
        q = -child.mean if child is not None and child.visits else 0.0
        score = q + c_puct * node.priors[action] * math.sqrt(total) / (1 + visits)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def _local_masked_softmax(logits: np.ndarray, legal_mask: int) -> np.ndarray:
    priors = np.zeros(ACTION_SIZE, dtype=np.float32)
    legal = [a for a in range(ACTION_SIZE) if legal_mask & (1 << a)]
    xs = np.array([logits[a] for a in legal], dtype=np.float64)
    xs -= xs.max()
    probs = np.exp(xs) / np.exp(xs).sum()
    for action, prob in zip(legal, probs):
        priors[action] = prob
    return priors
