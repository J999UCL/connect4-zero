from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from c4az.game import ACTION_SIZE, Position, encode_positions


@dataclass(frozen=True, slots=True)
class NetworkOutput:
    policy_logits: np.ndarray
    value: float


class Evaluator(Protocol):
    def evaluate(self, positions: list[Position]) -> list[NetworkOutput]:
        ...


class TorchEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device | str = "cpu",
        batch_size: int = 256,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.calls = 0
        self.positions_evaluated = 0

    def evaluate(self, positions: list[Position]) -> list[NetworkOutput]:
        outputs: list[NetworkOutput] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(positions), self.batch_size):
                chunk = positions[start : start + self.batch_size]
                planes = encode_positions(chunk, device=self.device)
                logits, values = self.model(planes)
                logits_np = logits.detach().cpu().numpy().astype(np.float32)
                values_np = values.detach().cpu().numpy().astype(np.float32)
                outputs.extend(
                    NetworkOutput(policy_logits=logits_np[i], value=float(values_np[i]))
                    for i in range(len(chunk))
                )
        self.calls += 1
        self.positions_evaluated += len(positions)
        return outputs


class UniformEvaluator:
    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)
        self.calls = 0
        self.positions_evaluated = 0

    def evaluate(self, positions: list[Position]) -> list[NetworkOutput]:
        self.calls += 1
        self.positions_evaluated += len(positions)
        logits = np.zeros(ACTION_SIZE, dtype=np.float32)
        return [NetworkOutput(policy_logits=logits.copy(), value=self.value) for _ in positions]


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    simulations_per_move: int = 128
    c_puct: float = 1.5
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    temperature: float = 1.0
    seed: int | None = None
    trace: bool = False


@dataclass(slots=True)
class Node:
    position: Position
    parent: "Node | None" = None
    parent_action: int | None = None
    prior: float = 1.0
    children: list["Node | None"] = field(default_factory=lambda: [None] * ACTION_SIZE)
    priors: np.ndarray = field(default_factory=lambda: np.zeros(ACTION_SIZE, dtype=np.float32))
    legal_mask: int = 0
    visits: int = 0
    value_sum: float = 0.0
    terminal_value: float | None = None
    expanded: bool = False

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def child_visits(self) -> np.ndarray:
        return np.array([child.visits if child is not None else 0 for child in self.children], dtype=np.uint32)


@dataclass(frozen=True, slots=True)
class SearchResult:
    root: Node
    policy: np.ndarray
    visit_counts: np.ndarray
    q_values: np.ndarray
    root_value: float
    trace: list[dict]


class PUCTMCTS:
    def __init__(self, evaluator: Evaluator, config: MCTSConfig | None = None) -> None:
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self._rng = random.Random(self.config.seed)
        self._np_rng = np.random.default_rng(self.config.seed)
        self.last_trace: list[dict] = []

    def search(
        self,
        position: Position,
        *,
        root: Node | None = None,
        add_root_noise: bool = False,
        temperature: float | None = None,
    ) -> SearchResult:
        if root is None or root.position != position:
            root = Node(position=position, terminal_value=position.terminal_value)
        self.last_trace = []
        if not root.expanded and not root.position.is_terminal:
            self._expand(root)
        if add_root_noise and root.expanded:
            self._add_root_noise(root)

        for simulation in range(self.config.simulations_per_move):
            leaf_path = self._select(root, simulation)
            leaf = leaf_path[-1]
            if leaf.position.is_terminal:
                value = float(leaf.position.terminal_value)
                leaf.terminal_value = value
                self._trace("terminal", simulation=simulation, path=self._path_actions(leaf_path), value=value)
            else:
                value = self._expand(leaf)
                self._trace("evaluate", simulation=simulation, path=self._path_actions(leaf_path), value=value)
            self._backup(leaf_path, value)

        visits = root.child_visits()
        tau = self.config.temperature if temperature is None else temperature
        policy = visit_counts_to_policy(visits, tau, root.legal_mask)
        q_values = self._root_q_values(root)
        return SearchResult(
            root=root,
            policy=policy,
            visit_counts=visits,
            q_values=q_values,
            root_value=root.mean_value,
            trace=list(self.last_trace),
        )

    def advance_tree(self, root: Node, action: int) -> Node | None:
        child = root.children[action]
        if child is None:
            return None
        child.parent = None
        child.parent_action = None
        return child

    def _select(self, root: Node, simulation: int) -> list[Node]:
        node = root
        path = [node]
        while node.expanded and not node.position.is_terminal:
            action = self._select_action(node)
            child = node.children[action]
            if child is None:
                child = Node(
                    position=node.position.play(action),
                    parent=node,
                    parent_action=action,
                    prior=float(node.priors[action]),
                    terminal_value=None,
                )
                node.children[action] = child
            path.append(child)
            self._trace("select", simulation=simulation, action=action, score=self._score(node, action))
            if not child.expanded:
                break
            node = child
        return path

    def _select_action(self, node: Node) -> int:
        legal_mask = node.legal_mask
        best_action = -1
        best_score = -float("inf")
        for action in range(ACTION_SIZE):
            if not (legal_mask & (1 << action)):
                continue
            score = self._score(node, action)
            if score > best_score:
                best_score = score
                best_action = action
        if best_action < 0:
            raise RuntimeError("no legal action available during selection")
        return best_action

    def _score(self, node: Node, action: int) -> float:
        child = node.children[action]
        visits = child.visits if child is not None else 0
        q = -child.mean_value if child is not None and child.visits else 0.0
        total_visits = sum(c.visits for c in node.children if c is not None)
        u = self.config.c_puct * float(node.priors[action]) * math.sqrt(total_visits) / (1 + visits)
        return q + u

    def _expand(self, node: Node) -> float:
        if node.position.is_terminal:
            node.terminal_value = float(node.position.terminal_value)
            return node.terminal_value
        output = self.evaluator.evaluate([node.position])[0]
        node.legal_mask = node.position.legal_mask()
        node.priors = masked_softmax(output.policy_logits, node.legal_mask)
        node.expanded = True
        return float(output.value)

    def _backup(self, path: list[Node], leaf_value: float) -> None:
        value = float(leaf_value)
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            self._trace("backup", action=node.parent_action, visits=node.visits, value=value, mean=node.mean_value)
            value = -value

    def _root_q_values(self, root: Node) -> np.ndarray:
        q = np.zeros(ACTION_SIZE, dtype=np.float32)
        for action, child in enumerate(root.children):
            if child is not None and child.visits:
                q[action] = -child.mean_value
        return q

    def _add_root_noise(self, root: Node) -> None:
        legal = [a for a in range(ACTION_SIZE) if root.legal_mask & (1 << a)]
        if not legal:
            return
        noise = self._np_rng.dirichlet([self.config.root_dirichlet_alpha] * len(legal))
        mixed = root.priors.copy()
        for i, action in enumerate(legal):
            mixed[action] = (
                (1.0 - self.config.root_exploration_fraction) * mixed[action]
                + self.config.root_exploration_fraction * noise[i]
            )
        root.priors = _renormalize_legal(mixed, root.legal_mask)
        self._trace("root_noise", legal=legal, priors=root.priors.tolist())

    def _trace(self, event: str, **payload: object) -> None:
        if self.config.trace:
            self.last_trace.append({"event": event, **payload})

    @staticmethod
    def _path_actions(path: list[Node]) -> list[int]:
        return [node.parent_action for node in path[1:] if node.parent_action is not None]


def masked_softmax(logits: np.ndarray, legal_mask: int) -> np.ndarray:
    priors = np.zeros(ACTION_SIZE, dtype=np.float32)
    legal = [action for action in range(ACTION_SIZE) if legal_mask & (1 << action)]
    if not legal:
        return priors
    legal_logits = np.array([float(logits[action]) for action in legal], dtype=np.float64)
    legal_logits -= legal_logits.max()
    exp = np.exp(legal_logits)
    probs = exp / exp.sum()
    for action, prob in zip(legal, probs):
        priors[action] = float(prob)
    return priors


def _renormalize_legal(values: np.ndarray, legal_mask: int) -> np.ndarray:
    out = np.zeros(ACTION_SIZE, dtype=np.float32)
    legal = [action for action in range(ACTION_SIZE) if legal_mask & (1 << action)]
    total = float(sum(values[action] for action in legal))
    if total <= 0.0:
        for action in legal:
            out[action] = 1.0 / len(legal)
        return out
    for action in legal:
        out[action] = float(values[action] / total)
    return out


def visit_counts_to_policy(visit_counts: np.ndarray, temperature: float, legal_mask: int) -> np.ndarray:
    counts = visit_counts.astype(np.float64)
    legal = np.array([bool(legal_mask & (1 << action)) for action in range(ACTION_SIZE)])
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    if not legal.any():
        return policy
    counts[~legal] = 0
    if temperature == 0:
        legal_counts = counts.copy()
        if legal_counts.max() <= 0:
            first = int(np.flatnonzero(legal)[0])
            policy[first] = 1.0
            return policy
        policy[int(np.argmax(legal_counts))] = 1.0
        return policy
    scaled = np.power(counts, 1.0 / temperature)
    scaled[~legal] = 0
    total = float(scaled.sum())
    if total <= 0:
        policy[legal] = 1.0 / int(legal.sum())
        return policy
    return (scaled / total).astype(np.float32)


def choose_action(policy: np.ndarray, rng: random.Random | None = None) -> int:
    generator = rng or random
    threshold = generator.random()
    cumulative = 0.0
    for action, probability in enumerate(policy):
        cumulative += float(probability)
        if threshold <= cumulative:
            return action
    return int(np.argmax(policy))
