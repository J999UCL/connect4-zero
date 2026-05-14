import torch

from connect4_zero.data import SelfPlayConfig, SelfPlayDataset, SelfPlayGenerator, SelfPlayShardWriter
from connect4_zero.game.constants import ACTION_SIZE
from connect4_zero.game.symmetries import make_symmetry_permutations
from connect4_zero.search.types import BatchedSearchResult


class ScriptedSearch:
    def __init__(self, actions: list[int]) -> None:
        self.actions = actions
        self.calls = 0

    def search_batch(self, roots) -> BatchedSearchResult:
        action = self.actions[self.calls]
        self.calls += 1
        batch_size = roots.batch_size
        policy = torch.zeros((batch_size, ACTION_SIZE), dtype=torch.float32, device=roots.device)
        policy[:, action] = 1.0
        visit_counts = torch.zeros_like(policy)
        visit_counts[:, action] = 1.0
        q_values = torch.zeros_like(policy)
        root_values = torch.zeros(batch_size, dtype=torch.float32, device=roots.device)
        return BatchedSearchResult(
            visit_counts=visit_counts,
            policy=policy,
            q_values=q_values,
            root_values=root_values,
        )


class ReusableScriptedSearch(ScriptedSearch):
    def __init__(self, actions: list[int]) -> None:
        super().__init__(actions)
        self.root_tree_batches: list[list[object | None]] = []
        self.last_trees: list[object] = []
        self.last_reused_roots = 0
        self.last_fresh_roots = 0

    def search_batch_with_trees(self, roots, root_trees) -> BatchedSearchResult:
        self.root_tree_batches.append(list(root_trees))
        self.last_reused_roots = sum(tree is not None for tree in root_trees)
        self.last_fresh_roots = sum(tree is None for tree in root_trees)
        result = self.search_batch(roots)
        self.last_trees = [object() for _ in range(roots.batch_size)]
        return result

    def advance_tree(self, tree, action: int):
        return (tree, action)


def test_self_play_generator_assigns_final_values_by_position_player() -> None:
    search = ScriptedSearch([0, 4, 1, 5, 2, 6, 3])
    generator = SelfPlayGenerator(
        search=search,  # type: ignore[arg-type]
        config=SelfPlayConfig(batch_size=1, action_temperature=0, max_plies=8),
    )

    samples = generator.generate(num_games=1)

    assert samples.num_samples == 7
    assert samples.actions.tolist() == [0, 4, 1, 5, 2, 6, 3]
    assert samples.values.tolist() == [1, -1, 1, -1, 1, -1, 1]
    assert samples.policies.shape == (7, ACTION_SIZE)
    assert torch.allclose(samples.policies.sum(dim=1), torch.ones(7))


def test_self_play_generator_reuses_search_trees_between_plies() -> None:
    search = ReusableScriptedSearch([0, 4, 1, 5, 2, 6, 3])
    generator = SelfPlayGenerator(
        search=search,  # type: ignore[arg-type]
        config=SelfPlayConfig(batch_size=1, action_temperature=0, max_plies=8),
    )

    samples = generator.generate(num_games=1)

    assert samples.num_samples == 7
    assert search.root_tree_batches[0] == [None]
    assert all(batch[0] is not None for batch in search.root_tree_batches[1:])


def test_shard_writer_and_dataset_round_trip_with_symmetry(tmp_path) -> None:
    search = ScriptedSearch([0, 4, 1, 5, 2, 6, 3])
    generator = SelfPlayGenerator(
        search=search,  # type: ignore[arg-type]
        config=SelfPlayConfig(batch_size=1, action_temperature=0, max_plies=8),
    )
    samples = generator.generate(num_games=1)

    writer = SelfPlayShardWriter(tmp_path, samples_per_shard=3, metadata={"run_id": "test"})
    writer.write(samples)

    dataset = SelfPlayDataset(tmp_path / "manifest.jsonl", apply_symmetries=False)
    first = dataset[0]

    assert len(dataset) == samples.num_samples
    assert first["input"].shape == (2, 4, 4, 4)
    assert first["policy"].shape == (ACTION_SIZE,)
    assert first["value"].item() == 1.0
    assert first["action"].item() == 0

    augmented = SelfPlayDataset(tmp_path / "manifest.jsonl", apply_symmetries=True)
    permutation = make_symmetry_permutations()[1]
    rotated = augmented[1]

    assert len(augmented) == samples.num_samples * 8
    assert rotated["action"].item() == permutation[0].item()
    assert rotated["policy"][permutation[0]].item() == 1.0
