from connect4_zero.game.constants import ACTION_SIZE
from connect4_zero.game.symmetries import make_symmetry_permutations


def test_symmetry_permutations_are_valid_action_permutations() -> None:
    permutations = make_symmetry_permutations()

    assert permutations.shape == (8, ACTION_SIZE)
    for row in permutations.tolist():
        assert sorted(row) == list(range(ACTION_SIZE))


def test_first_symmetry_is_identity() -> None:
    permutations = make_symmetry_permutations()

    assert permutations[0].tolist() == list(range(ACTION_SIZE))
