import random

import numpy as np

from c4zero_tools.datasets import Sample
from c4zero_train.replay import ReplayBuffer
from c4zero_train.symmetry import Symmetry, action_permutation, transform_sample


def make_sample() -> Sample:
    policy = np.zeros(16, dtype=np.float32)
    policy[0] = 0.25
    policy[5] = 0.75
    visits = np.zeros(16, dtype=np.uint32)
    visits[0] = 1
    visits[5] = 3
    current = (1 << 0) | (1 << 5)
    opponent = 1 << 10
    return Sample(
        current_bits=current,
        opponent_bits=opponent,
        heights=tuple([0] * 16),
        ply=3,
        game_id=9,
        legal_mask=(1 << 0) | (1 << 5) | (1 << 10),
        action=5,
        policy=policy,
        visit_counts=visits,
        value=-1.0,
    )


def test_rot90_transforms_board_targets_and_metadata():
    sample = make_sample()
    transformed = transform_sample(sample, Symmetry.ROT90)

    assert transformed.current_bits == (1 << 3) | (1 << 6)
    assert transformed.opponent_bits == (1 << 9)
    assert transformed.legal_mask == ((1 << 3) | (1 << 6) | (1 << 9))
    assert transformed.action == 6
    assert transformed.policy[3] == 0.25
    assert transformed.policy[6] == 0.75
    assert transformed.visit_counts[3] == 1
    assert transformed.visit_counts[6] == 3
    assert transformed.ply == sample.ply
    assert transformed.game_id == sample.game_id
    assert transformed.value == sample.value


def test_all_symmetries_preserve_policy_visit_totals_and_legality():
    sample = make_sample()
    for symmetry in Symmetry:
        transformed = transform_sample(sample, symmetry)
        assert transformed.policy.sum() == np.float32(1.0)
        assert int(transformed.visit_counts.sum()) == 4
        permutation = action_permutation(symmetry)
        assert transformed.legal_mask == sum(1 << permutation[action] for action in {0, 5, 10})
        assert transformed.legal_mask & (1 << transformed.action)
        assert transformed.policy[transformed.action] > 0.0


def test_replay_batch_can_apply_random_symmetry_augmentation():
    sample = make_sample()
    replay = ReplayBuffer([sample])
    batch = replay.sample_batch(16, random.Random(4), augment_symmetries=True)

    assert len(batch) == 16
    assert {augmented.current_bits for augmented in batch} != {sample.current_bits}
    for augmented in batch:
        assert augmented.policy.sum() == np.float32(1.0)
        assert int(augmented.visit_counts.sum()) == 4
        assert augmented.legal_mask & (1 << augmented.action)


def test_replay_orbit_batch_expands_each_base_sample_to_all_symmetries():
    sample = make_sample()
    replay = ReplayBuffer([sample])
    batch = replay.sample_orbit_batch(2, random.Random(1))

    assert len(batch) == 16
    for offset in (0, 8):
        for symmetry in Symmetry:
            expected = transform_sample(sample, symmetry)
            actual = batch[offset + int(symmetry)]
            assert actual.current_bits == expected.current_bits
            assert actual.opponent_bits == expected.opponent_bits
            assert actual.legal_mask == expected.legal_mask
            assert actual.action == expected.action
            assert np.array_equal(actual.policy, expected.policy)
            assert np.array_equal(actual.visit_counts, expected.visit_counts)
            assert actual.value == sample.value
            assert actual.ply == sample.ply
