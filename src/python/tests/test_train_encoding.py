import numpy as np

from c4zero_tools.datasets import Sample, encode_sample
from c4zero_train.encoding import encode_bits, encode_samples


def make_sample() -> Sample:
    return Sample(
        current_bits=(1 << 0) | (1 << 63),
        opponent_bits=(1 << 21),
        heights=tuple([0] * 16),
        ply=3,
        game_id=7,
        legal_mask=0xFFFF,
        action=5,
        policy=np.eye(1, 16, 5, dtype=np.float32).reshape(16),
        visit_counts=np.arange(16, dtype=np.uint32),
        value=1.0,
    )


def test_encoder_bit_layout_matches_tools_reader():
    sample = make_sample()
    expected = encode_sample(sample)
    actual = encode_bits(sample.current_bits, sample.opponent_bits)
    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 0, 0, 0] == 1.0
    assert actual[0, 3, 3, 3] == 1.0
    assert actual[1, 1, 1, 1] == 1.0


def test_encode_samples_tensor_shape():
    tensor = encode_samples([make_sample(), make_sample()])
    assert tuple(tensor.shape) == (2, 2, 4, 4, 4)
