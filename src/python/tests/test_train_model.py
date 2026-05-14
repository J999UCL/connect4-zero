import torch

from c4zero_train.model import create_model, count_parameters


def test_model_parameter_counts():
    assert count_parameters(create_model("tiny")) == 46791
    assert count_parameters(create_model("small")) == 229879
    assert count_parameters(create_model("medium")) == 1338711


def test_model_forward_shapes_and_value_range():
    model = create_model("tiny")
    x = torch.zeros((3, 2, 4, 4, 4), dtype=torch.float32)
    logits, values = model(x)
    assert logits.shape == (3, 16)
    assert values.shape == (3,)
    assert torch.all(values <= 1.0)
    assert torch.all(values >= -1.0)
