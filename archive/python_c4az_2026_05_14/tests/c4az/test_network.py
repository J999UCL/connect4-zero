import torch

from c4az.game import initial_position
from c4az.network import AlphaZeroNet, NetworkConfig, count_parameters, create_model
from c4az.train import load_checkpoint, save_checkpoint


def test_model_presets_have_expected_parameter_ranges() -> None:
    counts = {preset: count_parameters(create_model(preset)) for preset in ("tiny", "small", "medium")}

    assert 40_000 <= counts["tiny"] <= 55_000
    assert 210_000 <= counts["small"] <= 250_000
    assert 1_300_000 <= counts["medium"] <= 1_400_000


def test_forward_shapes_and_value_range() -> None:
    model = create_model("tiny")
    inputs = torch.zeros(3, 2, 4, 4, 4)
    logits, values = model(inputs)

    assert logits.shape == (3, 16)
    assert values.shape == (3,)
    assert torch.all(values <= 1)
    assert torch.all(values >= -1)


def test_gradients_flow_through_policy_and_value_heads() -> None:
    model = create_model("tiny")
    inputs = torch.randn(2, 2, 4, 4, 4)
    logits, values = model(inputs)
    loss = logits[:, 0].mean() + values.mean()
    loss.backward()

    policy_grad = model.policy_head[-1].weight.grad
    value_grad = model.value_head[-2].weight.grad
    assert policy_grad is not None and torch.count_nonzero(policy_grad) > 0
    assert value_grad is not None and torch.count_nonzero(value_grad) > 0


def test_checkpoint_round_trip_preserves_eval_outputs(tmp_path) -> None:
    model = AlphaZeroNet(NetworkConfig.for_preset("tiny"))
    model.eval()
    inputs = torch.randn(2, 2, 4, 4, 4)
    before = model(inputs)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(path, model, step=11, epoch=2)
    loaded, payload = load_checkpoint(path)
    loaded.eval()
    after = loaded(inputs)

    assert payload["step"] == 11
    assert torch.allclose(before[0], after[0])
    assert torch.allclose(before[1], after[1])


def test_evaluate_positions_uses_bitboard_encoder() -> None:
    model = create_model("tiny")
    logits, values = model.evaluate_positions([initial_position(), initial_position()])
    assert logits.shape == (2, 16)
    assert values.shape == (2,)
