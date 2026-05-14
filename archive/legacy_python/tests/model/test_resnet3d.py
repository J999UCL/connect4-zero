import torch

from connect4_zero.game.constants import CURRENT_PLAYER, OPPONENT_PLAYER
from connect4_zero.model import Connect4ResNet3D, ResNet3DConfig, count_parameters, encode_boards
from connect4_zero.model.checkpoint import load_checkpoint, save_checkpoint


def test_encode_boards_maps_canonical_pieces_to_planes() -> None:
    boards = torch.zeros((2, 4, 4, 4), dtype=torch.int8)
    boards[0, 0, 0, 0] = CURRENT_PLAYER
    boards[0, 1, 0, 0] = OPPONENT_PLAYER

    planes = encode_boards(boards)

    assert planes.shape == (2, 2, 4, 4, 4)
    assert planes[0, 0, 0, 0, 0].item() == 1.0
    assert planes[0, 1, 1, 0, 0].item() == 1.0
    assert planes.sum().item() == 2.0


def test_resnet_forward_shapes_values_and_gradients() -> None:
    model = Connect4ResNet3D(ResNet3DConfig())
    boards = torch.zeros((3, 4, 4, 4), dtype=torch.int8)

    policy_logits, values = model(boards)
    loss = policy_logits.mean() + values.mean()
    loss.backward()

    assert policy_logits.shape == (3, 16)
    assert values.shape == (3,)
    assert torch.all(values.ge(-1.0))
    assert torch.all(values.le(1.0))
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_resnet_parameter_count_is_expected_scale() -> None:
    model = Connect4ResNet3D()

    assert 1_300_000 <= count_parameters(model) <= 1_400_000
    assert count_parameters(model) == 1_338_711


def test_checkpoint_round_trip_preserves_eval_outputs(tmp_path) -> None:
    model = Connect4ResNet3D()
    model.eval()
    inputs = torch.zeros((2, 4, 4, 4), dtype=torch.int8)
    with torch.no_grad():
        expected_logits, expected_values = model(inputs)

    path = tmp_path / "model.pt"
    save_checkpoint(path, model, step=12, epoch=3, metrics={"loss": 1.5})
    loaded = load_checkpoint(path).model
    loaded.eval()
    with torch.no_grad():
        actual_logits, actual_values = loaded(inputs)

    assert torch.equal(expected_logits, actual_logits)
    assert torch.equal(expected_values, actual_values)
