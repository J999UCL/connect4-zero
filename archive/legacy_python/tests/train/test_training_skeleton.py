import torch

from connect4_zero.model import Connect4ResNet3D
from connect4_zero.train import AlphaZeroLoss, TrainerConfig, train_step


def test_alphazero_loss_returns_named_components() -> None:
    loss_fn = AlphaZeroLoss()
    logits = torch.zeros((3, 16), dtype=torch.float32)
    values = torch.zeros(3, dtype=torch.float32)
    policy = torch.full((3, 16), 1 / 16, dtype=torch.float32)
    targets = torch.tensor([1.0, -1.0, 0.0])

    losses = loss_fn(logits, values, policy, targets)

    assert losses.total.item() > 0
    assert losses.policy.item() > 0
    assert losses.value.item() > 0


def test_train_step_runs_without_nans() -> None:
    model = Connect4ResNet3D()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch = {
        "input": torch.zeros((4, 2, 4, 4, 4), dtype=torch.float32),
        "policy": torch.full((4, 16), 1 / 16, dtype=torch.float32),
        "value": torch.tensor([1.0, -1.0, 0.0, 1.0], dtype=torch.float32),
        "legal_mask": torch.ones((4, 16), dtype=torch.bool),
    }

    metrics = train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        loss_fn=AlphaZeroLoss(),
        device=torch.device("cpu"),
        config=TrainerConfig(batch_size=4, num_workers=0),
    )

    assert metrics["loss"] > 0
    assert metrics["policy_loss"] > 0
    assert metrics["value_loss"] >= 0
    assert metrics["grad_norm"] > 0
