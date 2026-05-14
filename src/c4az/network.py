from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from c4az.game import ACTION_SIZE, BOARD_SIZE, Position, encode_positions


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    preset: str = "small"
    input_channels: int = 2
    channels: int = 32
    num_res_blocks: int = 4
    policy_head_channels: int = 2
    value_head_channels: int = 1
    value_hidden_dim: int = 64

    @classmethod
    def for_preset(cls, preset: str) -> "NetworkConfig":
        if preset == "tiny":
            return cls(preset="tiny", channels=16, num_res_blocks=3, value_hidden_dim=32)
        if preset == "small":
            return cls(preset="small", channels=32, num_res_blocks=4, value_hidden_dim=64)
        if preset == "medium":
            return cls(preset="medium", channels=64, num_res_blocks=6, value_hidden_dim=64)
        raise ValueError(f"unknown network preset: {preset}")


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class AlphaZeroNet(nn.Module):
    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig.for_preset("small")
        c = self.config.channels
        self.stem = nn.Sequential(
            nn.Conv3d(self.config.input_channels, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*(ResidualBlock3D(c) for _ in range(self.config.num_res_blocks)))
        self.policy_head = nn.Sequential(
            nn.Conv3d(c, self.config.policy_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.config.policy_head_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.config.policy_head_channels * BOARD_SIZE**3, ACTION_SIZE),
        )
        self.value_conv = nn.Sequential(
            nn.Conv3d(c, self.config.value_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.config.value_head_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.config.value_head_channels * BOARD_SIZE**3, self.config.value_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.value_hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.tower(self.stem(x))
        logits = self.policy_head(features)
        values = self.value_head(self.value_conv(features)).squeeze(-1)
        return logits, values

    def evaluate_positions(self, positions: list[Position], device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        model_device = next(self.parameters()).device
        target_device = torch.device(device) if device is not None else model_device
        planes = encode_positions(positions, device=target_device)
        return self(planes)


def create_model(preset: str = "small") -> AlphaZeroNet:
    return AlphaZeroNet(NetworkConfig.for_preset(preset))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
