"""Small 3D ResNet policy/value model for 4x4x4 Connect Four."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from connect4_zero.game.constants import ACTION_SIZE, BOARD_SIZE, CURRENT_PLAYER, OPPONENT_PLAYER


@dataclass(frozen=True)
class ResNet3DConfig:
    """Architecture choices for the first AlphaZero network."""

    input_channels: int = 2
    channels: int = 64
    num_res_blocks: int = 6
    board_size: int = BOARD_SIZE
    policy_head_channels: int = 2
    value_head_channels: int = 1
    value_hidden_dim: int = 64

    def __post_init__(self) -> None:
        if self.input_channels != 2:
            raise ValueError("input_channels must be 2 for current/opponent planes")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.num_res_blocks <= 0:
            raise ValueError("num_res_blocks must be positive")
        if self.board_size != BOARD_SIZE:
            raise ValueError(f"board_size must be {BOARD_SIZE}")
        if self.policy_head_channels <= 0:
            raise ValueError("policy_head_channels must be positive")
        if self.value_head_channels <= 0:
            raise ValueError("value_head_channels must be positive")
        if self.value_hidden_dim <= 0:
            raise ValueError("value_hidden_dim must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ResNet3DConfig":
        return cls(**values)


class ResidualBlock3D(nn.Module):
    """Two-convolution 3D residual block."""

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


class Connect4ResNet3D(nn.Module):
    """AlphaZero-style 3D ResNet with policy and value heads."""

    def __init__(self, config: ResNet3DConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ResNet3DConfig()
        board_volume = self.config.board_size**3

        self.stem = nn.Sequential(
            nn.Conv3d(
                self.config.input_channels,
                self.config.channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(self.config.channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock3D(self.config.channels) for _ in range(self.config.num_res_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv3d(
                self.config.channels,
                self.config.policy_head_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm3d(self.config.policy_head_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.config.policy_head_channels * board_volume, ACTION_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv3d(
                self.config.channels,
                self.config.value_head_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm3d(self.config.value_head_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.config.value_head_channels * board_volume, self.config.value_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.value_hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        planes = encode_boards(x) if x.ndim == 4 else x.to(dtype=torch.float32)
        if tuple(planes.shape[1:]) != (
            self.config.input_channels,
            self.config.board_size,
            self.config.board_size,
            self.config.board_size,
        ):
            raise ValueError(
                "input must have shape "
                f"(B, {self.config.input_channels}, {self.config.board_size}, "
                f"{self.config.board_size}, {self.config.board_size})"
            )

        features = self.res_blocks(self.stem(planes))
        policy_logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return policy_logits, values


def encode_boards(boards: torch.Tensor) -> torch.Tensor:
    """Encode canonical int boards as current/opponent occupancy planes."""
    if boards.ndim != 4:
        raise ValueError(f"boards must have shape (B, 4, 4, 4), got {tuple(boards.shape)}")
    current = boards.eq(CURRENT_PLAYER)
    opponent = boards.eq(OPPONENT_PLAYER)
    return torch.stack((current, opponent), dim=1).to(dtype=torch.float32)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
