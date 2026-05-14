from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import torch
from torch import nn

BOARD_SIZE = 4
ACTION_SIZE = 16
INPUT_CHANNELS = 2


@dataclass(frozen=True, slots=True)
class ModelConfig:
    preset: str
    channels: int
    residual_blocks: int
    value_hidden: int
    input_channels: int = INPUT_CHANNELS
    policy_head_channels: int = 2
    value_head_channels: int = 1

    @classmethod
    def for_preset(cls, preset: str) -> "ModelConfig":
        if preset == "tiny":
            return cls(preset="tiny", channels=16, residual_blocks=3, value_hidden=32)
        if preset == "small":
            return cls(preset="small", channels=32, residual_blocks=4, value_hidden=64)
        if preset == "medium":
            return cls(preset="medium", channels=64, residual_blocks=6, value_hidden=64)
        raise ValueError(f"unknown model preset: {preset}")

    def stable_hash(self) -> str:
        payload = "|".join(f"{key}={value}" for key, value in sorted(asdict(self).items()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + residual)


class AlphaZeroNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        c = config.channels
        self.stem_conv = nn.Conv3d(config.input_channels, c, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm3d(c)
        self.tower = nn.Sequential(*(ResidualBlock3D(c) for _ in range(config.residual_blocks)))
        self.policy_conv = nn.Conv3d(c, config.policy_head_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm3d(config.policy_head_channels)
        self.policy_fc = nn.Linear(config.policy_head_channels * BOARD_SIZE**3, ACTION_SIZE)
        self.value_conv = nn.Conv3d(c, config.value_head_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm3d(config.value_head_channels)
        self.value_fc1 = nn.Linear(config.value_head_channels * BOARD_SIZE**3, config.value_hidden)
        self.value_fc2 = nn.Linear(config.value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.stem_bn(self.stem_conv(x)))
        x = self.tower(x)

        policy = torch.relu(self.policy_bn(self.policy_conv(x)))
        policy = torch.flatten(policy, start_dim=1)
        policy_logits = self.policy_fc(policy)

        value = torch.relu(self.value_bn(self.value_conv(x)))
        value = torch.flatten(value, start_dim=1)
        value = torch.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)
        return policy_logits, value


def create_model(preset: str = "small") -> AlphaZeroNet:
    return AlphaZeroNet(ModelConfig.for_preset(preset))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
