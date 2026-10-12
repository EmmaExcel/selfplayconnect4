import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class Connect4Network(nn.Module):
    def __init__(self, num_residual_blocks: int = 5, num_channels: int = 128,
                 rows: int = 6, cols: int = 7):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.num_actions = cols

        self.conv_input = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_residual_blocks)]
        )

        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * rows * cols, cols)

        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * rows * cols, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x, legal_moves_mask=None):
        x = F.relu(self.bn_input(self.conv_input(x)))

