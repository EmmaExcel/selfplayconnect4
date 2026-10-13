import torch
import torch.nn as nn


class AlphaZeroLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_policy: torch.Tensor, value: torch.Tensor,
                target_policy: torch.Tensor, target_value: torch.Tensor) -> tuple:
