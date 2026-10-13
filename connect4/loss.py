import torch
import torch.nn as nn


class AlphaZeroLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_policy: torch.Tensor, value: torch.Tensor,
                target_policy: torch.Tensor, target_value: torch.Tensor) -> tuple:
        safe_log_policy = torch.where(target_policy > 0, log_policy, torch.zeros_like(log_policy))
        policy_loss = -torch.mean(torch.sum(target_policy * safe_log_policy, dim=1))
        value_loss = torch.mean((value - target_value) ** 2)
        total_loss = policy_loss + value_loss
        return total_loss, policy_loss, value_loss
