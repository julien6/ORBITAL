from __future__ import annotations

import torch
from torch import nn


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, joint_obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def action_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def value(self, joint_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(joint_obs).squeeze(-1)
