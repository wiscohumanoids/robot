"""Small PyTorch networks for EXPO editing of genuine VLA action proposals.

Inputs are already encoded observation vectors and checkpoint-normalized
actions. This module never changes normalization or constructs base proposals.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def scaled_tanh_log_prob(pre_tanh: Tensor, mean: Tensor, log_std: Tensor,
                         scale: float) -> Tensor:
    """Log density of ``scale * tanh(z)`` summed over the last dimension.

    Computing the Jacobian from z keeps the result finite at tanh saturation.
    The scale Jacobian is essential: this is the density of the physical edit
    in normalized action coordinates, not the unscaled tanh variable.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Edit scale must be positive and finite")
    gaussian = -.5 * ((pre_tanh - mean) * torch.exp(-log_std)).square()
    gaussian = gaussian - log_std - .5 * math.log(2 * math.pi)
    log_jacobian = 2 * (math.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))
    return (gaussian - log_jacobian - math.log(scale)).sum(dim=-1)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                         nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
                         nn.LayerNorm(hidden_dim), nn.SiLU(),
                         nn.Linear(hidden_dim, output_dim))


class SquashedGaussianEditor(nn.Module):
    """Stochastic bounded edit conditioned on observation and base action."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256,
                 residual_bound: float = .05):
        super().__init__()
        if min(obs_dim, action_dim, hidden_dim) < 1:
            raise ValueError("Network dimensions must be positive")
        if not math.isfinite(residual_bound) or residual_bound <= 0:
            raise ValueError("residual_bound must be positive and finite")
        self.action_dim = action_dim
        self.residual_bound = float(residual_bound)
        self.network = _mlp(obs_dim + action_dim, hidden_dim, action_dim * 2)
        # Start near the base policy, with a small but nonzero exploration edit.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():
            self.network[-1].bias[action_dim:].fill_(-1.)

    def forward(self, obs: Tensor, base_actions: Tensor) -> tuple[Tensor, Tensor]:
        output = self.network(torch.cat((obs, base_actions), dim=-1))
        mean, log_std = output.chunk(2, dim=-1)
        return mean, log_std.clamp(-5., 2.)

    def sample(self, obs: Tensor, base_actions: Tensor, deterministic: bool = False,
               generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        mean, log_std = self(obs, base_actions)
        if deterministic:
            pre_tanh = mean
        else:
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device,
                                generator=generator)
            pre_tanh = mean + log_std.exp() * noise
        edit = self.residual_bound * pre_tanh.tanh()
        return edit, scaled_tanh_log_prob(pre_tanh, mean, log_std, self.residual_bound)


class QEnsemble(nn.Module):
    """Independent Q networks; result shape is (number of critics, batch)."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256,
                 num_critics: int = 10):
        super().__init__()
        if min(obs_dim, action_dim, hidden_dim, num_critics) < 1:
            raise ValueError("Network dimensions must be positive")
        self.heads = nn.ModuleList([_mlp(obs_dim + action_dim, hidden_dim, 1)
                                   for _ in range(num_critics)])

    def forward(self, obs: Tensor, actions: Tensor,
                indices: Tensor | None = None) -> Tensor:
        inputs = torch.cat((obs, actions), dim=-1)
        chosen = range(len(self.heads)) if indices is None else indices.tolist()
        return torch.stack([self.heads[i](inputs).squeeze(-1) for i in chosen], dim=0)
