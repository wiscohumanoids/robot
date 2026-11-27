"""Small adversarial motion prior discriminator.

Clean-room implementation of the standard AMP objective: a discriminator is
trained to score reference transitions high and policy transitions low; the
policy receives a style reward based on the discriminator's score on its own
transitions.

No code is taken from beyondAMP, AMP_mjlab, Cyclotron or AMP_for_hardware.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
  """Resolve a discriminator activation by name."""
  if name == "elu":
    return nn.ELU()
  if name == "relu":
    return nn.ReLU()
  if name == "gelu":
    return nn.GELU()
  raise ValueError(f"unknown activation: {name}")


@dataclass(frozen=True)
class DiscriminatorCfg:
  """Discriminator-only settings; the runner receives these from AmpCfg."""
  input_dim: int
  hidden_dims: tuple[int, ...] = (1024, 512)
  activation: str = "elu"
  gradient_penalty: float = 10.0


class AMPDiscriminator(nn.Module):
  """MLP over a single transition, (s_t, s_t+1)."""

  def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (1024, 512),
               activation: str = "elu"):
    """Build the MLP."""
    super().__init__()
    layers: list[nn.Module] = []
    last = input_dim
    for width in hidden_dims:
      layers += [nn.Linear(last, width), _activation(activation)]
      last = width
    layers.append(nn.Linear(last, 1))
    self.net = nn.Sequential(*layers)

  def forward(self, pair: torch.Tensor) -> torch.Tensor:
    """Discriminator logits for a batch of transitions, shape (N,)."""
    return self.net(pair).squeeze(-1)

  def style_reward(self, pair: torch.Tensor) -> torch.Tensor:
    """AMP style reward in [0, 1], shape (N,)."""
    with torch.no_grad():
      d = self(pair)
      return torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)

  def loss(self, expert: torch.Tensor, policy: torch.Tensor,
           gradient_penalty: float = 10.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Least-squares AMP loss with a reference gradient penalty."""
    expert = expert.detach().requires_grad_(True)
    d_expert = self(expert)
    d_policy = self(policy.detach())
    loss_expert = 0.5 * ((d_expert - 1.0) ** 2).mean()
    loss_policy = 0.5 * ((d_policy + 1.0) ** 2).mean()
    loss = loss_expert + loss_policy

    gp = torch.zeros((), device=expert.device)
    if gradient_penalty > 0.0:
      grad = torch.autograd.grad(
        outputs=d_expert.sum(), inputs=expert, create_graph=True, retain_graph=True)[0]
      gp = ((grad.norm(dim=-1) - 1.0) ** 2).mean()
      loss = loss + gradient_penalty * gp

    metrics = {
      "amp/disc_expert": float(d_expert.mean().detach()),
      "amp/disc_policy": float(d_policy.mean().detach()),
      "amp/disc_accuracy": float(((d_expert > 0).float().mean() + (d_policy < 0).float().mean()) * 0.5),
      "amp/disc_loss": float(loss.detach()),
      "amp/gradient_penalty": float(gp.detach()),
    }
    return loss, metrics
