"""AMP discriminator tests, skipped when mjlab/torch is unavailable."""
import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

from wisco_amp.discriminator import AMPDiscriminator  # noqa: E402


def test_discriminator_loss_and_style_reward():
  torch.manual_seed(0)
  disc = AMPDiscriminator(input_dim=8, hidden_dims=(16, 8))
  expert = torch.randn(64, 8)
  policy = torch.randn(64, 8) + 2.0
  loss, metrics = disc.loss(expert, policy, gradient_penalty=1.0)
  assert loss.ndim == 0
  assert "amp/disc_accuracy" in metrics
  loss.backward()
  assert all(p.grad is not None for p in disc.parameters() if p.requires_grad)
  with torch.no_grad():
    reward = disc.style_reward(torch.randn(16, 8))
  assert reward.shape == (16,)
  assert torch.all(reward >= 0.0) and torch.all(reward <= 1.0)
