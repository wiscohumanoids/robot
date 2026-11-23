"""The shared local policy: one small network for every local skill and object, on the wrist image, the relative
state and a condition (which skill, which object: learning/cond.py), sized to train in minutes.

LocalPolicy ("ACT-lite"): a convolutional encoder for the 64 x 64 wrist image (random-shift augmented while
training, as in DrQ / HIL-SERL), the 5-number state, and a head predicting a chunk of the next `chunk` actions (ACT's
idea: predicting several steps ahead smooths the behaviour and averages out demonstrator noise). At run time the
first action of each fresh chunk is executed, 10 times a second.

SAC (for the RL stage): a tanh-Gaussian actor built on the same encoder and trunk (so it starts from the imitation
policy's weights), and two Q networks with their own encoder. Small on purpose: HIL-SERL reaches near-perfect
insertion with a ResNet-10 at 128 x 128 on one GPU in 1-2.5 h; this is smaller still, to run on a laptop or the VM.

Checkpoints: models/local/<name>.pt (SHARED) with the network's settings and a short record of what it was trained
on; models/local/latest.json names the one the skills load.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from learning.cond import CONDITION_DIM

ROOT = pathlib.Path(__file__).resolve().parents[1]
ACTION_DIM, STATE_DIM = 5, 5
SHARED = 'local'                       # the one policy for every local skill
STATE_SCALE = torch.tensor([1 / 0.01, 1 / 0.01, 1 / 0.01, 1 / 0.1, 1 / 0.04])   # metres and radians to ~unit


def device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ's augmentation: pad by replication and crop back at a random offset, per image."""
    n, c, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    eps = 1.0 / (h + 2 * pad)
    ar = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device)[:h]
    ar = ar.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base = torch.cat([ar, ar.transpose(1, 0)], 2).unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, (n, 1, 1, 2), device=x.device, dtype=x.dtype) * 2.0 / (h + 2 * pad)
    return F.grid_sample(x, base + shift, padding_mode='zeros', align_corners=False)


class Encoder(nn.Module):
    def __init__(self, out: int = 256):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 32, 5, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, 2), nn.ReLU(), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(64 * 6 * 6, out), nn.LayerNorm(out), nn.Tanh())

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(img))


def prep(wrist, state, dev, augment: bool = False, cond=None):
    """Batch tensors: images (N, 3, 64, 64) in [0, 1] (randomly shifted when `augment`), states scaled, and the
    condition (N, CONDITION_DIM) appended to the state (zeros when not given)."""
    img = torch.as_tensor(np.asarray(wrist), device=dev).permute(0, 3, 1, 2).float() / 255.0
    if augment:
        img = random_shift(img)
    st = torch.as_tensor(np.asarray(state), device=dev, dtype=torch.float32) * STATE_SCALE.to(dev)
    c = torch.zeros((st.shape[0], CONDITION_DIM), device=dev) if cond is None else \
        torch.as_tensor(np.asarray(cond), device=dev, dtype=torch.float32)
    return img, torch.cat([st, c], -1)


class LocalPolicy(nn.Module):
    """Image + state -> a chunk of actions."""

    def __init__(self, chunk: int = 4, hidden: int = 256):
        super().__init__()
        self.chunk = chunk
        self.enc = Encoder()
        self.cond = nn.Sequential(nn.Linear(CONDITION_DIM, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(256 + STATE_DIM + 64, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                   nn.ReLU())
        self.head = nn.Linear(hidden, chunk * ACTION_DIM)

    def features(self, img, st):
        """`st`: the scaled state with the condition appended (see `prep`)."""
        s, c = st[:, :STATE_DIM], st[:, STATE_DIM:]
        return self.trunk(torch.cat([self.enc(img), s, self.cond(c)], -1))

    def forward(self, img, st):
        return torch.tanh(self.head(self.features(img, st))).view(-1, self.chunk, ACTION_DIM)

    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        dev = next(self.parameters()).device
        c = obs.get('cond')
        img, st = prep(obs['wrist'][None], obs['state'][None], dev, cond=None if c is None else c[None])
        return self(img, st)[0, 0].cpu().numpy()


class Actor(nn.Module):
    """SAC actor on the LocalPolicy's encoder and trunk (first action of the chunk as the mean)."""

    def __init__(self, base: LocalPolicy):
        super().__init__()
        self.base = base
        self.log_std = nn.Linear(base.head.in_features, ACTION_DIM)
        nn.init.constant_(self.log_std.bias, -1.5)
        nn.init.zeros_(self.log_std.weight)

    def forward(self, img, st, deterministic: bool = False):
        h = self.base.features(img, st)
        mu = self.base.head(h).view(-1, self.base.chunk, ACTION_DIM)[:, 0]
        log_std = self.log_std(h).clamp(-5, 1)
        if deterministic:
            return torch.tanh(mu), None
        eps = torch.randn_like(mu)
        pre = mu + eps * log_std.exp()
        a = torch.tanh(pre)
        logp = (-0.5 * eps ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1) - torch.log(1 - a ** 2 + 1e-6).sum(-1)
        return a, logp


class Critic(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.enc = Encoder()
        self.q = nn.ModuleList([nn.Sequential(nn.Linear(256 + STATE_DIM + CONDITION_DIM + ACTION_DIM, hidden), nn.ReLU(),
                                              nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                                for _ in range(2)])

    def forward(self, img, st, a):
        z = torch.cat([self.enc(img), st, a], -1)
        return [q(z).squeeze(-1) for q in self.q]


# ---------- checkpoints ----------
def models_dir(skill: str) -> pathlib.Path:
    """models/<skill>/ (PANTHERA_AGENT_MODELS moves the models folder; the VM keeps it next to its data)."""
    import os
    d = pathlib.Path(os.environ.get('PANTHERA_AGENT_MODELS') or ROOT / 'models') / skill
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(policy: LocalPolicy, skill: str, name: str, record: dict, make_latest: bool = True) -> pathlib.Path:
    path = models_dir(skill) / f'{name}.pt'
    torch.save(dict(state_dict=policy.state_dict(), chunk=policy.chunk, record=record, time=time.time()), path)
    if make_latest:
        (models_dir(skill) / 'latest.json').write_text(json.dumps(dict(name=name, record=record), indent=1))
    return path


def load(skill: str, name: str = 'latest', dev: str | None = None) -> LocalPolicy | None:
    d = models_dir(skill)
    if name == 'latest':
        if not (d / 'latest.json').exists():
            return None
        name = json.loads((d / 'latest.json').read_text())['name']
    path = d / f'{name}.pt'
    if not path.exists():
        return None
    ck = torch.load(path, map_location='cpu', weights_only=False)
    p = LocalPolicy(chunk=ck['chunk'])
    p.load_state_dict(ck['state_dict'])
    p.record = ck.get('record', {})
    return p.to(dev or device()).eval()
