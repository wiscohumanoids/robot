"""Training from stored episodes: the imitation policy (BC on action chunks) and the success classifier.

    .venv/bin/python -m learning.train bc --skill insert --steps 3000          # -> models/insert/bc_<time>.pt
    .venv/bin/python -m learning.train reward --skill insert --steps 1500      # -> models/insert/reward.pt

Imitation targets: the supervisor's action where it gave one (DAgger relabels and interventions), else the action
that was executed; only from successful episodes unless an episode carries supervisor labels throughout (a failed
policy rollout relabelled by the expert still teaches what should have been done). Runs on CUDA (the VM), Apple's
GPU or the CPU.

The success classifier (HIL-SERL's reward): a small CNN on single wrist frames, positives = the last frames of
successful episodes, negatives = everything else (the earlier frames, and failed episodes). HIL-SERL reports >95 %
accuracy from about 200 positive and 1,000 negative frames. Labels: the simulator's truth, or Claude's
(learning/label.py), or a person's.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from learning.cond import condition
from learning.episodes import EpisodeStore, all_stores
from learning.policy import SHARED, Encoder, LocalPolicy, device, prep, save


def _targets(z: dict) -> np.ndarray:
    a = z['action'].copy()
    has = ~np.isnan(z['expert']).any(axis=1)
    a[has] = z['expert'][has]
    return a


def chunk_dataset(items: list[tuple], chunk: int):
    """(wrist, state, condition, target chunk (N, chunk, 5), mask (N, chunk)) over (store, entry) pairs."""
    W, S, C, A, M = [], [], [], [], []
    for store, e in items:
        z = store.load(e['id'])
        tgt = _targets(z)
        T = len(tgt)
        cond = z['cond'] if 'cond' in z else np.repeat(condition(store.skill, None)[None], T, axis=0)
        for t in range(T):
            idx = np.arange(t, t + chunk)
            m = idx < T
            a = tgt[np.minimum(idx, T - 1)].copy()
            W.append(z['wrist'][t])
            S.append(z['state'][t])
            C.append(cond[t])
            A.append(a)
            M.append(m)
    return np.stack(W), np.stack(S), np.stack(C), np.stack(A).astype(np.float32), np.stack(M)


def train_bc(skill: str = SHARED, steps: int = 3000, chunk: int = 4, batch: int = 128, lr: float = 3e-4,
             init: LocalPolicy | None = None, name: str | None = None, log=print, store_root=None,
             make_latest: bool = True) -> tuple:
    """Fit the shared imitation policy on every skill's usable episodes (successful ones, and relabelled ones: the
    supervisor's action at every step); returns (policy, record). With `make_latest`, the skills load it from now
    on. `skill`: SHARED for all skills' episodes, or one skill's name to train on that skill alone."""
    stores = all_stores(store_root) if skill == SHARED else [EpisodeStore(skill, store_root)]
    items = [(st, e) for st in stores for e in st.entries()
             if (e.get('success') or e.get('relabelled')) and not str(e.get('source', '')).startswith('eval')]
    if not items:
        raise RuntimeError(f'no usable episodes in {[str(st.folder) for st in stores]}')
    eps = items
    W, S, Cn, A, M = chunk_dataset(items, chunk)
    dev = device()
    pol = (init or LocalPolicy(chunk=chunk)).to(dev).train()
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    n = len(W)
    order = np.random.default_rng(0).permutation(n)
    val = order[: max(1, n // 10)]
    tr = order[max(1, n // 10):]
    t0 = time.time()
    losses = []
    for step in range(1, steps + 1):
        b = np.random.choice(tr, min(batch, len(tr)))
        img, st = prep(W[b], S[b], dev, augment=True, cond=Cn[b])
        tgt = torch.as_tensor(A[b], device=dev)
        m = torch.as_tensor(M[b], device=dev, dtype=torch.float32)[..., None]
        loss = ((pol(img, st) - tgt).abs() * m).sum() / (m.sum() * A.shape[-1])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if step % max(1, steps // 10) == 0 or step == steps:
            with torch.no_grad():
                pol.eval()
                img, st = prep(W[val], S[val], dev, cond=Cn[val])
                vl = float(((pol(img, st) - torch.as_tensor(A[val], device=dev)).abs()
                            * torch.as_tensor(M[val], device=dev)[..., None]).mean())
                pol.train()
            log(dict(event='bc', step=step, loss=round(float(np.mean(losses[-50:])), 4), val=round(vl, 4),
                     seconds=round(time.time() - t0, 1)))
    tasks = sorted({str(e.get('task', st.skill)) for st, e in items})
    record = dict(kind='bc', episodes=len(eps), frames=int(n), steps=steps, chunk=chunk, device=dev, tasks=tasks,
                  final_loss=round(float(np.mean(losses[-50:])), 4), seconds=round(time.time() - t0, 1))
    name = name or f'bc_{time.strftime("%Y%m%d_%H%M%S")}'
    record['name'] = name
    save(pol.eval(), skill, name, record, make_latest=make_latest)
    return pol, record


class SuccessClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(128)
        self.out = nn.Linear(128, 1)

    def forward(self, img):
        return self.out(self.enc(img)).squeeze(-1)

    @torch.no_grad()
    def prob(self, wrist: np.ndarray) -> float:
        dev = next(self.parameters()).device
        img, _ = prep(wrist[None], np.zeros((1, 5), np.float32), dev)
        return float(torch.sigmoid(self(img))[0])


def train_reward(skill: str, steps: int = 1500, last: int = 2, batch: int = 128, log=print, store_root=None):
    """Fit the success classifier; returns (model, record with held-out accuracy)."""
    store = EpisodeStore(skill, store_root)
    X, Y = [], []
    for e in store.entries():
        z = store.load(e['id'])
        ok = e.get('claude_success', e.get('success'))
        T = len(z['wrist'])
        for t in range(T):
            X.append(z['wrist'][t])
            Y.append(1.0 if ok and t >= T - last else 0.0)
    X, Y = np.stack(X), np.array(Y, np.float32)
    if Y.sum() < 4:
        raise RuntimeError('too few successful frames to train a classifier')
    dev = device()
    model = SuccessClassifier().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    val, tr = idx[: len(X) // 5], idx[len(X) // 5:]
    pos = tr[Y[tr] > 0.5]
    neg = tr[Y[tr] < 0.5]
    for step in range(1, steps + 1):
        b = np.concatenate([rng.choice(pos, batch // 2), rng.choice(neg, batch // 2)])     # balanced batches
        img, _ = prep(X[b], np.zeros((len(b), 5), np.float32), dev, augment=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(img), torch.as_tensor(Y[b], device=dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        model.eval()
        img, _ = prep(X[val], np.zeros((len(val), 5), np.float32), dev)
        p = torch.sigmoid(model(img)).cpu().numpy()
    yv = Y[val]
    record = dict(kind='reward', frames=int(len(X)), positives=int(Y.sum()),
                  accuracy=round(float(((p > 0.5) == (yv > 0.5)).mean()), 3),
                  false_positive_rate=round(float(((p > 0.5) & (yv < 0.5)).sum() / max((yv < 0.5).sum(), 1)), 4),
                  recall=round(float(((p > 0.5) & (yv > 0.5)).sum() / max((yv > 0.5).sum(), 1)), 3))
    log(dict(event='reward_model', **record))
    from learning.policy import models_dir
    torch.save(dict(state_dict=model.state_dict(), record=record), models_dir(skill) / 'reward.pt')
    return model, record


def load_reward(skill: str):
    from learning.policy import models_dir
    path = models_dir(skill) / 'reward.pt'
    if not path.exists():
        return None
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = SuccessClassifier()
    m.load_state_dict(ck['state_dict'])
    return m.to(device()).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['bc', 'reward'])
    ap.add_argument('--skill', default='insert')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--chunk', type=int, default=4)
    ap.add_argument('--name')
    a = ap.parse_args()
    log = lambda d: print(json.dumps(d), flush=True)
    if a.what == 'bc':
        train_bc(a.skill, a.steps, a.chunk, name=a.name, log=log)
    else:
        train_reward(a.skill, a.steps, log=log)


if __name__ == '__main__':
    main()
