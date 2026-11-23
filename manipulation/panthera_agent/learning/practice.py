"""Practice: collect, imitate, correct, reinforce, evaluate a local skill, with a supervisor (learning/supervisor.py).

    .venv/bin/python -m learning.practice demos --episodes 30        # supervisor demonstrations
    .venv/bin/python -m learning.practice bc --steps 3000             # imitation on everything collected
    .venv/bin/python -m learning.practice dagger --rounds 3 --episodes 15
    .venv/bin/python -m learning.practice rl --episodes 150           # RLPD-style SAC with interventions
    .venv/bin/python -m learning.practice eval --episodes 30 --policy latest|scripted|expert

Every attempt is stored as an episode (learning/episodes.py), every number goes to
runs/practice_<skill>_<time>/metrics.jsonl, and `on_event` (tools/viewer.py's Practice panel) hears each episode.

The RL stage is HIL-SERL's recipe at small scale: an off-policy actor-critic (SAC) whose every batch is half
demonstrations (and interventions), half its own experience (RLPD); a sparse reward (success, from the simulator, the
geometric check or the trained classifier); the supervisor takes over when the policy drifts too far, and those
steps go into both halves. The actor starts from the imitation policy.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import time

import numpy as np
import torch
from torch.nn import functional as F

from learning.episodes import Episode, EpisodeStore
from learning.policy import Actor, Critic, LocalPolicy, device, load, prep, save

ROOT = pathlib.Path(__file__).resolve().parents[1]


class Practice:
    """One practice session on an environment with a supervisor."""

    def __init__(self, env, supervisor, store_root=None, on_event=None, log_dir: pathlib.Path | None = None):
        self.env, self.sup = env, supervisor
        self.skill = env.skill
        self.store = EpisodeStore(self.skill, store_root)
        self.on_event = on_event
        self.log_dir = log_dir or ROOT / 'runs' / f'practice_{self.skill}_{time.strftime("%Y%m%d_%H%M%S")}'
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, d: dict) -> None:
        d = dict(t=round(time.time(), 2), **d)
        with open(self.log_dir / 'metrics.jsonl', 'a') as f:
            f.write(json.dumps(d, default=str) + '\n')
        if self.on_event:
            self.on_event('practice', d)

    # ---------- one episode ----------
    def episode(self, act, source: str, label_expert: bool = False, intervene: bool = False,
                save_it: bool = True, reward_model=None) -> dict:
        """Run one attempt with `act(obs) -> action`. With `label_expert`, the supervisor's action is stored for
        every step (DAgger); with `intervene`, the supervisor may take over."""
        env = self.env
        obs = self.sup.reset(env)
        ep = Episode(self.skill, meta=dict(offset=np.round(getattr(env, 'offset', np.zeros(3)), 4).tolist()))
        done, info, n_int = False, {}, 0
        frames = [obs['wrist']]
        while not done:
            a = np.asarray(act(obs), float)
            take = self.sup.intervene(env, a) if intervene else None
            label = self.sup.demo(env) if label_expert else None
            if take is not None:
                a, label, n_int = take, take, n_int + 1
            nxt, r, done, info = env.step(a)
            if reward_model is not None:
                r = float(reward_model.prob(nxt['wrist']) > 0.5)
            ep.add(obs, a, label, take is not None, r)
            obs = nxt
            frames.append(obs['wrist'])
        ok = self.sup.judge(env)
        views = env.final_views() if save_it and hasattr(env, 'final_views') else None
        eid = self.store.save(ep, views, source=source, success=ok, judged_by=self.sup.name,
                              relabelled=label_expert) if save_it else ''
        out = dict(event='episode', source=source, success=ok, steps=len(ep), interventions=n_int, id=eid,
                   dropped=bool(info.get('dropped')))
        self.log(out)
        if self.on_event:
            self.on_event('practice_frames', dict(id=eid, frames=frames[::max(1, len(frames) // 6)]))
        return out

    # ---------- stages ----------
    def demos(self, n: int) -> float:
        res = [self.episode(lambda o: self.sup.demo(self.env), 'expert' if self.sup.name == 'sim' else 'demo',
                            label_expert=True) for _ in range(n)]
        return self._rate(res, 'demos')

    def evaluate(self, n: int, policy: str = 'latest', save_it: bool = False) -> float:
        if policy == 'expert':
            act = lambda o: self.sup.demo(self.env)
        elif policy == 'scripted':
            act = lambda o: self.env.scripted_search()
        else:
            p = load(self.skill, policy)
            if p is None:
                raise RuntimeError(f'no {self.skill} policy "{policy}" in models/')
            act = p.act
        res = [self.episode(act, f'eval_{policy}', save_it=save_it) for _ in range(n)]
        return self._rate(res, f'eval {policy}')

    def bc(self, steps: int, init=None) -> LocalPolicy:
        from learning.train import train_bc
        pol, record = train_bc(self.skill, steps, init=init, log=self.log)
        self.log(dict(event='trained', **record))
        return pol

    def dagger(self, rounds: int, n: int, steps: int) -> None:
        """Roll out the current policy, the supervisor labels what it should have done at every step, retrain."""
        pol = load(self.skill) or self.bc(steps)
        for k in range(rounds):
            res = [self.episode(pol.act, 'dagger', label_expert=True) for _ in range(n)]
            self._rate(res, f'dagger round {k + 1} (before retraining)')
            pol = self.bc(steps, init=copy.deepcopy(pol).train())

    def rl(self, n: int, batch: int = 128, utd: int = 2, gamma: float = 0.97, lr: float = 3e-4,
           eval_every: int = 25, eval_n: int = 10, reward_model=None, warmup: int = 400,
           bc_weight: float = 2.0) -> None:
        """RLPD-style SAC with interventions. The critic trains alone for `warmup` updates first, and the actor
        keeps an imitation term on the demonstration half of each batch (weight `bc_weight`): starting from the
        imitation policy, an untrained critic's gradients would otherwise undo it within a few hundred updates."""
        dev = device()
        base = load(self.skill) or LocalPolicy()
        actor = Actor(copy.deepcopy(base).to(dev).train()).to(dev)
        critic = Critic().to(dev)
        target = copy.deepcopy(critic)
        log_alpha = torch.tensor(float(np.log(0.05)), device=dev, dtype=torch.float32, requires_grad=True)
        opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
        opt_c = torch.optim.Adam(critic.parameters(), lr=lr)
        opt_t = torch.optim.Adam([log_alpha], lr=lr)
        demo, online = _Buffer(), _Buffer()
        for e in self.store.filter(success=True):
            demo.add_episode(self.store.load(e['id']))
        self.log(dict(event='rl_start', demo_transitions=len(demo), device=dev))
        best = self.evaluate(eval_n, 'latest') if load(self.skill) else 0.0

        def act(obs):
            img, st = prep(obs['wrist'][None], obs['state'][None], dev)
            with torch.no_grad():
                a, _ = actor(img, st)
            return a[0].cpu().numpy()

        updates = [0]

        def update():
            parts = [b.sample(batch // 2) for b in (demo, online) if len(b)]
            if not parts:
                return None
            n_demo = len(parts[0][0]) if len(demo) else 0
            W, S, A, R, W2, S2, D = (np.concatenate(x) for x in zip(*parts))
            updates[0] += 1
            img, st = prep(W, S, dev, augment=True)
            img2, st2 = prep(W2, S2, dev, augment=True)
            a = torch.as_tensor(A, device=dev)
            r = torch.as_tensor(R, device=dev)
            d = torch.as_tensor(D, device=dev)
            with torch.no_grad():
                a2, logp2 = actor(img2, st2)
                q1t, q2t = target(img2, st2, a2)
                y = r + gamma * (1 - d) * (torch.min(q1t, q2t) - log_alpha.exp() * logp2)
            q1, q2 = critic(img, st, a)
            lc = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            opt_c.zero_grad()
            lc.backward()
            opt_c.step()
            if updates[0] <= warmup:
                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), target.parameters()):
                        pt.mul_(0.995).add_(0.005 * p)
                return float(lc), 0.0
            pa, logp = actor(img, st)
            q1p, q2p = critic(img, st, pa)
            la = (log_alpha.exp().detach() * logp - torch.min(q1p, q2p)).mean()
            if n_demo and bc_weight > 0:
                mu, _ = actor(img[:n_demo], st[:n_demo], deterministic=True)
                la = la + bc_weight * (mu - a[:n_demo]).abs().mean()
            opt_a.zero_grad()
            la.backward()
            opt_a.step()
            lt = -(log_alpha * (logp.detach() - ACTION_ENTROPY).mean())
            opt_t.zero_grad()
            lt.backward()
            opt_t.step()
            with torch.no_grad():
                for p, pt in zip(critic.parameters(), target.parameters()):
                    pt.mul_(0.995).add_(0.005 * p)
            return float(lc), float(la)

        for k in range(1, n + 1):
            env = self.env
            obs = self.sup.reset(env)
            ep = Episode(self.skill)
            done, n_int, losses = False, 0, []
            while not done:
                a = act(obs)
                take = self.sup.intervene(env, a)
                if take is not None:
                    a, n_int = np.asarray(take, float), n_int + 1
                nxt, r, done, info = env.step(a)
                if reward_model is not None:
                    r = float(reward_model.prob(nxt['wrist']) > 0.5)
                tr = (obs['wrist'], obs['state'], a.astype(np.float32), r, nxt['wrist'], nxt['state'], float(done))
                online.add(*tr)
                if take is not None:
                    demo.add(*tr)
                ep.add(obs, a, take, take is not None, r)
                obs = nxt
                for _ in range(utd):
                    out = update()
                    if out:
                        losses.append(out)
            ok = self.sup.judge(env)
            eid = self.store.save(ep, env.final_views() if hasattr(env, 'final_views') else None, source='rl',
                                  success=ok, judged_by=self.sup.name)
            self.log(dict(event='episode', source='rl', episode=k, success=ok, steps=len(ep), interventions=n_int,
                          id=eid, critic_loss=round(float(np.mean([x[0] for x in losses])), 4) if losses else None,
                          alpha=round(float(log_alpha.exp()), 4)))
            if k % eval_every == 0 or k == n:
                pol = actor.base.eval()
                name = f'rl_{time.strftime("%Y%m%d_%H%M%S")}'
                save(pol, self.skill, name, dict(kind='rl', episodes=k, interventions_last=n_int), make_latest=False)
                rate = self._rate([self.episode(lambda o: actor_det(actor, o, dev), 'eval_rl', save_it=False)
                                   for _ in range(eval_n)], f'rl after {k} episodes')
                if rate >= best:
                    best = rate
                    save(pol, self.skill, name, dict(kind='rl', episodes=k, eval_success=rate))
                    self.log(dict(event='promoted', name=name, success=rate))
                actor.base.train()

    def _rate(self, res, what: str) -> float:
        rate = float(np.mean([r['success'] for r in res])) if res else 0.0
        self.log(dict(event='rate', what=what, success=round(rate, 3), n=len(res),
                      interventions=int(sum(r.get('interventions', 0) for r in res))))
        return rate


ACTION_ENTROPY = -5.0


def actor_det(actor: Actor, obs, dev):
    img, st = prep(obs['wrist'][None], obs['state'][None], dev)
    with torch.no_grad():
        a, _ = actor(img, st, deterministic=True)
    return a[0].cpu().numpy()


class _Buffer:
    def __init__(self, cap: int = 200_000):
        self.items, self.cap = [], cap

    def __len__(self):
        return len(self.items)

    def add(self, *t):
        self.items.append(t)
        if len(self.items) > self.cap:
            self.items.pop(0)

    def add_episode(self, z: dict):
        T = len(z['action'])
        for t in range(T):
            t2 = min(t + 1, T - 1)
            self.add(z['wrist'][t], z['state'][t], z['action'][t], float(z['reward'][t]), z['wrist'][t2],
                     z['state'][t2], float(t == T - 1))

    def sample(self, n):
        idx = np.random.randint(0, len(self.items), n)
        cols = list(zip(*[self.items[i] for i in idx]))
        return [np.stack(c).astype(np.float32 if i not in (0, 4) else np.uint8) for i, c in enumerate(cols)]


def make_env(args):
    from core.build import load_config
    from learning.envs import InsertEnv
    lo, hi = (int(v) for v in args.layouts.split('-'))
    return InsertEnv(load_config(), seed=args.seed, offset_m=args.offset_mm / 1000, offset_deg=args.offset_deg,
                     layouts=list(range(lo, hi + 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['demos', 'bc', 'dagger', 'rl', 'eval', 'reward'])
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--policy', default='latest')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--layouts', default='0-9', help='scene seeds to draw episodes from, e.g. 0-9 (train), 100-109 '
                    '(held out)')
    ap.add_argument('--offset-mm', type=float, default=6.0, help='handoff error radius')
    ap.add_argument('--offset-deg', type=float, default=8.0)
    ap.add_argument('--reward', choices=['truth', 'classifier'], default='truth')
    a = ap.parse_args()
    from learning.supervisor import SimSupervisor
    env = make_env(a)
    pr = Practice(env, SimSupervisor(), on_event=lambda k, d: print(json.dumps(d, default=str), flush=True)
                  if k == 'practice' else None)
    if a.stage == 'demos':
        pr.demos(a.episodes)
    elif a.stage == 'bc':
        pr.bc(a.steps)
    elif a.stage == 'dagger':
        pr.dagger(a.rounds, a.episodes, a.steps)
    elif a.stage == 'rl':
        rm = None
        if a.reward == 'classifier':
            from learning.train import load_reward
            rm = load_reward(env.skill)
        pr.rl(a.episodes, reward_model=rm)
    elif a.stage == 'reward':
        from learning.train import train_reward
        train_reward(env.skill, a.steps, log=pr.log)
    else:
        pr.evaluate(a.episodes, a.policy)


if __name__ == '__main__':
    main()
