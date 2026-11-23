"""The overnight training job: everything collected so far -> a better local policy, or the current one kept.

    python -m learning.nightly --skill insert --out ~/hsim_runs/panthera_agent_nightly/<time>

Steps, each written to <out>/report.json and report.md:

1. Inventory: episodes per source (expert, demo, dagger, rl, agent, real), successes, Claude labels and their
   agreement with the recorded outcome (where both exist).
2. Success classifier (HIL-SERL's reward model) retrained on all labelled frames; its held-out accuracy and false
   positive rate (the number that matters for a reward).
3. A candidate policy: imitation on every usable episode, starting from the current policy's weights.
4. Evaluation in the sim (headless) of the candidate and the current policy on held-out layouts the training never
   saw, the same episodes for both (same seeds).
5. Promotion: the candidate becomes `latest` only if it beats the current one by `--margin`; otherwise it is kept
   under its own name and `latest` is untouched. The day's robot runs load `latest`.
6. Optional: a LeRobot export of the successful episodes (learning/export_lerobot.py), for an ACT fine-tune with
   LeRobot's own trainer, when LeRobot is installed.

On the VM: learning/nightly.sh (environment, paths), launched by the job queue (infra/vm_jobs/jobs.yaml,
`panthera_agent_nightly`) or by cron.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import time

import numpy as np


def evaluate(skill: str, policy, layouts: list[int], episodes: int, seed: int = 1234) -> float:
    """Success rate of a policy (callable obs -> action) on the training scene's held-out layouts, privileged
    supervisor judging, no interventions."""
    from core.build import load_config
    from learning.envs import InsertEnv
    env = InsertEnv(load_config(), seed=seed, layouts=layouts)
    ok = 0
    for _ in range(episodes):
        obs, done = env.reset(), False
        while not done:
            obs, _, done, info = env.step(policy(obs))
        ok += bool(env.success())
    return ok / max(episodes, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skill', default='insert')
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('runs') / f'nightly_{time.strftime("%Y%m%d_%H%M%S")}')
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--eval-episodes', type=int, default=60)
    ap.add_argument('--eval-layouts', default='200-229')
    ap.add_argument('--margin', type=float, default=0.03)
    ap.add_argument('--export-lerobot', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report: dict = dict(skill=a.skill, started=time.strftime('%Y-%m-%d %H:%M:%S'))
    log = lambda d: print(json.dumps(d, default=str), flush=True)

    from learning.episodes import EpisodeStore
    store = EpisodeStore(a.skill)
    entries = store.entries()
    by = collections.Counter(e.get('source') for e in entries)
    ok_by = collections.Counter(e.get('source') for e in entries if e.get('success'))
    from learning.label import agreement
    report['inventory'] = dict(episodes=len(entries), by_source=dict(by), successes_by_source=dict(ok_by),
                               claude_labels=agreement(a.skill))
    log(dict(step='inventory', **report['inventory']))

    from learning.train import train_bc, train_reward
    try:
        _, rec = train_reward(a.skill, steps=2000, log=log)
        report['reward_model'] = rec
    except RuntimeError as e:
        report['reward_model'] = dict(skipped=str(e))

    from learning.policy import load, models_dir
    current = load(a.skill)
    name = f'nightly_{time.strftime("%Y%m%d_%H%M%S")}'
    import copy
    cand, rec = train_bc(a.skill, a.steps, init=None if current is None else copy.deepcopy(current).train(),
                         name=name, log=log, make_latest=False)
    report['candidate'] = rec

    lo, hi = (int(v) for v in a.eval_layouts.split('-'))
    layouts = list(range(lo, hi + 1))
    c_rate = evaluate(a.skill, cand.eval().act, layouts, a.eval_episodes)
    cur_rate = evaluate(a.skill, current.act, layouts, a.eval_episodes) if current is not None else None
    report['evaluation'] = dict(layouts=a.eval_layouts, episodes=a.eval_episodes, candidate=c_rate, current=cur_rate)
    promote = cur_rate is None or c_rate >= cur_rate + a.margin
    report['promoted'] = promote
    if promote:
        (models_dir(a.skill) / 'latest.json').write_text(json.dumps(dict(name=name, record=rec), indent=1))
    log(dict(step='evaluation', **report['evaluation'], promoted=promote))

    if a.export_lerobot:
        try:
            from learning.export_lerobot import export
            report['lerobot'] = export(a.skill, a.out / 'lerobot')
        except ImportError as e:
            report['lerobot'] = dict(skipped=f'LeRobot is not installed here ({e})')

    report['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (a.out / 'report.json').write_text(json.dumps(report, indent=1, default=str))
    (a.out / 'report.md').write_text(_markdown(report))
    log(dict(step='done', out=str(a.out)))


def _markdown(r: dict) -> str:
    inv, ev = r['inventory'], r['evaluation']
    lines = [f'# Nightly: {r["skill"]} ({r["started"]})', '',
             f'Episodes: {inv["episodes"]} ({", ".join(f"{k}: {v}" for k, v in inv["by_source"].items())}); '
             f'successes {dict(inv["successes_by_source"])}.', '',
             f'Claude labels vs recorded outcome: {inv["claude_labels"]}.', '',
             f'Success classifier: {r["reward_model"]}.', '',
             f'Candidate `{r["candidate"]["name"]}`: imitation on {r["candidate"]["episodes"]} episodes '
             f'({r["candidate"]["frames"]} frames), {r["candidate"]["seconds"]} s on {r["candidate"]["device"]}.', '',
             f'Held-out layouts {ev["layouts"]}, {ev["episodes"]} episodes each: candidate {ev["candidate"]:.0%}, '
             f'current {"none" if ev["current"] is None else format(ev["current"], ".0%")}. '
             f'{"Promoted to latest." if r["promoted"] else "Not promoted: latest unchanged."}']
    if 'lerobot' in r:
        lines += ['', f'LeRobot export: {r["lerobot"]}']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
