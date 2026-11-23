"""The daily learning cycle of the shared local policy, with Claude growing the task set.

    python -m learning.curriculum init                # the first task (a 24 mm square peg, 1 mm clearance)
    python -m learning.curriculum morning             # practice: demos for new tasks, DAgger rounds, RL if enabled
    python -m learning.curriculum night               # retrain, test on held-out layouts, promote if better, then:
    python -m learning.curriculum step                #   Claude reads the results and adds tasks when ready
    python -m learning.curriculum status              # the tasks and their latest results

State: <data>/curriculum.json (the tasks, their specs, every phase's results, the settings). Log: one file,
<data>/learning.log (PANTHERA_AGENT_LOG), every phase of every day appended in plain lines; nothing else writes
there. Weights: models/local/ (latest.json names the promoted one). On the VM, <data> is ~/hsim_data/panthera_agent
(tools/learning_vm.sh runs the phases there and pulls everything back).

**Morning (practice).** Every active task, in the sim on its training layouts:
- a new task first gets `demos_new_task` demonstrations from the privileged expert (the sim's stand-in for a person);
- then `dagger_rounds` rounds: the current policy tries each task, the expert labels what it should have done at
  every step, and the policy is fine-tuned on everything (a working copy, not promoted);
- then `rl_episodes` RL episodes per task with the expert's interventions (0 by default: in the runs so far RL made
  the policy worse; turn it on to try).
Practice success rates are logged per task.

**Night (imitation and promotion).** One imitation run on every episode so far (all tasks, all days), starting from
the promoted policy; the candidate and the promoted policy are both tested on every task's held-out layouts (the
same episodes for both). The candidate is promoted only if its mean success is higher by `margin` and no task drops
by more than `max_drop`. A task at or above `master_at` on two nights running is marked mastered: still tested each
night and still in the training data, no longer practised every morning. The success classifier is retrained too.

**Step (Claude grows the curriculum).** Right after the night: Claude gets the tasks, their results and trends, and
the ranges a task may take, and answers in a fixed form: an assessment, which tasks to focus the next morning on,
and new tasks to add. The rule is enforced here, not left to Claude: new tasks are accepted only when every active
task is at or above `bar` (else they are logged and dropped), at most `adds_per_day`, each checked against the
ranges (scenes/catalog.py INSERT_RANGES). Only the insertion family can be added today: it is the one with a sim
environment and an expert.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import time

import numpy as np

from learning.episodes import EpisodeStore, data_root

DEFAULTS = dict(
    bar=0.9, master_at=0.95, max_active=6, adds_per_day=2,
    train_layouts=[0, 39], eval_layouts=[200, 229],
    morning=dict(demos_new_task=30, dagger_rounds=2, episodes_per_task=20, finetune_steps=1500, rl_episodes=0),
    night=dict(steps=8000, eval_episodes=40, margin=0.02, max_drop=0.10, mastered_eval_episodes=20),
    model='claude-opus-5-5')

FIRST_TASK = dict(id='square24_1mm', skill='insert', peg_shape='box', peg_width=0.024, peg_height=0.06,
                  peg_colour='red', clearance=0.001, fixture_colour='orange', offset_mm=6.0, offset_deg=8.0)


# ---------- state and log ----------
def state_path() -> pathlib.Path:
    return data_root() / 'curriculum.json'


def log_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get('PANTHERA_AGENT_LOG') or data_root() / 'learning.log')


def log(phase: str, text: str) -> None:
    """One line in the learning log (and on stdout)."""
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} [{phase}] {text}'
    print(line, flush=True)
    log_path().parent.mkdir(parents=True, exist_ok=True)
    with open(log_path(), 'a') as f:
        f.write(line + '\n')


def load_state() -> dict:
    p = state_path()
    if not p.exists():
        raise SystemExit(f'no curriculum at {p}: run `python -m learning.curriculum init` first')
    st = json.loads(p.read_text())
    st['settings'] = _merge(copy.deepcopy(DEFAULTS), st.get('settings', {}))
    return st


def save_state(st: dict) -> None:
    state_path().parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path().with_suffix('.tmp')
    tmp.write_text(json.dumps(st, indent=1, default=float))
    tmp.replace(state_path())


def _merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


def _layouts(pair) -> list[int]:
    return list(range(int(pair[0]), int(pair[1]) + 1))


def _active(st: dict, include_mastered: bool = False) -> list[dict]:
    return [t for t in st['tasks'] if t['status'] == 'active' or (include_mastered and t['status'] == 'mastered')]


def _latest(task: dict, phase: str, key: str = 'success'):
    r = [x for x in task['results'] if x['phase'] == phase and key in x]
    return r[-1][key] if r else None


def _env(spec: dict, layouts: list[int], seed: int):
    from core.build import load_config
    from learning.envs import InsertEnv
    return InsertEnv(load_config(), seed=seed, layouts=layouts, spec=spec)


# ---------- phases ----------
def init(args) -> None:
    if state_path().exists() and not args.force:
        raise SystemExit(f'{state_path()} exists (use --force to start over)')
    st = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), settings={}, days=[],
              tasks=[dict(id=FIRST_TASK['id'], spec=FIRST_TASK, added=time.strftime('%Y-%m-%d'), by='init',
                          status='active', results=[])])
    save_state(st)
    log('init', f'curriculum started with {FIRST_TASK["id"]}: {describe(FIRST_TASK)}')


def describe(spec: dict) -> str:
    return (f'{spec["peg_colour"]} {"round" if spec["peg_shape"] == "cylinder" else "square"} peg '
            f'{spec["peg_width"] * 1000:.0f} mm x {spec["peg_height"] * 1000:.0f} mm into a {spec["fixture_colour"]} '
            f'block, {spec["clearance"] * 1000:.1f} mm clearance a side, handoff off by up to {spec["offset_mm"]:.0f} '
            f'mm / {spec["offset_deg"]:.0f} deg')


def morning(args) -> None:
    from learning.policy import SHARED, load
    from learning.practice import Practice
    from learning.supervisor import SimSupervisor
    st = load_state()
    s = st['settings']
    m = s['morning']
    tasks = _active(st)
    focus = set((st['days'][-1].get('step') or {}).get('focus', [])) if st['days'] else set()
    tasks.sort(key=lambda t: (t['id'] not in focus, t['added']))
    day = dict(date=time.strftime('%Y-%m-%d'), morning=dict(started=time.strftime('%H:%M:%S'), tasks={}))
    log('morning', f'practice on {len(tasks)} active task(s): {", ".join(t["id"] for t in tasks)}'
                   + (f' (focus: {", ".join(sorted(focus))})' if focus else ''))
    layouts = _layouts(s['train_layouts'])
    sup = SimSupervisor()
    store = EpisodeStore('insert')

    def practise(task, fn):
        env = _env(task['spec'], layouts, seed=int(time.time()) % 100000)
        try:
            return fn(Practice(env, sup, on_event=None))
        finally:
            env.close()

    # Demonstrations for tasks that have few.
    for t in tasks:
        have = sum(1 for e in store.entries() if e.get('task') == t['id'] and e.get('source') == 'expert')
        need = int(m['demos_new_task']) - have
        if need > 0:
            rate = practise(t, lambda pr: pr.demos(need))
            log('morning', f'{t["id"]}: {need} expert demonstrations ({rate:.0%} succeeded)')
    # DAgger rounds, all tasks each round, then a fine-tune of the working policy on everything.
    from learning.train import train_bc
    pol = load(SHARED)
    if pol is None:
        pol, rec = train_bc(SHARED, int(st['settings']['night']['steps']) // 2, log=lambda d: None)
        log('morning', f'no promoted policy yet: trained a first one on {rec["episodes"]} episodes')
    for r in range(int(m['dagger_rounds'])):
        for t in tasks:
            n = int(m['episodes_per_task']) * (2 if t['id'] in focus else 1)
            res = practise(t, lambda pr: [pr.episode(pol.act, 'dagger', label_expert=True) for _ in range(n)])
            rate = float(np.mean([x['success'] for x in res]))
            day['morning']['tasks'].setdefault(t['id'], []).append(round(rate, 3))
            t['results'].append(dict(date=day['date'], phase='morning', round=r + 1, success=round(rate, 3), n=n))
            log('morning', f'round {r + 1}: {t["id"]} {rate:.0%} of {n} (the working policy, before this round\'s '
                           f'fine-tune)')
        pol, rec = train_bc(SHARED, int(m['finetune_steps']), init=copy.deepcopy(pol).train(), log=lambda d: None,
                            name=f'morning_{time.strftime("%Y%m%d_%H%M%S")}', make_latest=False)
        log('morning', f'round {r + 1}: fine-tuned the working copy on {rec["episodes"]} episodes '
                       f'({rec["frames"]} frames, {rec["seconds"]} s on {rec["device"]})')
    if int(m['rl_episodes']) > 0:
        for t in tasks:
            practise(t, lambda pr: pr.rl(int(m['rl_episodes'])))
            log('morning', f'{t["id"]}: {m["rl_episodes"]} RL episodes with interventions (see the practice log)')
    day['morning']['finished'] = time.strftime('%H:%M:%S')
    day['morning']['episodes_total'] = len(store.entries())
    st['days'].append(day)
    save_state(st)
    log('morning', f'done; {len(store.entries())} insertion episodes in all')


def night(args) -> None:
    from learning.policy import SHARED, load, models_dir
    from learning.train import train_bc, train_reward
    st = load_state()
    s, n = st['settings'], st['settings']['night']
    today = time.strftime('%Y-%m-%d')
    day = st['days'][-1] if st['days'] and st['days'][-1]['date'] == today else dict(date=today)
    if day not in st['days']:
        st['days'].append(day)
    log('night', 'imitation on every episode so far, then held-out tests')
    try:
        _, rec = train_reward('insert', steps=2000, log=lambda d: None)
        log('night', f'success classifier: {rec["accuracy"]:.1%} held-out accuracy, {rec["false_positive_rate"]:.1%} '
                     f'false positives ({rec["frames"]} frames)')
    except RuntimeError as e:
        log('night', f'success classifier skipped: {e}')
    current = load(SHARED)
    name = f'night_{time.strftime("%Y%m%d_%H%M%S")}'
    cand, rec = train_bc(SHARED, int(n['steps']), init=None if current is None else copy.deepcopy(current).train(),
                         name=name, log=lambda d: None, make_latest=False)
    log('night', f'candidate {name}: {rec["episodes"]} episodes, {rec["frames"]} frames, tasks {rec["tasks"]}, '
                 f'{rec["seconds"]} s on {rec["device"]}')
    layouts = _layouts(s['eval_layouts'])
    results = {}
    for t in _active(st, include_mastered=True):
        k = int(n['mastered_eval_episodes'] if t['status'] == 'mastered' else n['eval_episodes'])
        results[t['id']] = dict(candidate=_evaluate(t['spec'], cand.eval().act, layouts, k),
                                current=None if current is None else _evaluate(t['spec'], current.act, layouts, k),
                                n=k)
        r = results[t['id']]
        log('night', f'{t["id"]}: candidate {r["candidate"]:.0%}, promoted policy '
                     f'{"none" if r["current"] is None else format(r["current"], ".0%")} ({k} held-out episodes)')
    cm = float(np.mean([r['candidate'] for r in results.values()]))
    if current is None:
        promote, why = True, 'no promoted policy yet'
    else:
        um = float(np.mean([r['current'] for r in results.values()]))
        drops = [tid for tid, r in results.items() if r['candidate'] < r['current'] - n['max_drop']]
        promote = cm >= um + n['margin'] and not drops
        why = (f'mean {cm:.0%} vs {um:.0%}' + (f'; drops on {", ".join(drops)}' if drops else ''))
    if promote:
        (models_dir(SHARED) / 'latest.json').write_text(json.dumps(dict(name=name, record=rec), indent=1))
    log('night', f'{"PROMOTED " + name if promote else "not promoted, latest unchanged"} ({why})')
    for t in _active(st, include_mastered=True):
        r = results[t['id']]
        score = r['candidate'] if promote or r['current'] is None else r['current']
        t['results'].append(dict(date=today, phase='night', success=round(score, 3), n=r['n'],
                                 candidate=round(r['candidate'], 3), promoted=promote))
        nights = [x['success'] for x in t['results'] if x['phase'] == 'night'][-2:]
        if t['status'] == 'active' and len(nights) == 2 and min(nights) >= s['master_at']:
            t['status'] = 'mastered'
            log('night', f'{t["id"]} mastered ({nights[0]:.0%}, {nights[1]:.0%})')
    day['night'] = dict(finished=time.strftime('%H:%M:%S'), candidate=name, promoted=promote, results=results)
    save_state(st)
    if not args.no_step:
        step(args)


def _evaluate(spec: dict, policy, layouts: list[int], episodes: int, seed: int = 4242) -> float:
    env = _env(spec, layouts, seed)
    try:
        ok = 0
        for _ in range(episodes):
            obs, done = env.reset(), False
            while not done:
                obs, _, done, _ = env.step(policy(obs))
            ok += bool(env.success())
        return ok / max(episodes, 1)
    finally:
        env.close()


# ---------- the curriculum step: Claude ----------
STEP_SCHEMA = {'type': 'object', 'additionalProperties': False,
               'required': ['assessment', 'focus', 'add', 'notes'],
               'properties': {
                   'assessment': {'type': 'string'},
                   'focus': {'type': 'array', 'items': {'type': 'string'}},
                   'add': {'type': 'array', 'items': {
                       'type': 'object', 'additionalProperties': False,
                       'required': ['id', 'peg_shape', 'peg_width_mm', 'peg_height_mm', 'peg_colour', 'clearance_mm',
                                    'fixture_colour', 'offset_mm', 'offset_deg', 'why'],
                       'properties': {
                           'id': {'type': 'string'},
                           'peg_shape': {'type': 'string', 'enum': ['box', 'cylinder']},
                           'peg_width_mm': {'type': 'number'}, 'peg_height_mm': {'type': 'number'},
                           'peg_colour': {'type': 'string'}, 'clearance_mm': {'type': 'number'},
                           'fixture_colour': {'type': 'string'}, 'offset_mm': {'type': 'number'},
                           'offset_deg': {'type': 'number'}, 'why': {'type': 'string'}}}},
                   'notes': {'type': 'string'}}}


def _key():
    """ANTHROPIC_API_KEY from the environment, the repo's .env, or <data>/anthropic.env (the VM's copy)."""
    if not os.environ.get('ANTHROPIC_API_KEY'):
        from core.build import load_dotenv
        load_dotenv()
        f = data_root() / 'anthropic.env'
        if not os.environ.get('ANTHROPIC_API_KEY') and f.exists():
            load_dotenv(f)
    return bool(os.environ.get('ANTHROPIC_API_KEY'))


def step(args) -> None:
    from scenes.catalog import INSERT_RANGES
    st = load_state()
    s = st['settings']
    tasks = _active(st, include_mastered=True)
    ready = all((_latest(t, 'night') or 0) >= s['bar'] for t in _active(st))
    summary = [dict(id=t['id'], status=t['status'], task=describe(t['spec']), added=t['added'],
                    nights=[x['success'] for x in t['results'] if x['phase'] == 'night'][-5:],
                    morning_practice=[x['success'] for x in t['results'] if x['phase'] == 'morning'][-4:])
               for t in tasks]
    r = INSERT_RANGES
    prompt = (
        'You run the curriculum of a robot arm\'s shared local policy: one small network that finishes contact '
        'skills (today: putting a held peg into a hole) from a wrist camera, trained every night on all practice '
        'so far. Each task is a peg and a fixture; the arm hands over with the peg above the hole, off by up to '
        'offset_mm and offset_deg; the policy must find the hole and insert.\n\n'
        f'Tasks and results (success on held-out layouts each night, newest last):\n{json.dumps(summary, indent=1)}\n\n'
        f'Rule (enforced by the code): new tasks are accepted only when every active task is at or above '
        f'{s["bar"]:.0%} on its latest night; at most {s["adds_per_day"]} a day; at most {s["max_active"]} active '
        f'tasks. Every active task is {"" if ready else "NOT "}at the bar now.\n'
        f'Allowed ranges: shape {r["peg_shape"]}; width {r["peg_width"][0] * 1000:.0f}-{r["peg_width"][1] * 1000:.0f} '
        f'mm; height {r["peg_height"][0] * 1000:.0f}-{r["peg_height"][1] * 1000:.0f} mm; clearance '
        f'{r["clearance"][0] * 1000:.1f}-{r["clearance"][1] * 1000:.1f} mm a side; handoff error up to '
        f'{r["offset_mm"][1]:.0f} mm and {r["offset_deg"][1]:.0f} deg; colours {", ".join(r["colours"])} (peg and '
        f'fixture different).\n\n'
        'Decide: (1) an honest assessment of how the policy is doing and why; (2) which tasks the next morning\'s '
        'practice should focus on (ids); (3) if and only if every active task is at the bar, one or two new tasks '
        'that extend what it can do one step at a time (a new shape or size, a new colour so it does not lean on '
        'colour, a tighter fit, a larger handoff error), each with a short id and why; otherwise add none. Keep '
        'steps small: a task far beyond the current ones teaches nothing.')
    if not _key():
        log('step', 'skipped: no ANTHROPIC_API_KEY (the environment, the repo .env, or <data>/anthropic.env)')
        return
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(model=s.get('model', DEFAULTS['model']), max_tokens=4000,
                                  messages=[{'role': 'user', 'content': prompt}],
                                  output_config={'format': {'type': 'json_schema', 'schema': STEP_SCHEMA}})
    text = next(b.text for b in resp.content if getattr(b, 'type', '') == 'text')
    out = json.loads(text)
    log('step', f'Claude: {out["assessment"]}')
    if out.get('focus'):
        log('step', f'focus next morning: {", ".join(out["focus"])}')
    added, refused = [], []
    for a in out.get('add', [])[:int(s['adds_per_day'])]:
        spec, why = _spec_from(a, r, st)
        if not ready:
            refused.append(f'{a.get("id")} (not every active task is at the bar)')
        elif len(_active(st)) >= s['max_active']:
            refused.append(f'{a.get("id")} (already {s["max_active"]} active tasks)')
        elif spec is None:
            refused.append(f'{a.get("id")} ({why})')
        else:
            st['tasks'].append(dict(id=spec['id'], spec=spec, added=time.strftime('%Y-%m-%d'), by='claude',
                                    why=a.get('why', ''), status='active', results=[]))
            added.append(spec['id'])
            log('step', f'added {spec["id"]}: {describe(spec)} ({a.get("why", "")})')
    for x in refused:
        log('step', f'refused {x}')
    if out.get('notes'):
        log('step', f'notes: {out["notes"]}')
    today = time.strftime('%Y-%m-%d')
    day = st['days'][-1] if st['days'] and st['days'][-1]['date'] == today else None
    if day is None:
        day = dict(date=today)
        st['days'].append(day)
    day['step'] = dict(assessment=out['assessment'], focus=out.get('focus', []), added=added, refused=refused,
                       notes=out.get('notes', ''), usage=dict(input=resp.usage.input_tokens,
                                                             output=resp.usage.output_tokens))
    save_state(st)


def _spec_from(a: dict, r: dict, st: dict):
    """A proposed task as a spec, or (None, why not)."""
    import re
    tid = re.sub(r'[^a-z0-9_]+', '_', str(a.get('id', '')).lower()).strip('_')[:40]
    if not tid or any(t['id'] == tid for t in st['tasks']):
        return None, 'missing or repeated id'
    spec = dict(id=tid, skill='insert', peg_shape=a['peg_shape'], peg_width=a['peg_width_mm'] / 1000,
                peg_height=a['peg_height_mm'] / 1000, peg_colour=str(a['peg_colour']).lower(),
                clearance=a['clearance_mm'] / 1000, fixture_colour=str(a['fixture_colour']).lower(),
                offset_mm=float(a['offset_mm']), offset_deg=float(a['offset_deg']))
    checks = [(spec['peg_shape'] in r['peg_shape'], 'shape'),
              (r['peg_width'][0] <= spec['peg_width'] <= r['peg_width'][1], 'width out of range'),
              (r['peg_height'][0] <= spec['peg_height'] <= r['peg_height'][1], 'height out of range'),
              (r['clearance'][0] <= spec['clearance'] <= r['clearance'][1], 'clearance out of range'),
              (r['offset_mm'][0] <= spec['offset_mm'] <= r['offset_mm'][1], 'handoff error out of range'),
              (r['offset_deg'][0] <= spec['offset_deg'] <= r['offset_deg'][1], 'handoff turn out of range'),
              (spec['peg_colour'] in r['colours'] and spec['fixture_colour'] in r['colours'], 'colour'),
              (spec['peg_colour'] != spec['fixture_colour'], 'same colours')]
    bad = [why for ok, why in checks if not ok]
    return (None, ', '.join(bad)) if bad else (spec, '')


def status(args) -> None:
    st = load_state()
    for t in st['tasks']:
        nights = [x['success'] for x in t['results'] if x['phase'] == 'night']
        print(f'{t["id"]:<22} {t["status"]:<9} added {t["added"]} by {t["by"]:<7} nights '
              f'{" ".join(f"{v:.0%}" for v in nights[-6:]) or "-"}  {describe(t["spec"])}')
    from learning.policy import SHARED, models_dir
    lp = models_dir(SHARED) / 'latest.json'
    print('promoted policy:', json.loads(lp.read_text())['name'] if lp.exists() else 'none')
    print('log:', log_path())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('phase', choices=['init', 'morning', 'night', 'step', 'status'])
    ap.add_argument('--force', action='store_true', help='init: start over')
    ap.add_argument('--no-step', action='store_true', help='night: skip the Claude step')
    args = ap.parse_args()
    {'init': init, 'morning': morning, 'night': night, 'step': step, 'status': status}[args.phase](args)


if __name__ == '__main__':
    main()
