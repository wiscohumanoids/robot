#!/usr/bin/env python3
"""Benchmark the IK solvers in arm_stack.model.ik and write docs/ik_report.md.

1. Random in-limit configurations -> FK targets, solved from
   (i) the nominal seed (mid-range posture) and
   (ii) a seed near the truth (uniform +-0.1 rad per joint, as when tracking).
   Success: position error < 0.1 mm and rotation error < 0.1 deg. Failures are
   classified as singular (sigma_min(J) < 1e-3 at the end), at a joint limit,
   or a local minimum (neither); max_iters / timeouts are counted separately.
2. The same on masked tasks: position-only (3-DOF) and free tool spin (5-DOF).
3. Continuous tracking: a closed 6D path (FK of a smooth periodic joint
   trajectory, 100 Hz, several loops) solved incrementally from the previous
   solution. Reports joint jumps and null-space drift (elbow motion between
   loops for the same tool pose).

  python3 tools/ik_benchmark.py            # 2000 targets
  python3 tools/ik_benchmark.py --quick    # 300 targets, short path
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pinocchio as pin  # noqa: E402

from arm_stack.config import load_arm  # noqa: E402
from arm_stack.model import ik  # noqa: E402
from arm_stack.model.robot import ArmModel  # noqa: E402

DEG = math.pi / 180
OK_POS = 1e-4
OK_ROT = 0.1 * DEG
SING = 1e-3

SOLVERS = [
    ('dls', dict(method='dls')),
    ('lm', dict(method='lm')),
    ('clik', dict(method='clik')),
    ('multistart-lm', dict(method='multistart', base='lm', n_starts=40)),
    ('multistart-clik', dict(method='multistart', base='clik', n_starts=40)),
]


def log(*a):
    print('[ik-bench]', *a, flush=True)


def classify(model, res):
    if res.pos_err < OK_POS and res.rot_err < OK_ROT:
        return 'ok'
    if res.status in ('max_iters', 'timeout'):
        return 'max_iters'
    J = model.jacobian(res.q)
    if np.linalg.svd(J, compute_uv=False)[-1] < SING:
        return 'singular'
    if np.any((res.q - model.lower < 1e-6) | (model.upper - res.q < 1e-6)):
        return 'limit'
    return 'local_min'


def run_set(model, targets, seeds, opts, rng_seed=0):
    rows = []
    rng = np.random.default_rng(rng_seed)
    for T, q0 in zip(targets, seeds):
        o = dict(opts)
        if o.get('method') == 'multistart':
            o['rng'] = rng
        res = ik.solve(model, T, q0, **o)
        lo_ok = bool(np.all(res.q >= model.lower - 1e-12) and np.all(res.q <= model.upper + 1e-12))
        rows.append(dict(ok=res.pos_err < OK_POS and res.rot_err < OK_ROT, time=res.time, iters=res.iters,
                         cls=classify(model, res), in_limits=lo_ok, starts=res.starts,
                         pos=res.pos_err, rot=res.rot_err))
    return rows


def stats(rows):
    t = np.array([r['time'] for r in rows]) * 1e3
    it = np.array([r['iters'] for r in rows])
    cls = {}
    for r in rows:
        cls[r['cls']] = cls.get(r['cls'], 0) + 1
    return dict(n=len(rows), success=float(np.mean([r['ok'] for r in rows])),
                t_med=float(np.median(t)), t_p95=float(np.percentile(t, 95)), t_max=float(t.max()),
                it_med=float(np.median(it)), it_p95=float(np.percentile(it, 95)),
                starts_mean=float(np.mean([r['starts'] for r in rows])),
                limits_ok=bool(all(r['in_limits'] for r in rows)),
                fail=dict((k, v) for k, v in cls.items() if k != 'ok'))


# ----------------------------------------------------------------------------- tracking
def periodic_path(model, rng, steps_per_loop, loops, amp=0.3):
    mid = ik.mid_posture(model)
    half = 0.5 * (model.upper - model.lower)
    harm = np.array([1, 2, 1, 3, 2, 1, 2])[:model.n]
    phase = rng.uniform(0, 2 * np.pi, model.n)
    t = np.arange(steps_per_loop * loops) / steps_per_loop
    Q = mid + amp * half * np.sin(2 * np.pi * harm[None] * t[:, None] + phase)
    # keep J6 inside its narrow range with margin
    return np.clip(Q, model.lower + 0.05, model.upper - 0.05)


def track(model, targets, q0, name, opts, steps_per_loop, dt):
    q = q0.copy()
    out = dict(name=name, ok=0, rate_limited=0, jumps=0)
    Qs, dqs, times = [], [], []
    tracker = None
    if name.startswith('tracker'):
        tracker = ik.IKTracker(model, q0, dt=dt, **opts)
    for T in targets:
        if tracker is not None:
            r = tracker.update(T)
            qn = r.q
            ok = r.reachable
            out['rate_limited'] += r.rate_limited
            times.append(r.ik.time)
        else:
            res = ik.solve(model, T, q, **opts)
            qn, ok = res.q, res.success
            times.append(res.time)
        dqs.append(np.max(np.abs(qn - q)))
        q = qn
        out['ok'] += bool(ok)
        Qs.append(q.copy())
    Qs = np.array(Qs)
    dqs = np.array(dqs)
    out['ok'] /= len(targets)
    out['max_step_rad'] = float(dqs.max())
    out['max_vel_rad_s'] = float(dqs.max() / dt)
    out['jumps'] = int((dqs > 0.05).sum())
    out['t_med_ms'] = float(np.median(times) * 1e3)
    out['t_p95_ms'] = float(np.percentile(times, 95) * 1e3)
    starts = Qs[::steps_per_loop]          # solutions at the loop start: the same tool pose
    E = np.array([model.fk(q, 'elbow').translation for q in starts])
    dq = [float(np.degrees(np.max(np.abs(starts[i + 1] - starts[i])))) for i in range(len(starts) - 1)]
    de = [float(1e3 * np.linalg.norm(E[i + 1] - E[i])) for i in range(len(E) - 1)]
    out['loop_dq_deg'] = dq          # change of q over each loop, same tool pose
    out['loop_delbow_mm'] = de
    out['first_loop_deg'] = dq[0]
    out['later_loop_deg'] = float(np.mean(dq[1:])) if len(dq) > 1 else None
    out['later_loop_elbow_mm'] = float(np.mean(de[1:])) if len(de) > 1 else None
    out['total_drift_deg'] = float(np.degrees(np.max(np.abs(starts[-1] - starts[0]))))
    out['swivel_range_deg'] = float(np.degrees(np.ptp([ik.swivel_angle(model, q) for q in Qs[::10]])))
    return out


def unreachable_demo(model, dt):
    """IKTracker on a straight line from the mid-range tool pose outward (0.6 m),
    orientation fixed. Ground truth per point: multistart LM (40 restarts)."""
    q0 = ik.mid_posture(model)
    M0 = model.fk(q0)
    d = M0.translation / np.linalg.norm(M0.translation)
    tr = ik.IKTracker(model, q0, dt=dt)
    offs = np.linspace(0, 0.6, 121)
    flags, errs, truth = [], [], []
    rng = np.random.default_rng(0)
    for s in offs:
        T = pin.SE3(M0.rotation, M0.translation + s * d)
        for _ in range(5):                      # 5 updates per point: let the rate limit catch up
            r = tr.update(T)
        flags.append(r.reachable)
        errs.append(r.pos_err)
        truth.append(ik.solve(model, T, q0, method='multistart', n_starts=40, rng=rng).success)
    flags, truth = np.array(flags), np.array(truth)
    first = lambda f: float(offs[int(np.argmin(f))]) if not f.all() else None  # noqa: E731
    return dict(first_unreachable_offset_m=first(flags), first_truly_unreachable_m=first(truth),
                agree=float(np.mean(flags == truth)), false_unreachable=int(np.sum(~flags & truth)),
                final_pos_err_m=float(errs[-1]), final_reachable=bool(flags[-1]),
                q_in_limits=bool(np.all(tr.q >= model.lower) and np.all(tr.q <= model.upper)))


# ----------------------------------------------------------------------------- report
def table(results, seed_names, col='seed'):
    L = ['| solver | ' + col + ' | success | median ms | p95 ms | median iters | p95 iters | failures |',
         '|---|---|---|---|---|---|---|---|']
    for name, per in results.items():
        for sn in seed_names:
            s = per.get(sn)
            if not s:
                continue
            f = ', '.join('%s %d' % (k, v) for k, v in sorted(s['fail'].items())) or '-'
            L.append('| %s | %s | %.1f%% | %.2f | %.2f | %.0f | %.0f | %s |'
                     % (name, sn, 100 * s['success'], s['t_med'], s['t_p95'], s['it_med'], s['it_p95'], f))
    return '\n'.join(L)


def fmt_m(x):
    return 'never' if x is None else '%.3f' % x


def write_report(path, R):
    L = ['# arm01 inverse kinematics: solvers and benchmark\n',
         'Generated by `tools/ik_benchmark.py` (%d targets, seed %d, %.0f s). Solvers: '
         '`arm_stack/model/ik.py`; model: `urdf/arm01/arm01.urdf`, tool frame `tool0`. Python 3.10, numpy %s, '
         'Pinocchio %s, single thread. FK %.1f us, FK+Jacobian %.1f us per call.\n'
         % (R['n'], R['seed'], R['seconds'], np.__version__, pin.__version__, R['fk_us'], R['jac_us']),
         'Success = position error < 0.1 mm and rotation error < 0.1 deg. Targets: FK of uniform random '
         'configurations inside the joint limits, so every target is reachable. Seeds: **nominal** = '
         'mid-range posture; **near** = truth + U(+-0.1 rad) per joint (tracking-like). Failure classes '
         '(final iterate): *limit* = a joint at a stop, *singular* = sigma_min(J) < 1e-3, *local_min* = '
         'neither, *max_iters* = iteration cap.\n',
         'Every result of every solver was inside the joint limits: %s.\n' % ('yes' if R['all_in_limits'] else 'NO'),
         '## Solvers\n',
         '| name | method |\n|---|---|',
         '| dls | damped least squares on the log6 twist error, Nakamura manipulability damping, clamp to limits |',
         '| lm | Levenberg-Marquardt, damping lam*E + 1e-6 (Sugihara), accept/reject, projection + active set |',
         '| clik | weighted-least-norm CLIK (Chan-Dubey limit weights), null space: rest posture (mid-range) '
         '+ limit barrier; active set |',
         '| multistart-lm | lm from the seed, then up to 40 random restarts in the limits, first feasible |',
         '| multistart-clik | the same with clik (solutions keep the posture objective) |\n',
         '## Full 6-DOF pose\n', table(R['full'], ['nominal', 'near']), '',
         '## Partial tasks (nominal seed)\n',
         'position = 3-DOF (rotation free); free_z = 5-DOF (spin about the target tool z free). '
         'Errors measured on the constrained components only.\n',
         table(R['masked'], list(next(iter(R['masked'].values())).keys()), col='task'), '',
         '## Continuous tracking\n',
         'Closed path: FK of a periodic joint trajectory (mid-range +- 30%% of half-range, harmonics 1-3), '
         '%d steps per loop at %.0f Hz, %d loops, each step seeded with the previous solution. '
         'jump = a step with max |dq| > 0.05 rad (5 rad/s). For the same tool pose (loop start): the joint '
         'change over each loop (max abs) and the elbow displacement. The first loop includes any transient '
         'toward the rest posture; later loops show drift that keeps accumulating (0 = cyclic).\n'
         % (R['traj']['steps_per_loop'], 1 / R['traj']['dt'], R['traj']['loops']),
         '| solver | step success | max step [rad] | jumps | median ms | p95 ms | first loop [deg] | '
         'later loops [deg/loop] | elbow, later loops [mm/loop] | total drift [deg] |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for t in R['traj']['runs']:
        L.append('| %s | %.1f%% | %.3f | %d | %.2f | %.2f | %.2f | %.3f | %.2f | %.2f |'
                 % (t['name'], 100 * t['ok'], t['max_step_rad'], t['jumps'], t['t_med_ms'], t['t_p95_ms'],
                    t['first_loop_deg'], t['later_loop_deg'], t['later_loop_elbow_mm'], t['total_drift_deg']))
    u = R['unreachable']
    L += ['',
          '`tracker-clik` is `IKTracker` (CLIK, warm start, min 1 null-space step per update, joint speed '
          'capped at min(URDF limit, 1.5 rad/s)); it was rate limited on %d steps.\n'
          % next(t['rate_limited'] for t in R['traj']['runs'] if t['name'].startswith('tracker')),
          '**Unreachable targets.** IKTracker on a straight line from the mid-range tool pose outward (0.6 m, '
          'orientation fixed): it flags the target unreachable from %s m on; multistart (40 restarts) says the '
          'pose becomes unreachable at %s m; the flag agrees with multistart on %.0f%% of the points (%d points '
          'reachable by a restart but not by the local tracker). Beyond reach it holds the closest feasible '
          'pose (%.0f mm short at the end), joints inside limits: %s.\n'
          % (fmt_m(u['first_unreachable_offset_m']), fmt_m(u['first_truly_unreachable_m']), 100 * u['agree'],
             u['false_unreachable'], 1e3 * u['final_pos_err_m'], u['q_in_limits'])]
    L += ['## Findings\n'] + R['findings'] + ['']
    L += ['## Recommendation\n'] + R['recommendation'] + ['']
    L += ['## Optional packages\n', R['packages'], '']
    if os.path.exists(os.path.join(os.path.dirname(path), 'workspace_report.md')):
        L += ['## Workspace\n', 'See [workspace_report.md](workspace_report.md) (reachability, dexterous region, '
              'ISO 9283 poses).\n']
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


def probe_packages():
    found = []
    for name in ('pink', 'trac_ik', 'tracikpy', 'trac_ik_python'):
        try:
            __import__(name)
            found.append(name)
        except Exception:
            pass
    if found:
        return 'Importable: %s (not benchmarked here).' % ', '.join(found)
    return ('TRAC-IK (trac_ik / tracikpy / trac_ik_python) and pink are not importable in WSL. '
            '`pin-pink` and `qpsolvers` are pure Python (~170 kB of wheels), but pink needs a compiled QP '
            'backend (quadprog, proxsuite, daqp, osqp: none installed), so it was not installed or benchmarked. '
            'pink is a differential (velocity-level QP) IK like `clik` here, with inequality limits instead of '
            'WLN weights.')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', default='config/arms/arm01.yaml')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--mask-n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='docs/ik_report.md')
    ap.add_argument('--json', default='docs/ik_benchmark.json')
    a = ap.parse_args(argv)
    if a.quick:
        a.n, a.mask_n = 300, 100
    t0 = time.time()
    arm = load_arm(a.arm)
    model = ArmModel(arm.urdf, arm.names)
    rng = np.random.default_rng(a.seed)
    Qt = rng.uniform(model.lower, model.upper, (a.n, model.n))
    targets = [model.fk(q) for q in Qt]
    nominal = np.tile(ik.mid_posture(model), (a.n, 1))
    near = np.clip(Qt + rng.uniform(-0.1, 0.1, Qt.shape), model.lower, model.upper)
    tt = time.perf_counter()
    for q in Qt[:500]:
        model.fk(q)
    fk_us = (time.perf_counter() - tt) / 500 * 1e6
    kin = ik._Kin(model)
    tt = time.perf_counter()
    for q in Qt[:500]:
        kin.pose_jac(q)
    jac_us = (time.perf_counter() - tt) / 500 * 1e6

    R = dict(n=a.n, seed=a.seed, fk_us=fk_us, jac_us=jac_us, full={}, masked={})
    all_in = True
    for name, opts in SOLVERS:
        R['full'][name] = {}
        for sn, seeds in (('nominal', nominal), ('near', near)):
            rows = run_set(model, targets, seeds, opts)
            s = stats(rows)
            all_in &= s['limits_ok']
            R['full'][name][sn] = s
            log('%-16s %-7s success %.1f%%  med %.2f ms  p95 %.2f ms  %s'
                % (name, sn, 100 * s['success'], s['t_med'], s['t_p95'], s['fail']))
    for mask in ('position', 'free_z'):
        for name, opts in SOLVERS:
            if name in ('dls', 'multistart-clik'):
                continue
            o = dict(opts, mask=mask)
            rows = run_set(model, targets[:a.mask_n], nominal[:a.mask_n], o)
            s = stats(rows)
            all_in &= s['limits_ok']
            R['masked'].setdefault(name, {})[mask] = s
            log('%-16s %-8s success %.1f%%  med %.2f ms' % (name, mask, 100 * s['success'], s['t_med']))
    R['all_in_limits'] = all_in

    # tracking
    dt = 0.01
    spl = 300 if a.quick else 1000
    loops = 3 if a.quick else 4
    Qp = periodic_path(model, np.random.default_rng(a.seed + 11), spl, loops)
    path = [model.fk(q) for q in Qp]
    runs = []
    for name, opts in (('dls', dict(method='dls')), ('lm', dict(method='lm')),
                       ('clik', dict(method='clik')), ('clik min_iters=1', dict(method='clik', min_iters=1)),
                       ('tracker-clik', dict(method='clik'))):
        r = track(model, path, Qp[0], name, opts, spl, dt)
        runs.append(r)
        log('track %-16s ok %.3f max step %.3f jumps %d loop dq %s rate-limited %d'
            % (name, r['ok'], r['max_step_rad'], r['jumps'], np.round(r['loop_dq_deg'], 3), r['rate_limited']))
    R['traj'] = dict(dt=dt, steps_per_loop=spl, loops=loops, runs=runs)
    R['unreachable'] = unreachable_demo(model, dt)
    R['packages'] = probe_packages()
    R['seconds'] = time.time() - t0
    R['findings'], R['recommendation'] = findings(R)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    write_report(a.out, R)
    with open(a.json, 'w') as f:
        json.dump(R, f, indent=1, default=float)
    log('wrote %s in %.0f s' % (a.out, R['seconds']))
    return R


def findings(R):
    F = R['full']
    f = []
    single = {k: F[k]['nominal']['success'] for k in ('dls', 'lm', 'clik')}
    f.append('- From the nominal seed the single-shot solvers reach %s of the targets. Almost all failures '
             'end with a joint at a stop: the solution lies across a joint limit from the seed (J1..J7 ROM, '
             'above all J6 +-35 deg), a local minimum that gradient methods cannot leave. Restarts fix it: '
             'multistart-lm %.1f%%, multistart-clik %.1f%%.'
             % (', '.join('%s %.1f%%' % (k, 100 * v) for k, v in single.items()),
                100 * F['multistart-lm']['nominal']['success'], 100 * F['multistart-clik']['nominal']['success']))
    f.append('- From a near seed (tracking) all solvers converge in a few iterations: %s.'
             % ', '.join('%s %.1f%% (%.2f ms median)' % (k, 100 * F[k]['near']['success'], F[k]['near']['t_med'])
                         for k in ('dls', 'lm', 'clik')))
    runs = {r['name']: r for r in R['traj']['runs']}
    f.append('- Tracking: plain pseudo-inverse solvers (dls, lm) have no preference in the null space, so the '
             'posture is not cyclic: each loop of the same closed tool path leaves the joints %.2f deg (dls) / '
             '%.2f deg (lm) further away (elbow %.1f / %.1f mm per loop), without bound. CLIK with the '
             'rest-posture term moves toward rest during the first loop (%.1f deg) and is then cyclic '
             '(%.3f deg/loop; IKTracker %.3f deg/loop).'
             % (runs['dls']['later_loop_deg'], runs['lm']['later_loop_deg'], runs['dls']['later_loop_elbow_mm'],
                runs['lm']['later_loop_elbow_mm'], runs['clik min_iters=1']['first_loop_deg'],
                runs['clik min_iters=1']['later_loop_deg'], runs['tracker-clik']['later_loop_deg']))
    rec = ['- **Teleop / task-space control:** `IKTracker(model, q_now, dt, method=\'clik\')` (defaults: '
           'max_iters 30, min_iters 1, k_rest 0.05 toward mid-range, k_limits 0.2, WLN on, ns_step 0.05 rad, '
           'joint speed cap min(URDF, 1.5 rad/s)). It is the only option that is both continuous and cyclic.',
           '- **One-shot targets (calibration poses, planning):** `ik.solve(model, T, q_seed, method=\'multistart\', '
           'base=\'clik\', n_starts=40)` (add `select=\'rest\'` to prefer the posture nearest mid-range). '
           '`base=\'lm\'` is the fastest if posture does not matter.',
           '- `lm` is the fastest local solver (best for many warm-started solves, e.g. batch planning; '
           '`solve_batch` is its vectorized form). `dls` is kept as the reference baseline.']
    return f, rec


if __name__ == '__main__':
    main()
