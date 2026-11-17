"""`arm zero`: zero offsets from the zeroing jig.

The arm sits in the jig (drives disabled), and 1 s of encoder counts per
joint is averaged. Re-seat 5 times; the spread must stay under 0.1 deg.
zero_offset = direction * counts / counts_per_rad - q_jig, so that
q = direction * counts / counts_per_rad - zero_offset reads q_jig in the jig.
"""

from __future__ import annotations

import numpy as np
import time

from .common import DEG, StageContext, StageResult, safe_home

SPREAD_LIMIT = 0.1 * DEG


def jig_pose(ctx: StageContext) -> np.ndarray:
    """The configured jig pose, or the home pose when none is set."""
    pose = ctx.arm.raw.get('jig_pose_rad')
    if pose is None:          # the jig holds the arm at its home pose
        return safe_home(ctx.model)
    return np.array([pose[n] for n in ctx.arm.names], float)


def run(ctx: StageContext, reseats: int = 5) -> StageResult:
    """`arm zero`: with the drives disabled and the arm in the jig, take the zero offsets (averaged over reseats).
    """
    client, arm = ctx.client, ctx.arm
    if not ctx.sim and np.any(client.state().status & 4):
        raise RuntimeError('Zeroing requires disabled drives. Restart arm01.launch.py with enable_motion:=false.')
    q_jig = jig_pose(ctx)
    units = getattr(client, 'units', None)
    direction = np.array([j.direction for j in arm.joints], float)
    cpr = np.array([j.counts_per_rev for j in arm.joints], float) / (2 * np.pi)
    samples = []
    for k in range(reseats):
        try:
            client.place_in_jig(q_jig)
        except NotImplementedError:
            input('Re-seat %d/%d: put the arm in the zeroing jig (%s), then press Enter '
                  % (k + 1, reseats, ', '.join('%s %.1f deg' % (n, v / DEG) for n, v in zip(arm.names, q_jig))))
        reads = []
        n_reads = 100 if not ctx.sim else 50
        for _ in range(n_reads):
            state = client.state()
            if not ctx.sim:
                if np.any(state.status & 4):
                    raise RuntimeError('A drive became enabled during zeroing.')
                time.sleep(0.01)
            reads.append(state.raw_counts)
        counts = np.mean(reads, axis=0)
        samples.append(direction * counts / cpr - q_jig)
    samples = np.array(samples)
    offsets = samples.mean(axis=0)
    spread = samples.max(axis=0) - samples.min(axis=0)
    lines = ['%-15s offset %+8.4f deg, re-seat spread %.4f deg' % (n, offsets[i] / DEG, spread[i] / DEG)
             for i, n in enumerate(arm.names)]
    passed = bool(np.all(spread < SPREAD_LIMIT))
    _ = units
    return StageResult('zero', passed, {'spread_deg': (spread / DEG).tolist(), 'reseats': reseats},
                       {'zero_offsets_rad': {n: float(offsets[i]) for i, n in enumerate(arm.names)}},
                       lines)
