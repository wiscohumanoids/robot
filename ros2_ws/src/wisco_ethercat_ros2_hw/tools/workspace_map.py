#!/usr/bin/env python3
"""Workspace / reachability map of the arm's tool frame.

Voxel grid (default 2 cm) over the tool positions. Per voxel:
  reachable   some tool position in the voxel is reachable,
  frac        fraction of a fixed set of ~uniform orientations (default 50)
              reachable with the tool at the voxel center, by IK,
  manip       max Yoshikawa manipulability sqrt(det J J^T) seen in the voxel.

Method (vectorized numpy, arm_stack.model.ik.BatchKinematics / solve_batch):
  1. forward-sample random configurations inside the limits; each sample marks
     its voxel reachable, updates the voxel's max manipulability, and is kept
     as a seed for the reference orientation nearest to its own;
  2. for every reachable voxel x orientation: batched LM IK to (center, R_k)
     from the best seed in the voxel, then random restarts for the failures;
  3. unsampled voxels on the boundary of the sampled set: position-only IK.

Outputs (default robots/arm01/docs/workspace/): workspace.npz, PNG slices (XY, XZ, YZ)
through the dexterous region, summary.json, and the markdown section
robots/arm01/docs/workspace_report.md. Also places an ISO 9283 cube in the dexterous region
and checks the poses arm_stack.calib.mocap_repeat.iso_poses would use.

  python3 tools/workspace_map.py              # 2 cm, ~minutes
  python3 tools/workspace_map.py --quick      # 4 cm, fewer samples
  python3 tools/workspace_map.py --relax-j6 90   # also map with J6 at +-90 deg
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from arm_stack.config import load_arm  # noqa: E402
from arm_stack.model import ik  # noqa: E402
from arm_stack.model.robot import ArmModel  # noqa: E402

DEG = math.pi / 180


def log(*a):
    print('[workspace]', *a, flush=True)


def reference_orientations(n: int, seed: int = 7) -> np.ndarray:
    """n quaternions spread over SO(3): greedy farthest-point on random rotations."""
    rng = np.random.default_rng(seed)
    cand = rng.normal(size=(20000, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    cand[cand[:, 3] < 0] *= -1
    chosen = [np.array([0.0, 0.0, 0.0, 1.0])]
    dmin = 2 * np.arccos(np.clip(np.abs(cand @ chosen[0]), 0, 1))
    for _ in range(n - 1):
        k = int(np.argmax(dmin))
        chosen.append(cand[k])
        dmin = np.minimum(dmin, 2 * np.arccos(np.clip(np.abs(cand @ cand[k]), 0, 1)))
    return np.array(chosen)


def covering_radius_deg(refs, n=20000, seed=11):
    """Largest angle from a random rotation to its nearest reference (grid resolution)."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    d = np.abs(q @ refs.T).max(axis=1)
    ang = np.degrees(2 * np.arccos(np.clip(d, 0, 1)))
    return float(np.percentile(ang, 99)), float(np.median(ang))


def forward_sample(bk, n_samples, chunk, rng, refs):
    """Random configs -> positions, nearest reference bin and its |dot|, manipulability."""
    P, Q, B, D, W = [], [], [], [], []
    for s in range(0, n_samples, chunk):
        m = min(chunk, n_samples - s)
        Qs = rng.uniform(bk.lower, bk.upper, (m, bk.n))
        p, R, J = bk.fk(Qs, jacobian=True)
        qu = ik.quat_from_matrix(R)
        dots = np.abs(qu @ refs.T)
        b = np.argmax(dots, axis=1)
        P.append(p.astype(np.float32))
        Q.append(Qs.astype(np.float32))
        B.append(b.astype(np.int16))
        D.append(dots[np.arange(m), b].astype(np.float32))
        W.append(ik.manipulability_batch(J).astype(np.float32))
    return (np.concatenate(P), np.concatenate(Q), np.concatenate(B), np.concatenate(D),
            np.concatenate(W))


def batched(bk, p, R, Q0, chunk, **kw):
    out_q = np.empty_like(Q0)
    ok = np.zeros(len(p), bool)
    for s in range(0, len(p), chunk):
        sl = slice(s, s + chunk)
        q, o, _, _ = ik.solve_batch(bk, p[sl], None if R is None else R[sl], Q0[sl], **kw)
        out_q[sl], ok[sl] = q, o
    return out_q, ok


def build_map(model, bk, voxel, n_samples, refs, rng, chunk=100000, ik_chunk=25000, restarts=3,
              bounds=None, iters=40, skip_deg=45.0, dense=30, restart_deg=30.0):
    """skip_deg: in a voxel with >= `dense` samples, an orientation with no sample
    within skip_deg is taken as unreachable without IK (the reachable
    orientation set per voxel is small and densely sampled, so a reachable
    orientation always has samples much closer). Restarts only for failures
    whose nearest sample is within restart_deg."""
    t0 = time.time()
    n_or = len(refs)
    R_refs = ik.matrix_from_quat(refs)
    P, Qs, B, D, W = forward_sample(bk, n_samples, chunk, rng, refs)
    log('forward samples %d in %.1f s' % (len(P), time.time() - t0))
    if bounds is None:
        lo = np.floor(P.min(0) / voxel) * voxel - voxel
        hi = np.ceil(P.max(0) / voxel) * voxel + voxel
    else:
        lo, hi = bounds
        keep = np.all((P >= lo) & (P < hi), axis=1)
        if not keep.all():
            log('%d samples outside the given bounds dropped' % (~keep).sum())
            P, Qs, B, D, W = P[keep], Qs[keep], B[keep], D[keep], W[keep]
    shape = tuple(int(round(x)) for x in (hi - lo) / voxel)
    ijk = np.floor((P - lo) / voxel).astype(np.int64)
    lin = np.ravel_multi_index(ijk.T, shape)
    nvox = int(np.prod(shape))
    hits = np.bincount(lin, minlength=nvox)
    manip = np.zeros(nvox)
    np.maximum.at(manip, lin, W.astype(np.float64))
    # best sample per (voxel, orientation bin)
    key = lin * n_or + B
    order = np.lexsort((-D, key))
    first = np.ones(len(order), bool)
    first[1:] = key[order][1:] != key[order][:-1]
    rep = order[first]
    vox_hit = np.flatnonzero(hits)
    V = len(vox_hit)
    compact = np.full(nvox, -1, np.int64)
    compact[vox_hit] = np.arange(V)
    rep_q = np.zeros((V, n_or, bk.n), np.float32)
    rep_quat = np.zeros((V, n_or, 4), np.float32)
    rep_ok = np.zeros((V, n_or), bool)
    rv = compact[lin[rep]]
    rb = B[rep].astype(np.int64)
    rep_q[rv, rb] = Qs[rep]
    rep_ok[rv, rb] = True
    # quaternion of the rep sample (recompute from its q, exact)
    for s in range(0, len(rep), chunk):
        sl = slice(s, s + chunk)
        _, Rr = bk.fk(Qs[rep[sl]].astype(np.float64))
        rep_quat[rv[sl], rb[sl]] = ik.quat_from_matrix(Rr)
    log('%d voxels hit (of %d), %d seeds' % (V, nvox, len(rep)))
    # seeds: for each (voxel, k) the rep with the largest |dot| with ref k
    seed_idx = np.zeros((V, n_or), np.int64)
    near_deg = np.zeros((V, n_or))
    for s in range(0, V, 4000):
        sl = slice(s, s + 4000)
        dots = np.abs(np.einsum('vbi,ki->vbk', rep_quat[sl], refs.astype(np.float32)))
        dots[~rep_ok[sl]] = -1
        seed_idx[sl] = np.argmax(dots, axis=1)
        near_deg[sl] = np.degrees(2 * np.arccos(np.clip(dots.max(axis=1), 0, 1)))
    centers = lo + (np.array(np.unravel_index(vox_hit, shape)).T + 0.5) * voxel
    vv, kk = np.meshgrid(np.arange(V), np.arange(n_or), indexing='ij')
    vv, kk = vv.ravel(), kk.ravel()
    nd = near_deg[vv, kk]
    try_ = (nd < skip_deg) | (hits[vox_hit][vv] < dense)
    vv, kk, nd = vv[try_], kk[try_], nd[try_]
    seeds = rep_q[vv, seed_idx[vv, kk]].astype(np.float64)
    p_t = centers[vv]
    R_t = R_refs[kk]
    t1 = time.time()
    q_sol, ok = batched(bk, p_t, R_t, seeds, ik_chunk, max_iters=iters, stall_ratio=0.95)
    log('pass 1: %d targets (%d skipped: no sample within %.0f deg), %.1f%% solved in %.1f s'
        % (len(p_t), V * n_or - len(p_t), skip_deg, 100 * ok.mean(), time.time() - t1))
    for r in range(restarts):
        bad = np.flatnonzero(~ok & (nd < restart_deg))
        if bad.size == 0:
            break
        t1 = time.time()
        q0 = rng.uniform(bk.lower, bk.upper, (bad.size, bk.n))
        q2, ok2 = batched(bk, p_t[bad], R_t[bad], q0, ik_chunk, max_iters=iters + 20, stall_ratio=0.95)
        q_sol[bad[ok2]] = q2[ok2]
        ok[bad[ok2]] = True
        log('restart %d: %d tried, %d more solved in %.1f s' % (r + 1, bad.size, ok2.sum(), time.time() - t1))
    frac_c = np.bincount(vv[ok], minlength=V) / n_or
    # manipulability of the IK solutions
    good = np.flatnonzero(ok)
    for s in range(0, good.size, chunk):
        g = good[s:s + chunk]
        _, _, J = bk.fk(q_sol[g], jacobian=True)
        np.maximum.at(manip, vox_hit[vv[g]], ik.manipulability_batch(J))
    reach = hits > 0
    reach_by_or = np.zeros(nvox, bool)
    reach_by_or[vox_hit[frac_c > 0]] = True
    # boundary shell: position-only IK for unsampled neighbours of sampled voxels
    g3 = reach.reshape(shape)
    dil = g3.copy()
    for ax in range(3):
        for sh in (-1, 1):
            dil |= np.roll(g3, sh, axis=ax)
    shell = np.flatnonzero((dil & ~g3).ravel())
    shell_ok = 0
    if shell.size:
        cs = lo + (np.array(np.unravel_index(shell, shape)).T + 0.5) * voxel
        found = np.zeros(shell.size, bool)
        for r in range(4):
            idx = np.flatnonzero(~found)
            q0 = rng.uniform(bk.lower, bk.upper, (idx.size, bk.n))
            _, o = batched(bk, cs[idx], None, q0, ik_chunk, max_iters=60, stall_ratio=0.95)
            found[idx[o]] = True
        reach[shell[found]] = True
        shell_ok = int(found.sum())
    log('shell: %d boundary voxels tested, %d reachable' % (shell.size, shell_ok))
    frac = np.zeros(nvox)
    frac[vox_hit] = frac_c
    return dict(origin=lo, voxel=voxel, shape=np.array(shape), reachable=reach.reshape(shape),
                frac=frac.reshape(shape).astype(np.float32),
                manip=manip.reshape(shape).astype(np.float32), hits=hits.reshape(shape).astype(np.int32),
                refs=refs, seconds=time.time() - t0, targets=len(p_t), skipped=V * n_or - len(p_t),
                shell_added=shell_ok)


def box_sum(a: np.ndarray, s: int) -> np.ndarray:
    """Sum over every s x s x s window (valid positions), via a 3D integral image."""
    c = np.pad(a, ((1, 0), (1, 0), (1, 0))).astype(np.float64).cumsum(0).cumsum(1).cumsum(2)
    return (c[s:, s:, s:] - c[:-s, s:, s:] - c[s:, :-s, s:] - c[s:, s:, :-s]
            + c[:-s, :-s, s:] + c[:-s, s:, :-s] + c[s:, :-s, :-s] - c[:-s, :-s, :-s])


def largest_cube(frac: np.ndarray, manip: np.ndarray, thr: float):
    """Largest axis-aligned voxel cube where every voxel has frac >= thr.
    Among the largest, the one with the highest mean frac (then manipulability)."""
    good = (frac >= thr).astype(np.float64)
    best = None
    for s in range(min(frac.shape), 0, -1):
        cnt = box_sum(good, s)
        full = cnt >= s ** 3 - 0.5
        if not full.any():
            continue
        mf = box_sum(frac.astype(np.float64), s) / s ** 3
        mm = box_sum(manip.astype(np.float64), s) / s ** 3
        score = np.where(full, mf + 1e-3 * mm / max(mm.max(), 1e-12), -np.inf)
        i = np.unravel_index(int(np.argmax(score)), score.shape)
        best = dict(size_vox=s, corner=i, mean_frac=float(mf[i]), mean_manip=float(mm[i]))
        break
    return best


def iso_points(center, side):
    """ISO 9283 P1..P5: cube center and the 4 corners of the diagonal plane
    (x-,z-)..(x+,z+), each 10% of the plane diagonal in from its corner."""
    h = side / 2
    C = np.array([[-h, -h, -h], [-h, h, -h], [h, h, h], [h, -h, h]])
    return np.vstack([center, center + 0.8 * C])


def voxel_of(m, p):
    i = np.floor((np.asarray(p) - m['origin']) / m['voxel']).astype(int)
    if np.any(i < 0) or np.any(i >= m['shape']):
        return None
    return tuple(i)


def pose_quality(model, q):
    J = model.jacobian(q)
    sv = np.linalg.svd(J, compute_uv=False)
    clear = np.minimum(q - model.lower, model.upper - q)
    return dict(manip=float(np.sqrt(max(np.linalg.det(J @ J.T), 0))), sigma_min=float(sv[-1]),
                cond=float(sv[0] / max(sv[-1], 1e-12)), min_clear_deg=float(np.degrees(clear.min())),
                tightest_joint=model.joint_names[int(np.argmin(clear / (model.upper - model.lower)))])


def current_iso_poses(arm, model):
    from arm_stack.calib import mocap_repeat
    ctx = SimpleNamespace(arm=arm, model=model)
    return mocap_repeat.iso_poses(ctx)


def gravity_ok(arm, model, q, fraction=0.6):
    """The mocap_repeat feasibility rule: gravity torque < fraction x max torque."""
    lim = fraction * np.array([j.max_torque_nm for j in arm.joints])
    return bool(np.all(np.abs(model.gravity(q)) < lim))


def clearance(model, q):
    return float(np.min(np.minimum(q - model.lower, model.upper - q)))


def suggest_iso(arm, model, bk, center, side, rng, n_cand=400, seeds=8, shrink=(1.0, 0.75, 0.5),
                min_clear=8 * DEG):
    """ISO 9283 P1..P5 in the cube with ONE tool orientation, reachable at all five.

    Candidate orientations are those of random configurations whose tool lands
    within 3 cm of the cube center (reachable at P1 by construction). Each is
    tried at the five points by batched IK from random seeds plus the candidate
    configuration; per point the best-conditioned solution that clears the
    limits by min_clear is kept (else the one with most clearance); among candidates whose five poses all clear the limits by
    min_clear the best worst-case manipulability wins (else the best
    clearance). The scored configurations are returned as they are.
    The cube shrinks if no orientation works at its full size.
    """
    center = np.asarray(center, float)
    Qc, Rc = [], []
    for _ in range(40):
        Q = rng.uniform(bk.lower, bk.upper, (200000, bk.n))
        p, R = bk.fk(Q)
        k = np.linalg.norm(p - center, axis=1) < 0.03
        Qc.append(Q[k])
        Rc.append(R[k])
        if sum(len(x) for x in Qc) >= n_cand:
            break
    Qc = np.concatenate(Qc)[:n_cand]
    Rc = np.concatenate(Rc)[:n_cand]
    M = len(Qc)
    for f in shrink:
        pts = iso_points(center, side * f)
        S = seeds
        cand = np.repeat(np.arange(M), 5 * S)
        pt_i = np.tile(np.repeat(np.arange(5), S), M)
        Q0 = rng.uniform(bk.lower, bk.upper, (M * 5 * S, bk.n))
        Q0[::S] = Qc[cand[::S]]
        Qs, ok, _, _ = ik.solve_batch(bk, pts[pt_i], Rc[cand], Q0, max_iters=60, stall_ratio=0.95)
        ok = ok.reshape(M, 5, S)
        Qs = Qs.reshape(M, 5, S, bk.n)
        feas = ok.any(axis=2).all(axis=1)
        if not feas.any():
            log('ISO: no single orientation reaches all 5 points at side %.0f mm' % (1000 * side * f))
            continue
        best = None
        for c in np.flatnonzero(feas):
            qs = []
            for i in range(5):
                good = np.flatnonzero(ok[c, i])
                cl = np.array([clearance(model, Qs[c, i, g]) for g in good])
                mp = np.array([pose_quality(model, Qs[c, i, g])['manip'] for g in good])
                j = good[int(np.argmax(np.where(cl >= min_clear, mp, -1.0 + cl)))]
                qs.append(Qs[c, i, j])
            worst_clear = min(clearance(model, q) for q in qs)
            worst = min(pose_quality(model, q)['manip'] for q in qs)
            key = (worst_clear >= min_clear, worst if worst_clear >= min_clear else worst_clear)
            if best is None or key > best[0]:
                best = (key, c, qs)
        _, c, qs = best
        R = Rc[c]
        qs = np.array(qs)
        quality = [pose_quality(model, q) for q in qs]
        return dict(center=center.tolist(), side_m=side * f, points=pts.tolist(), candidates=M,
                    feasible_orientations=int(feas.sum()), orientation_R=R.round(5).tolist(),
                    q_rad=qs.round(4).tolist(), quality=quality, min_manip=min(x['manip'] for x in quality),
                    gravity_ok=[gravity_ok(arm, model, q) for q in qs])
    return dict(center=center.tolist(), side_m=side, points=iso_points(center, side).tolist(), q_rad=None)


def components(mask: np.ndarray, min_size=1):
    """6-connected components of a boolean grid: list of (size, voxel indices)."""
    lab = np.zeros(mask.shape, np.int32)
    out = []
    nid = 0
    for start in np.argwhere(mask):
        if lab[tuple(start)]:
            continue
        nid += 1
        stack = [tuple(start)]
        lab[stack[0]] = nid
        members = []
        while stack:
            v = stack.pop()
            members.append(v)
            for ax in range(3):
                for d in (-1, 1):
                    w = list(v)
                    w[ax] += d
                    if 0 <= w[ax] < mask.shape[ax]:
                        w = tuple(w)
                        if mask[w] and not lab[w]:
                            lab[w] = nid
                            stack.append(w)
        if len(members) >= min_size:
            out.append((len(members), np.array(members)))
    out.sort(key=lambda x: -x[0])
    return out


def render(m, out_dir, cube, iso_cur, iso_new, tag=''):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        log('matplotlib not available (%s): no PNGs' % e)
        return []
    origin, voxel, shape = m['origin'], m['voxel'], m['shape']
    frac = np.where(m['reachable'], m['frac'], np.nan)
    manip = np.where(m['reachable'], m['manip'], np.nan)
    c = np.array(cube['corner']) + cube['size_vox'] // 2 if cube else np.array(shape) // 2
    axes = {'xy': (2, 0, 1), 'xz': (1, 0, 2), 'yz': (0, 1, 2)}
    names = 'xyz'
    files = []
    for plane, (cut, a, b) in axes.items():
        fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
        sl = [slice(None)] * 3
        sl[cut] = int(c[cut])
        level = origin[cut] + (c[cut] + 0.5) * voxel
        ext = [origin[a], origin[a] + shape[a] * voxel, origin[b], origin[b] + shape[b] * voxel]
        for ax, data, title, cmap in ((axs[0], frac, 'orientation fraction reachable', 'viridis'),
                                      (axs[1], manip, 'max manipulability (Yoshikawa)', 'magma')):
            img = data[tuple(sl)]          # axes (a, b) with a < b
            im = ax.imshow(img.T, origin='lower', extent=ext, cmap=cmap,
                           vmin=0, vmax=np.nanmax(data) if np.isfinite(np.nanmax(data)) else 1)
            fig.colorbar(im, ax=ax, shrink=0.85)
            ax.set_xlabel('%s [m]' % names[a])
            ax.set_ylabel('%s [m]' % names[b])
            ax.set_title('%s, %s = %.2f m' % (title, names[cut], level), fontsize=10)
            ax.plot(0, 0, 'w+', ms=10, mew=2)
            if cube:
                lo = origin + np.array(cube['corner']) * voxel
                s = cube['size_vox'] * voxel
                ax.add_patch(plt.Rectangle((lo[a], lo[b]), s, s, fill=False, ec='red', lw=1.5))
            for pts, mk, col, lab in ((iso_cur, 'o', 'white', 'current iso_poses'),
                                      (iso_new, 's', 'red', 'suggested ISO P1-P5')):
                if pts is not None and len(pts):
                    ax.scatter(pts[:, a], pts[:, b], marker=mk, facecolors='none', edgecolors=col,
                               s=50, lw=1.5, label=lab)
            ax.legend(loc='lower right', fontsize=7, framealpha=0.5)
        fig.suptitle('arm01 tool0 workspace%s, %s slice (voxel %.0f mm; base at +)'
                     % (tag, plane.upper(), voxel * 1000))
        fig.tight_layout()
        path = os.path.join(out_dir, 'workspace%s_%s.png' % (tag.replace(' ', '_'), plane))
        fig.savefig(path, dpi=90)
        plt.close(fig)
        files.append(path)
    return files


def summarize(m, thr_list=(0.1, 0.2, 0.3, 0.4)):
    reach = m['reachable']
    frac = m['frac'][reach]
    vol = reach.sum() * m['voxel'] ** 3
    out = dict(reachable_voxels=int(reach.sum()), reachable_volume_m3=float(vol),
               frac_percentiles={str(p): float(np.percentile(frac, p)) for p in (10, 25, 50, 75, 90, 99)},
               frac_max=float(frac.max()),
               volume_by_frac={str(t): float((m['frac'] >= t).sum() * m['voxel'] ** 3) for t in thr_list},
               zero_frac_interior=0)
    # interior holes: reachable voxels with no orientation, fully surrounded by reachable voxels
    g = reach & (m['frac'] == 0)
    inner = reach.copy()
    for ax in range(3):
        for sh in (-1, 1):
            inner &= np.roll(reach, sh, axis=ax)
    out['zero_frac_interior'] = int((g & inner).sum())
    out['zero_frac_total'] = int(g.sum())
    holes = []
    for size, mem in components(g & inner)[:5]:
        c = m['origin'] + (mem.mean(axis=0) + 0.5) * m['voxel']
        holes.append(dict(voxels=int(size), volume_cm3=float(size * m['voxel'] ** 3 * 1e6),
                          centroid_m=c.round(3).tolist(), dist_from_base_m=float(np.linalg.norm(c))))
    out['interior_holes'] = holes
    return out


def j6_comparison(model, bk, a, refs, m_fine):
    """Map actual and relaxed-J6 arms on one (coarse) grid and compare voxel by voxel."""
    k6 = model.joint_names.index('wrist_pitch')
    vc = a.relax_voxel
    if abs(vc - a.voxel) < 1e-12 and a.samples == a.relax_samples:
        mc = m_fine
    else:
        mc = build_map(model, bk, vc, a.relax_samples, refs, np.random.default_rng(a.seed + 2), restarts=1)
    bk2 = ik.BatchKinematics(model)
    bk2.lower[k6], bk2.upper[k6] = -a.relax_j6 * DEG, a.relax_j6 * DEG
    hi = mc['origin'] + mc['shape'] * vc
    m2 = build_map(model, bk2, vc, a.relax_samples, refs, np.random.default_rng(a.seed + 3), restarts=1,
                   bounds=(mc['origin'], hi))
    ra, rr = mc['reachable'], m2['reachable']
    fa, fr = mc['frac'], m2['frac']
    gain = np.where(rr | ra, fr - fa, np.nan)
    holes_filled = ra & (fa == 0) & (fr > 0)
    out = dict(j6_deg=a.relax_j6, voxel_m=vc,
               actual=summarize(mc), relaxed=summarize(m2),
               mean_frac_actual=float(fa[ra].mean()), mean_frac_relaxed=float(fr[rr].mean()),
               frac_ratio=float(fr[rr].mean() / max(fa[ra].mean(), 1e-9)),
               new_position_m3=float((rr & ~ra).sum() * vc ** 3),
               zero_frac_filled=int(holes_filled.sum()), zero_frac_actual=int((ra & (fa == 0)).sum()))
    np.savez_compressed(os.path.join(a.out, 'workspace_j6_%d.npz' % a.relax_j6),
                        **{k: v for k, v in m2.items() if isinstance(v, np.ndarray)})
    files = []
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        c = np.array(mc['shape']) // 2
        fig, axs = plt.subplots(1, 3, figsize=(16, 5))
        names = 'xyz'
        for ax, (cut, x, y) in zip(axs, ((2, 0, 1), (1, 0, 2), (0, 1, 2))):
            sl = [slice(None)] * 3
            sl[cut] = int(c[cut])
            o = mc['origin']
            ext = [o[x], o[x] + mc['shape'][x] * vc, o[y], o[y] + mc['shape'][y] * vc]
            im = ax.imshow(gain[tuple(sl)].T, origin='lower', extent=ext, cmap='RdBu_r', vmin=-0.4, vmax=0.4)
            ax.set_xlabel('%s [m]' % names[x])
            ax.set_ylabel('%s [m]' % names[y])
            ax.set_title('%s = %.2f m' % (names[cut], o[cut] + (c[cut] + 0.5) * vc), fontsize=10)
            ax.plot(0, 0, 'k+', ms=10, mew=2)
        fig.colorbar(im, ax=list(axs), shrink=0.8, label='frac(J6 +-%d) - frac(J6 +-35)' % a.relax_j6)
        fig.suptitle('Orientation fraction gained by widening J6 to +-%d deg (voxel %.0f mm)'
                     % (a.relax_j6, vc * 1000))
        path = os.path.join(a.out, 'workspace_j6_gain.png')
        fig.savefig(path, dpi=90)
        plt.close(fig)
        files.append(path)
    except Exception as e:  # pragma: no cover
        log('no J6 plot: %s' % e)
    log('J6 +-%d: mean frac %.3f -> %.3f, new positions %.3f m^3, zero-frac voxels filled %d of %d'
        % (a.relax_j6, out['mean_frac_actual'], out['mean_frac_relaxed'], out['new_position_m3'],
           out['zero_frac_filled'], out['zero_frac_actual']))
    return out, files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', default='arm01')
    ap.add_argument('--voxel', type=float, default=0.02)
    ap.add_argument('--samples', type=int, default=3000000)
    ap.add_argument('--orientations', type=int, default=50)
    ap.add_argument('--restarts', type=int, default=3)
    ap.add_argument('--quick', action='store_true', help='4 cm voxels, 600k samples, 1 restart')
    ap.add_argument('--dex', type=float, default=None, help='orientation fraction defining the dexterous region')
    ap.add_argument('--relax-j6', type=float, default=None, help='also map with J6 at +-this many deg')
    ap.add_argument('--relax-voxel', type=float, default=0.04, help='grid for the J6 comparison')
    ap.add_argument('--relax-samples', type=int, default=600000)
    ap.add_argument('--out', default=os.path.join(os.path.normpath(os.path.join(ROOT, '..', '..', '..', 'robots', 'arm01')), 'docs', 'workspace'))
    ap.add_argument('--report', default=os.path.join(os.path.normpath(os.path.join(ROOT, '..', '..', '..', 'robots', 'arm01')), 'docs', 'workspace_report.md'))
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    t_start = time.time()
    if a.quick:
        a.voxel, a.samples, a.restarts = 0.04, 600000, 1
    os.makedirs(a.out, exist_ok=True)
    arm = load_arm(a.arm)
    model = ArmModel(arm.urdf, arm.names)
    bk = ik.BatchKinematics(model)
    refs = reference_orientations(a.orientations)
    rng = np.random.default_rng(a.seed)
    m = build_map(model, bk, a.voxel, a.samples, refs, rng, restarts=a.restarts)
    s = summarize(m)
    log('reachable %.3f m^3, frac median %.2f max %.2f' % (s['reachable_volume_m3'],
                                                          s['frac_percentiles']['50'], s['frac_max']))
    dex = a.dex if a.dex is not None else round(0.5 * s['frac_percentiles']['99'], 2)
    cubes = {}
    for t in sorted(set([0.2, 0.3, 0.4, dex])):
        c = largest_cube(m['frac'], m['manip'], t)
        if c:
            c['side_m'] = c['size_vox'] * a.voxel
            c['center'] = (m['origin'] + (np.array(c['corner']) + c['size_vox'] / 2.0) * a.voxel).tolist()
            c['corner'] = [int(x) for x in c['corner']]
        cubes[str(t)] = c
    cube = cubes[str(dex)]
    log('dexterous threshold %.2f: cube %s' % (dex, cube))
    # ISO poses
    iso_q = current_iso_poses(arm, model)
    cur = []
    for q in iso_q:
        p = model.tool_position(q)
        v = voxel_of(m, p)
        d = pose_quality(model, q)
        d.update(q_deg=np.degrees(q).round(1).tolist(), tool_m=p.round(3).tolist(),
                 voxel_frac=float(m['frac'][v]) if v else None,
                 voxel_manip_max=float(m['manip'][v]) if v else None,
                 in_cube=bool(cube and v and all(cube['corner'][i] <= v[i] < cube['corner'][i] + cube['size_vox']
                                                 for i in range(3))))
        cur.append(d)
    sug = None
    if cube:
        sug = suggest_iso(arm, model, bk, cube['center'], cube['side_m'], np.random.default_rng(a.seed + 5))
        log('ISO suggestion: side %.0f mm, %s' % (1000 * sug['side_m'],
                                               'min manip %.4f' % sug['min_manip'] if sug.get('q_rad') else 'none'))
    configured = [list(map(float, q)) for q in (arm.raw.get('validation', {}).get('iso_poses_rad') or [])]
    applied = bool(sug and sug.get('q_rad') and len(configured) == len(sug['q_rad'])
                   and np.allclose(configured, np.round(sug['q_rad'], 4), atol=1e-3))
    files = render(m, a.out, cube, np.array([c['tool_m'] for c in cur]) if cur else None,
                   np.array(sug['points']) if sug else None)
    relaxed = None
    if a.relax_j6:
        relaxed, f2 = j6_comparison(model, bk, a, refs, m)
        files += f2
    np.savez_compressed(os.path.join(a.out, 'workspace.npz'),
                        **{k: v for k, v in m.items() if isinstance(v, np.ndarray)})
    cov99, cov50 = covering_radius_deg(refs)
    summary = dict(voxel_m=a.voxel, samples=a.samples, orientations=a.orientations,
                   grid_cover_deg_p99=cov99, grid_cover_deg_median=cov50,
                   seconds=m['seconds'], ik_targets=m['targets'], shell_added=m['shell_added'],
                   origin=m['origin'].tolist(), shape=m['shape'].tolist(), **s,
                   dex_threshold=dex, cubes=cubes, iso_current=cur, iso_configured=bool(configured), iso_applied=applied, iso_suggested=sug, relaxed_j6=relaxed,
                   images=[os.path.relpath(f, os.path.dirname(a.report) or '.') for f in files])
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    write_report(a.report, summary)
    log('done in %.0f s; wrote %s' % (time.time() - t_start, a.report))
    return summary


def write_report(path, S):
    L = []
    L.append('# arm01 workspace and reachability map\n')
    L.append('Generated by `tools/workspace_map.py` (voxel %.0f mm, %d forward samples, %d reference '
             'orientations, %d IK targets, %.0f s). Data: `robots/arm01/docs/workspace/workspace.npz`.\n'
             % (S['voxel_m'] * 1000, S['samples'], S['orientations'], S['ik_targets'], S['seconds']))
    L.append('Per voxel: *reachable* (some tool position in it), *frac* (share of %d orientations spread '
             'over SO(3) the IK reaches with tool0 at the voxel center), *manip* (max Yoshikawa '
             'sqrt(det JJ^T) of the 6x7 tool Jacobian). Base frame coordinates.\n' % S['orientations'])
    L.append('## Volume\n')
    L.append('| quantity | value |\n|---|---|')
    L.append('| reachable volume | %.3f m^3 (%d voxels) |' % (S['reachable_volume_m3'], S['reachable_voxels']))
    for t, v in S['volume_by_frac'].items():
        L.append('| volume with frac >= %s | %.4f m^3 |' % (t, v))
    L.append('| frac percentiles 10/25/50/75/90/99 | %s |' % ' / '.join('%.2f' % S['frac_percentiles'][k]
                                                                     for k in ('10', '25', '50', '75', '90', '99')))
    L.append('| max frac | %.2f |' % S['frac_max'])
    L.append('| reachable voxels with frac = 0 (all / interior) | %d / %d |\n'
             % (S['zero_frac_total'], S['zero_frac_interior']))
    L.append('## Dexterous region and ISO 9283 cube\n')
    L.append('Largest axis-aligned cube whose voxels all reach at least the given orientation fraction:\n')
    L.append('| frac >= | side | center [m] | mean frac |\n|---|---|---|---|')
    for t, c in S['cubes'].items():
        if c:
            L.append('| %s | %.0f mm | (%.3f, %.3f, %.3f) | %.2f |' % (t, c['side_m'] * 1000, *c['center'],
                                                                     c['mean_frac']))
        else:
            L.append('| %s | none | | |' % t)
    L.append('\nDexterous threshold used: frac >= %.2f.\n' % S['dex_threshold'])
    sug = S.get('iso_suggested')
    if sug and sug.get('q_rad'):
        L.append('Suggested ISO 9283 poses: cube side %.0f mm centered at (%.3f, %.3f, %.3f); P1 center, '
                 'P2-P5 on the diagonal plane 10%% in from its corners; one tool orientation for all five '
                 '(best worst-case manipulability among %d candidate orientations, %d feasible at all five); '
                 'per point the best-conditioned IK solution at least 8 deg from every joint limit. Gravity load under 60%% of max torque '
                 '(the mocap_repeat rule): %s.\n'
                 % (1000 * sug['side_m'], *sug['center'], sug['candidates'], sug['feasible_orientations'],
                    'all' if all(sug['gravity_ok']) else sug['gravity_ok']))
        L.append('| pose | tool [m] | q [deg] | manip | sigma_min | cond | min clearance |\n'
                 '|---|---|---|---|---|---|---|')
        for i, (p, q, qq) in enumerate(zip(sug['points'], sug['q_rad'], sug['quality'])):
            L.append('| P%d | (%.3f, %.3f, %.3f) | %s | %.4f | %.3f | %.0f | %.0f deg (%s) |'
                     % (i + 1, *p, ', '.join('%.0f' % d for d in np.degrees(q)), qq['manip'], qq['sigma_min'],
                        qq['cond'], qq['min_clear_deg'], qq['tightest_joint']))
        L.append('\nFor `validation.iso_poses_rad` in robots/arm01/config/arm01.yaml (%s):\n'
                 % ('applied' if S.get('iso_applied') else 'not applied'))
        L.append('```yaml\niso_poses_rad:\n' + ''.join('  - [%s]\n' % ', '.join('%.4f' % x for x in q)
                                                    for q in sug['q_rad']) + '```\n')
    elif sug:
        L.append('No single tool orientation reaches all five ISO points in the cube.\n')
    L.append('## Current `mocap_repeat.iso_poses` (%s)\n'
             % ('validation.iso_poses_rad from the config' if S.get('iso_configured')
                else 'config list empty: feasible random poses, seed 42'))
    L.append('| pose | tool [m] | voxel frac | manip (voxel max) | sigma_min | cond | min clearance | in cube |'
             '\n|---|---|---|---|---|---|---|---|')
    for i, c in enumerate(S['iso_current']):
        L.append('| %d | (%.3f, %.3f, %.3f) | %s | %.4f (%s) | %.3f | %.0f | %.0f deg (%s) | %s |'
                 % (i + 1, *c['tool_m'], '%.2f' % c['voxel_frac'] if c['voxel_frac'] is not None else '-',
                    c['manip'], '%.4f' % c['voxel_manip_max'] if c['voxel_manip_max'] is not None else '-',
                    c['sigma_min'], c['cond'], c['min_clear_deg'], c['tightest_joint'], c['in_cube']))
    L.append('')
    if S.get('interior_holes'):
        L.append('## Holes\n')
        L.append('Reachable voxels where none of the %d reference orientations is reachable, fully surrounded '
                 'by reachable voxels (largest 6-connected clusters). The reference set is coarse (a random '
                 'rotation is within %.0f deg of its nearest reference at the median, %.0f deg at p99), so these '
                 'are not position holes: the tool gets there, but only with an orientation set narrower than '
                 'the grid spacing.\n' % (S['orientations'], S['grid_cover_deg_median'], S['grid_cover_deg_p99']))
        L.append('| voxels | volume | centroid [m] | distance from base origin |\n|---|---|---|---|')
        for h in S['interior_holes']:
            L.append('| %d | %.0f cm^3 | (%.3f, %.3f, %.3f) | %.2f m |' % (h['voxels'], h['volume_cm3'],
                                                                      *h['centroid_m'], h['dist_from_base_m']))
        L.append('')
    rel = S.get('relaxed_j6')
    if rel:
        L.append('## Effect of the J6 ROM\n')
        L.append('Actual (J6 +-35 deg) vs J6 widened to +-%.0f deg, both on a %.0f mm grid with the same sampling. '
                 'Mean orientation fraction over the reachable voxels: %.3f -> %.3f (x%.1f). Positions only '
                 'reachable with the wider J6: %.3f m^3 (of %.3f). Reachable voxels with frac = 0: %d, of which '
                 '%d gain orientations with the wider J6.\n'
                 % (rel['j6_deg'], rel['voxel_m'] * 1000, rel['mean_frac_actual'], rel['mean_frac_relaxed'],
                    rel['frac_ratio'], rel['new_position_m3'], rel['relaxed']['reachable_volume_m3'],
                    rel['zero_frac_actual'], rel['zero_frac_filled']))
        L.append('| | J6 +-35 | J6 +-%.0f |\n|---|---|---|' % rel['j6_deg'])
        L.append('| reachable volume [m^3] | %.3f | %.3f |' % (rel['actual']['reachable_volume_m3'],
                                                             rel['relaxed']['reachable_volume_m3']))
        for t in rel['actual']['volume_by_frac']:
            L.append('| volume frac >= %s [m^3] | %.4f | %.4f |' % (t, rel['actual']['volume_by_frac'][t],
                                                                   rel['relaxed']['volume_by_frac'][t]))
        L.append('| frac median / p90 / max | %.2f / %.2f / %.2f | %.2f / %.2f / %.2f |\n'
                 % (rel['actual']['frac_percentiles']['50'], rel['actual']['frac_percentiles']['90'],
                    rel['actual']['frac_max'], rel['relaxed']['frac_percentiles']['50'],
                    rel['relaxed']['frac_percentiles']['90'], rel['relaxed']['frac_max']))
    L.append('## Slices\n')
    for f in S['images']:
        L.append('![%s](%s)\n' % (os.path.basename(f), f.replace(os.sep, '/')))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
