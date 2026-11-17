"""Generate manipulation demos: scripted sources, then MimicGen-style generation, in parallel CPU workers.

    python generate.py --mjb <scene>.mjb --out ~/hsim_data/manip/v1 --episodes 1000 [--workers 3] [--sources 60]

Writes only successful episodes to <out>/raw/ in the agreed raw format (episode_XXXXXX.npz/.json), source demos
first (source="scripted", indices 0..), then generated ones (source="mimicgen", indices 100000..).
Progress: <out>/progress.jsonl (one line per attempt), summary: <out>/stats.json.
Headless: run with MUJOCO_GL=egl.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sim"))
sys.path.insert(0, str(HERE))


def episode_meta(env, side, seed, source, ok, extra=None):
    """Metadata saved with each episode."""
    mo, mt = env.meta(env.obj), env.meta(env.target)
    return {"instruction": env.instruction, "object_category": mo["category"], "object_name": mo["asset"],
            "object_label": mo["label"], "target_category": mt["category"], "target_name": mt["asset"],
            "success": bool(ok), "fps": 20, "camera_intrinsics": env.camera_intrinsics(), "seed": int(seed),
            "source": source, "active_side": side, "scene": env.info["scene"], "version": env.info["version"],
            "arm_joint_order": ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"],
            "gripper": "0 = closed, 1 = open (Dex1-1, 94.8 mm pad gap when open)",
            "robot": "Unitree G1 29-dof + Dex1-1 grippers (unitree_ros g1_29dof_mode_15_with_dex1_1), pelvis fixed, legs/waist welded at zero",
            **(extra or {})}


def worker(job):
    """Generate episodes in one process until its quota of successes."""
    wid, mjb, out, kind, quota, seeds, src_files, wrist = job
    import expert as X
    from env import ManipEnv
    from episode_io import Recorder
    env = ManipEnv(mjb, wrist_cams=wrist)
    sources = []
    if kind == "mimicgen":
        import mimicgen as MG
        for f in src_files:
            meta = json.loads(Path(f).with_suffix(".json").read_text())
            sources.append(MG.Source(f, meta))
    rng = np.random.default_rng(wid + 12345)
    made, tried = 0, 0
    log = open(Path(out) / "progress.jsonl", "a")
    t0 = time.time()
    for seed in seeds:
        if made >= quota:
            break
        tried += 1
        t1 = time.time()
        env.plan_fail = None
        env.reset(seed)
        if not env.reset_ok:
            log.write(json.dumps({"w": wid, "kind": kind, "seed": seed, "result": "bad reset"}) + "\n"); continue
        side = X.choose_side(env)
        if kind == "scripted":
            wps = X.plan(env, side)
            traj = X.compute_traj(env, side, wps) if wps is not None else None
            src_id = None
        else:
            cands = [s for s in sources if s.side == side]
            if not cands:
                continue
            src_id = int(rng.integers(len(cands)))
            traj = MG.plan_generated(env, cands[src_id], rng)
        if traj is None:
            log.write(json.dumps({"w": wid, "kind": kind, "seed": seed, "result": "infeasible", "why": env.plan_fail,
                                  "s": round(time.time() - t1, 1)}) + "\n"); log.flush(); continue
        rec = Recorder()
        ok, why = X.play(env, side, traj, rec)
        log.write(json.dumps({"w": wid, "kind": kind, "seed": seed, "result": why, "steps": len(rec),
                              "s": round(time.time() - t1, 1)}) + "\n"); log.flush()
        if ok:
            idx = seed if kind == "scripted" else 100000 + seed
            rec.save(Path(out) / "raw", idx, episode_meta(env, side, seed, kind, ok, {"source_demo": src_id}))
            made += 1
    log.close()
    return {"worker": wid, "kind": kind, "made": made, "tried": tried, "seconds": round(time.time() - t0)}


def run(kind, args, quota, seed0, src_files=None):
    """Split a quota over worker processes and run them."""
    per = [quota // args.workers + (1 if i < quota % args.workers else 0) for i in range(args.workers)]
    jobs = [(i, args.mjb, args.out, kind, per[i], range(seed0 + i, seed0 + 10 ** 6, args.workers), src_files or [], args.wrist_cams)
            for i in range(args.workers) if per[i] > 0]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        return pool.map(worker, jobs)


def main():
    """Generate scripted source demos, then MimicGen-style episodes from them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=1000, help="total successful episodes (sources included)")
    ap.add_argument("--sources", type=int, default=0, help="scripted source demos (default: max(10, episodes // 10))")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--wrist-cams", action="store_true")
    args = ap.parse_args()
    out = Path(os.path.expanduser(args.out))
    (out / "raw").mkdir(parents=True, exist_ok=True)
    args.out = str(out)
    n_src = args.sources or max(10, args.episodes // 10)
    t0 = time.time()
    existing = sorted((out / "raw").glob("episode_0[0-9]*.npz"))
    src_files = [str(p) for p in existing if int(p.stem.split("_")[1]) < 100000]
    stats = {"sources_requested": n_src, "episodes_requested": args.episodes, "workers": args.workers}
    if len(src_files) < n_src:
        stats["scripted"] = run("scripted", args, n_src - len(src_files), seed0=len(src_files) * 7 + 1)
        src_files = [str(p) for p in sorted((out / "raw").glob("episode_*.npz")) if int(p.stem.split("_")[1]) < 100000]
    n_gen = args.episodes - len(src_files)
    if n_gen > 0:
        stats["mimicgen"] = run("mimicgen", args, n_gen, seed0=500000, src_files=src_files)
    raws = list((out / "raw").glob("episode_*.npz"))
    stats.update(total_episodes=len(raws), seconds=round(time.time() - t0),
                 episodes_per_hour=round(len(raws) / max(1e-6, (time.time() - t0) / 3600), 1))
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
