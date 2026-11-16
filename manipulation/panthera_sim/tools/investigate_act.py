#!/usr/bin/env python3
"""Reproducible ACT forensics; writes diagnostics without changing checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import torch
from teleop.dataset_contract import DEFAULT_RENDERED, DEFAULT_DATASET, PhysicsClock
from sim.panthera_env import PantheraSim
from sim.stack_task import stack_metrics


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def describe(x):
    x = np.asarray(x)
    return {"n": len(x), "mean": float(x.mean()),
            **{f"p{p}": float(np.percentile(x, p)) for p in (10, 50, 90, 95)}} if x.size else {}


def data_audit(output):
    from tools.replay_act_actions import replay
    episodes = []
    for path in sorted(DEFAULT_RENDERED.glob("episode_*")):
        d = np.load(path / "trajectory.npz")
        raw = np.load(ROOT / "data" / path.name / "data.npz")
        z = d["obj_pos"][:, :, 2]
        up = z > .0975
        first = np.flatnonzero(up.any(1))
        close = np.flatnonzero((d["ctrl"][:-1, 6] >= .01) & (d["ctrl"][1:, 6] < .01)) + 1
        episodes.append({"episode": path.name, "frames": len(d["q"])-1,
                         "seconds": float(d["t"][-1]),
                         "raw_sim_time_recorded": "sim_time" in raw,
                         "raw_fingers_recorded": "finger_q" in raw,
                         "initial_ee": d["ee_pos"][0].tolist(),
                         "initial_objects": d["obj_pos"][0].tolist(),
                         "initial_lifted": bool(up[0].any()),
                         "first_lift_cube": int(np.argmax(up[first[0]])) if len(first) else None,
                         "first_lift_seconds": float(d["t"][first[0]]) if len(first) else None,
                         "close_events": len(close),
                         "close_nearest_cube_mm": (1000*np.linalg.norm(d["ee_pos"][close, None]-d["obj_pos"][close], axis=2).min(1)).tolist(),
                         "final_three_stack": stack_metrics(d["obj_pos"][-1])["three_stack"]})
    summary = {"episodes": len(episodes), "frames": sum(r["frames"] for r in episodes),
               "seconds": describe([r["seconds"] for r in episodes]),
               "final_three_stack": sum(r["final_three_stack"] for r in episodes),
               "initial_lifted": sum(r["initial_lifted"] for r in episodes),
               "first_lift_cube": dict(Counter(str(r["first_lift_cube"]) for r in episodes)),
               "first_lift_seconds": describe([r["first_lift_seconds"] for r in episodes if r["first_lift_seconds"] is not None]),
               "close_nearest_cube_mm": describe([x for r in episodes for x in r["close_nearest_cube_mm"]]),
               "recorded_fingers": sum(r["raw_fingers_recorded"] for r in episodes),
               "recorded_sim_time": sum(r["raw_sim_time_recorded"] for r in episodes)}
    save(output / "data.json", {"summary": summary, "episodes": episodes})
    print("DATA", json.dumps(summary), flush=True)
    paths = sorted(DEFAULT_RENDERED.glob("episode_*"))
    rows = []
    for i in np.linspace(0, len(paths)-1, 32, dtype=int):
        rows.append(replay(paths[i]))
    save(output / "replay.json", rows)
    print("REPLAY", {k: sum(r[k] for r in rows) for k in ("grasped", "lifted", "final_three_stack")}, flush=True)


def load_policy(checkpoint):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ACTPolicy.from_pretrained(checkpoint, config=config).to(config.device).eval()
    pre, post = make_pre_post_processors(config, pretrained_path=str(checkpoint))
    return policy, pre, post


def offline(output, checkpoints):
    import pyarrow.parquet as pq
    tables = [pq.read_table(p, columns=["observation.state", "action", "episode_index", "index"])
              for p in sorted((DEFAULT_DATASET / "data").glob("**/*.parquet"))]
    state = np.concatenate([np.array(t["observation.state"].to_pylist(), np.float32) for t in tables])
    action = np.concatenate([np.array(t["action"].to_pylist(), np.float32) for t in tables])
    ep = np.concatenate([np.array(t["episode_index"]) for t in tables])
    idx = np.concatenate([np.array(t["index"]) for t in tables])
    assert np.array_equal(idx, np.arange(len(idx)))
    cache = np.load(DEFAULT_DATASET / "cache/decoded_images.uint8.npy", mmap_mode="r")
    held = json.loads((ROOT / "outputs/act/panthera_stack_30hz/experiment.json").read_text())["held_out_episode_indices"]
    sim = PantheraSim()
    fk_data = mujoco.MjData(sim.model)

    def fk(q):
        fk_data.qpos[sim.arm_qadr] = q
        mujoco.mj_kinematics(sim.model, fk_data)
        return fk_data.site_xpos[sim.ee_site].copy(), fk_data.site_xmat[sim.ee_site].reshape(3, 3).copy()

    results = []
    for checkpoint in checkpoints:
        policy, pre, post = load_policy(checkpoint)
        for split, pool in [("train", np.flatnonzero(~np.isin(ep, held))), ("val", np.flatnonzero(np.isin(ep, held)))]:
            chosen = pool[np.linspace(0, len(pool)-1, 1024, dtype=int)]
            predictions, shuffled = [], []
            with torch.inference_mode():
                for start in range(0, len(chosen), 16):
                    ids = chosen[start:start+16]
                    batch = {"observation.state": torch.from_numpy(state[ids]),
                             **{f"observation.images.{camera}": torch.from_numpy(np.array(cache[ids, c])).float()/255
                                for c, camera in enumerate(("shoulder", "wrist"))}}
                    batch = pre(batch)
                    predictions.append(post(policy.predict_action_chunk(batch)).cpu().numpy())
                    for camera in ("shoulder", "wrist"):
                        key = f"observation.images.{camera}"
                        batch[key] = batch[key].roll(1, 0)
                    shuffled.append(post(policy.predict_action_chunk(batch)).cpu().numpy())
            pred, shuf = np.concatenate(predictions), np.concatenate(shuffled)
            horizons = {}
            for h in (0, 2, 9, 29):
                valid = (chosen+h < len(ep)) & (ep[np.minimum(chosen+h, len(ep)-1)] == ep[chosen])
                ids, p = chosen[valid], pred[valid, h]
                target = action[ids+h]
                ee_err, angle_err, demo_move = [], [], []
                for i, ph, th in zip(ids, p, target):
                    pp, pr = fk(ph[:6]); tp, tr = fk(th[:6]); sp, _ = fk(state[i, :6])
                    ee_err.append(np.linalg.norm(pp-tp)*1000)
                    angle_err.append(np.degrees(np.arccos(np.clip((np.trace(pr.T@tr)-1)/2, -1, 1))))
                    demo_move.append(np.linalg.norm(tp-sp)*1000)
                move = target[:, :6]-state[ids, :6]
                predicted_move = p[:, :6]-state[ids, :6]
                moving = np.linalg.norm(move, axis=1) > .025
                closing = (state[ids, 6] >= .01) & (target[:, 6] < .01)
                horizons[str(h)] = {"n": len(ids), "joint_mae_deg": float(np.abs(p[:, :6]-target[:, :6]).mean()*180/np.pi),
                    "copy_state_joint_mae_deg": float(np.abs(move).mean()*180/np.pi),
                    "shuffled_image_joint_mae_deg": float(np.abs(shuf[valid, h, :6]-target[:, :6]).mean()*180/np.pi),
                    "gripper_mae_mm": float(np.abs(p[:, 6]-target[:, 6]).mean()*1000),
                    "fk_target_error_mm": describe(ee_err), "fk_orientation_error_deg": describe(angle_err),
                    "demonstrated_fk_motion_mm": describe(demo_move),
                    "moving_samples": int(moving.sum()),
                    "wrong_direction_fraction_moving": float((np.sum(move[moving]*predicted_move[moving], axis=1)<0).mean()),
                    "close_transition_samples": int(closing.sum()),
                    "close_transition_recall": float((p[closing, 6] < .01).mean()) if closing.any() else None}
            row = {"checkpoint": str(checkpoint), "split": split, "horizons": horizons}
            results.append(row)
            save(output / "offline.json", results)
            print("OFFLINE", str(checkpoint), split, json.dumps(horizons["0"]), flush=True)
        del policy


def rollout(output, checkpoints, episodes, modes, seed, demo=False):
    from train_act_rl import PantheraStackEnv
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
    from lerobot.policies.utils import prepare_observation_for_inference
    env = None
    results = []
    demo_paths = sorted(DEFAULT_RENDERED.glob("episode_*"))
    held = json.loads((ROOT / "outputs/act/panthera_stack_30hz/experiment.json").read_text())["held_out_episode_indices"]
    n_val = min(len(held), episodes//2)
    chosen_demos = held[:n_val] + [i for i in range(len(demo_paths)) if i not in held][:episodes-n_val]
    try:
        for checkpoint in checkpoints:
            policy, pre, post = load_policy(checkpoint)
            if policy.config.chunk_size != 30 or json.loads((checkpoint / "deployment.json").read_text())["fps"] != 30:
                raise ValueError("This diagnostic compares 30 Hz, 30-action checkpoints")
            for mode in modes:
                policy.config.temporal_ensemble_coeff = .01 if mode == "ensemble" else None
                policy.config.n_action_steps = 1 if mode == "ensemble" else int(mode.removeprefix("queue").removeprefix("anchored"))
                if mode == "ensemble":
                    policy.temporal_ensembler = ACTTemporalEnsembler(.01, 30)
                for trial in range(episodes):
                    if env is not None:
                        env.close()
                    env = PantheraStackEnv(seed+trial, 901, 30, None, .99, 0., 0., None, 0.)
                    if demo:
                        data = np.load(demo_paths[chosen_demos[trial]] / "trajectory.npz")
                        raw = np.load(ROOT / "data" / demo_paths[chosen_demos[trial]].name / "data.npz")
                        env.sim.reset(randomize=False)
                        env.sim.data.qpos[env.sim.arm_qadr] = data["q"][0]
                        env.sim.data.qvel[env.sim.arm_dofadr] = raw["dq"][0]
                        env.sim.data.qpos[env.sim.finger_qadr] = data["finger_q"][0]
                        env.sim.data.ctrl[:] = data["ctrl"][0]
                        env.sim.sync_control_state()
                        env.sim.set_object_poses(data["obj_pos"][0], data["obj_quat"][0])
                        mujoco.mj_forward(env.sim.model, env.sim.data)
                    policy.reset()
                    row = {"checkpoint": str(checkpoint), "mode": mode, "seed": seed+trial,
                           "demo_episode": chosen_demos[trial] if demo else None,
                           "demo_split": ("val" if chosen_demos[trial] in held else "train") if demo else None,
                           "grasped": False, "lifted": False, "two_stacked": False, "success": False,
                           "held_lift_0_5s": False, "first_grasp_step": None, "first_held_lift_step": None}
                    trace = {k: [] for k in ("q", "action", "ee", "objects", "grasp_flags", "nearest_mm", "pinched", "aligned", "finger_q")}
                    stable = 0
                    held_stable = np.zeros(3, dtype=int)
                    anchor_offset = np.zeros(6)
                    # Render every tick, as deployment does. Rendering cadence can
                    # alter a few pixels even with identical physics state.
                    with torch.inference_mode():
                        for step in range(450):
                            querying = mode == "ensemble" or len(policy._action_queue) == 0
                            state, shoulder, wrist = env.observation()
                            if querying:
                                obs = pre(prepare_observation_for_inference({"observation.state": state,
                                    "observation.images.shoulder": shoulder, "observation.images.wrist": wrist},
                                    torch.device(policy.config.device), task="stack the three colored cubes", robot_type="panthera_ht_sim"))
                            action = post(policy.select_action(obs)).cpu().numpy().reshape(-1)
                            if mode.startswith("anchored"):
                                if querying:
                                    anchor_offset = state[:6] - action[:6]
                                action[:6] += anchor_offset
                            env.step(action)
                            sim = env.sim
                            pos, _ = sim.object_poses()
                            flags = sim.grasp_flags()
                            metrics = stack_metrics(pos)
                            held_stable = np.where(flags & (pos[:, 2] > .0975), held_stable+1, 0)
                            stable = stable+1 if metrics["three_stack"] and not flags.any() else 0
                            row["grasped"] |= bool(flags.any()); row["lifted"] |= bool(metrics["lifted_cubes"])
                            row["two_stacked"] |= metrics["two_stack"]; row["success"] |= stable >= 15
                            row["held_lift_0_5s"] |= bool((held_stable >= 15).any())
                            if flags.any() and row["first_grasp_step"] is None: row["first_grasp_step"] = step
                            if (held_stable >= 15).any() and row["first_held_lift_step"] is None: row["first_held_lift_step"] = step
                            pinched = sim._pinched()
                            values = {"q": sim.q.copy(), "action": action.copy(), "ee": sim.ee_pos(), "objects": pos,
                                      "grasp_flags": flags, "nearest_mm": float(np.linalg.norm(pos-sim.ee_pos(), axis=1).min()*1000),
                                      "pinched": [n in pinched for n in sim.object_names],
                                      "aligned": [sim._face_aligned(i) for i in range(3)],
                                      "finger_q": sim.data.qpos[sim.finger_qadr].copy()}
                            for k, v in values.items(): trace[k].append(v)
                    arrays = {k: np.array(v) for k, v in trace.items()}
                    row["minimum_distance_mm"] = float(arrays["nearest_mm"].min())
                    row["ever_closed"] = bool((arrays["action"][:, 6] < .01).any())
                    row["ever_pinched"] = bool(arrays["pinched"].any())
                    row["closed_near_frames"] = int(((arrays["nearest_mm"] < 40) & (arrays["action"][:, 6] < .01)).sum())
                    row["last_3s_ee_span_mm"] = float(np.linalg.norm(np.ptp(arrays["ee"][-90:], axis=0))*1000)
                    stem = checkpoint.parent.name+"_"+checkpoint.name+"_"+mode+"_"+str(seed+trial)
                    np.savez_compressed(output / (stem+".npz"), **arrays)
                    results.append(row)
                    save(output / "rollouts.json", results)
                    print("ROLLOUT", json.dumps(row), flush=True)
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("data", "offline", "rollout"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", type=Path)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--modes", nargs="+", default=["queue30"])
    parser.add_argument("--seed", type=int, default=20261201)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.command != "data" and not args.checkpoints:
        parser.error("--checkpoints is required for offline and rollout diagnostics")
    if args.episodes < 1 or (args.demo and args.episodes > 213):
        parser.error("episodes must be positive and demo runs cannot exceed 213 episodes")
    if any(mode not in {"ensemble", "queue1", "queue3", "queue10", "queue30", "anchored30"} for mode in args.modes):
        parser.error("unsupported inference mode")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20260923)
    if args.command == "data": data_audit(args.output)
    elif args.command == "offline": offline(args.output, args.checkpoints)
    else: rollout(args.output, args.checkpoints, args.episodes, args.modes, args.seed, args.demo)


if __name__ == "__main__":
    main()
