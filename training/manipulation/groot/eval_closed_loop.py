"""Closed-loop success rate of a GR00T policy in the manipulation env (sim/env.py), through the VLA server (same path
the robot uses). Run on the VM with a python that has mujoco + numpy + Pillow (the datagen env).

  services/vla_server/run_vla_server.sh <checkpoint>                     # start the server first
  MUJOCO_GL=egl python eval_closed_loop.py --episodes 50 --mjb ../sim/build/kitchen_l1_s1_v1.mjb \
      --out ~/hsim_runs/groot_eval_v1 [--video 5]

Each episode: env.reset(seed) with held-out seeds (>= 1,000,000; datagen uses lower seeds), then repeat:
send ego image + state + instruction, execute the first `execution_horizon` actions of the returned chunk
at 20 Hz, until env.success() or --max-steps. Writes results.json (per-episode success, steps, object,
latency) and optional mp4s of the first N episodes.
"""
import argparse
import base64
import io
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parents[1] / "sim"
sys.path.insert(0, str(SIM_DIR))


def jpeg_b64(img):
    """Image as base64 JPEG."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def post(url, payload):
    """POST JSON and return the reply."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def write_mp4(frames, path, fps=20):
    """Write frames to an MP4 with ffmpeg."""
    T, H, W, _ = np.asarray(frames).shape
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                    "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                   input=np.ascontiguousarray(frames).tobytes(), check=True)


def main():
    """Closed-loop success rate over held-out seeds, through the VLA server."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8200")
    ap.add_argument("--mjb", default=str(SIM_DIR / "build" / "kitchen_l1_s1_v1.mjb"))
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed0", type=int, default=1_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", type=int, default=0, help="save mp4 of the first N episodes")
    a = ap.parse_args()
    from env import ManipEnv                                   # training/manipulation/sim/env.py

    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    health = json.loads(urllib.request.urlopen(a.url + "/health", timeout=10).read())
    exec_h = int(health["execution_horizon"])
    env = ManipEnv(a.mjb)
    results = []
    seed = a.seed0 - 1
    for ep in range(a.episodes):
        # same reset filter as datagen (reset_ok), and skip layouts that already count as success before any action
        while True:
            seed += 1
            obs = env.reset(seed)
            if getattr(env, "reset_ok", True) and not env.success():
                break
        frames, lat, steps, ok = [], [], 0, False
        while steps < a.max_steps and not ok:
            req = {"instruction": env.instruction, "images": {"ego_view": jpeg_b64(obs["ego_rgb"])},
                   "state": {"left_arm": obs["q_left_arm"].tolist(), "right_arm": obs["q_right_arm"].tolist(),
                             "left_gripper": [obs["grip_left"]], "right_gripper": [obs["grip_right"]],
                             "waist": obs["q_waist"].tolist()}}
            t0 = time.time()
            r = post(a.url + "/act", req)
            if "error" in r:
                raise SystemExit(f"server error: {r['error']}")
            lat.append(1000 * (time.time() - t0))
            A = {k: np.asarray(v) for k, v in r["actions"].items()}
            for i in range(min(exec_h, len(A["left_arm"]))):
                obs = env.step(A["left_arm"][i], A["right_arm"][i], float(A["left_gripper"][i][0]), float(A["right_gripper"][i][0]))
                steps += 1
                if ep < a.video:
                    frames.append(obs["ego_rgb"])
                if env.success():
                    ok = True
                    break
        results.append({"episode": ep, "seed": seed, "success": ok, "steps": steps, "object": env.obj,
                        "target": env.target, "instruction": env.instruction, "mean_latency_ms": float(np.mean(lat))})
        print(f"ep {ep}: {'SUCCESS' if ok else 'fail'} in {steps} steps ({env.obj} -> {env.target}), "
              f"server {np.mean(lat):.0f} ms", flush=True)
        if frames:
            write_mp4(frames, out / f"episode_{ep:03d}_{'ok' if ok else 'fail'}.mp4")
    rate = float(np.mean([r["success"] for r in results]))
    by_obj = {}
    for r in results:
        by_obj.setdefault(r["object"], []).append(r["success"])
    summary = {"success_rate": rate, "episodes": len(results), "by_object": {k: float(np.mean(v)) for k, v in by_obj.items()},
               "server": health, "max_steps": a.max_steps}
    (out / "results.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
    print(f"success rate {rate:.2%} over {len(results)} episodes -> {out / 'results.json'}")


if __name__ == "__main__":
    main()
