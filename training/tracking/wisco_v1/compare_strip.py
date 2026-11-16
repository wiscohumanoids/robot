"""Side-by-side check of a Wisco retarget against Unitree's G1 retarget of the same LAFAN1 take.

Renders N moments of the clip: top row G1 (Unitree's CSV, 30 fps), bottom row Wisco (our CSV, 30 fps),
same camera. Writes one PNG.
  .venv-mjlab/bin/python training/tracking/wisco_v1/compare_strip.py --g1 <g1.csv> --wisco <wisco.csv> \
      --out strip.png [--times 0,2.5,5,7.5,9.9] [--start-frame 0]
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parents[2] / "robots" / "wisco_v1" / "model"
WISCO_XML = MODEL_DIR / "wisco_v1_train.xml"
WISCO_JOINTS = json.loads((MODEL_DIR / "train_model_info.json").read_text())["policy_joints_in_model_order"]


def g1_model():
    import mjlab
    from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as G
    return mujoco.MjSpec.from_file(str(G.G1_XML))


def scene(spec):
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.55, 0.6, 0.65, 1])
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    return spec.compile()


def render_rows(m, joints, csv, frames, size=(320, 400)):
    data = np.loadtxt(csv, delimiter=",")
    d = mujoco.MjData(m); r = mujoco.Renderer(m, size[1], size[0])
    qadr = [m.jnt_qposadr[m.joint(n).id] for n in joints]
    cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 2.6, -10, 135
    out = []
    for f in frames:
        row = data[min(f, len(data) - 1)]
        d.qpos[0:3] = row[0:3]; d.qpos[3:7] = row[[6, 3, 4, 5]]; d.qpos[qadr] = row[7:7 + len(joints)]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = [row[0], row[1], 0.6]
        R = d.xmat[m.body("pelvis").id].reshape(3, 3)
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        cam.azimuth = yaw + 180 - 35                     # same 3/4-front view relative to each robot's heading
        r.update_scene(d, cam); out.append(r.render())
    r.close()
    return np.hstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", required=True); ap.add_argument("--wisco", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--times", default="0,2,4,6,8"); ap.add_argument("--start-frame", type=int, default=0)
    a = ap.parse_args()
    t = [float(x) for x in a.times.split(",")]
    from mjlab.tasks.tracking.config.g1 import env_cfgs  # noqa: F401  (ensures mjlab importable)
    import sys
    sys.path.insert(0, str(HERE.parent))   # training/tracking
    from motion_to_npz import G1_JOINTS
    g1_frames = [a.start_frame + int(round(x * 30)) for x in t]
    w_frames = [int(round(x * 30)) for x in t]
    top = render_rows(scene(g1_model()), G1_JOINTS, a.g1, g1_frames)
    bot = render_rows(scene(mujoco.MjSpec.from_file(str(WISCO_XML))), WISCO_JOINTS, a.wisco, w_frames)
    from PIL import Image
    Image.fromarray(np.vstack([top, bot])).save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
