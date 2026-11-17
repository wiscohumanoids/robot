"""Raw episode writer/reader (the format between data generation and the GR00T converter).

One episode_XXXXXX.npz + episode_XXXXXX.json per episode, T steps at 20 Hz:
  ego_rgb (T,240,320,3) uint8, [wrist_left_rgb, wrist_right_rgb (T,120,160,3)]
  q_left_arm (T,7), q_right_arm (T,7), q_waist (T,3) (welded: zeros), grip_left (T,), grip_right (T,)  [0 closed, 1 open]
  act_left_arm (T,7), act_right_arm (T,7) (joint position targets), act_grip_left (T,), act_grip_right (T,)
  obj_pose (T,7) (x y z qw qx qy qz, world)
  extras (beyond the agreed keys, safe to ignore): target_pose (T,7), tcp_active (T,7), segment (T,) int8
json: instruction, object_category, object_name, success, fps, camera_intrinsics, seed, source, + extras.
Arm joint order: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw.
"""
import json
from pathlib import Path

import numpy as np

from ik import mat_to_quat


class Recorder:
    """Collects one episode step by step."""
    def __init__(self):
        """Empty buffers for every field."""
        self.buf = {k: [] for k in ("ego_rgb", "wrist_left_rgb", "wrist_right_rgb", "q_left_arm", "q_right_arm", "q_waist",
                                    "grip_left", "grip_right", "act_left_arm", "act_right_arm", "act_grip_left",
                                    "act_grip_right", "obj_pose", "target_pose", "tcp_active", "segment")}

    def add(self, obs, ql, qr, gl, gr, tcp, seg):
        """Append one step."""
        b = self.buf
        for k in ("ego_rgb", "wrist_left_rgb", "wrist_right_rgb"):
            if k in obs:
                b[k].append(obs[k])
        b["q_left_arm"].append(obs["q_left_arm"]); b["q_right_arm"].append(obs["q_right_arm"])
        b["q_waist"].append(obs["q_waist"])
        b["grip_left"].append(obs["grip_left"]); b["grip_right"].append(obs["grip_right"])
        b["act_left_arm"].append(np.asarray(ql)); b["act_right_arm"].append(np.asarray(qr))
        b["act_grip_left"].append(float(gl)); b["act_grip_right"].append(float(gr))
        b["obj_pose"].append(obs["obj_pose"]); b["target_pose"].append(obs["target_pose"])
        b["tcp_active"].append(np.concatenate([tcp[0], mat_to_quat(tcp[1])]))
        b["segment"].append(seg)

    def __len__(self):
        return len(self.buf["segment"])

    def save(self, out_dir, index, meta):
        """Write episode_<index>.npz and .json."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        arrs = {}
        for k, v in self.buf.items():
            if not v:
                continue
            a = np.asarray(v)
            if k.endswith("rgb"):
                a = a.astype(np.uint8)
            elif k == "segment":
                a = a.astype(np.int8)
            else:
                a = a.astype(np.float32)
            arrs[k] = a
        stem = out_dir / f"episode_{index:06d}"
        np.savez_compressed(str(stem) + ".npz", **arrs)
        meta = {**meta, "num_steps": len(self), "keys": sorted(arrs)}
        Path(str(stem) + ".json").write_text(json.dumps(meta, indent=1, default=float))
        return stem
