"""Turn an mjlab training run into a deployable policy folder.

Reads the ONNX file mjlab exports next to its checkpoints, pulls out the metadata
mjlab embeds (joint order, PD gains, default pose, action scale, observation layout),
checks the model runs on CPU with onnxruntime, and writes:

    models/locomotion/<policy-id>/
        policy.onnx      (gitignored)
        manifest.yaml    (the deploy contract read by hsim_locomotion)
        env.yaml, agent.yaml   (mjlab's full training config, for reference)

Usage:
    uv run python scripts/export_manifest.py <run-dir> --policy-id g1_velocity_all_joints

Manifest versions:
  1  the policy drives every joint in `joints` (all have action_scale).
  2  the policy drives a subset (e.g. legs + waist). Written when the ONNX metadata has
     `action_joint_names` (hsim_tasks.runner.SubsetVelocityRunner). `joints` still lists every
     robot joint in observation order (default_pos, kp, kd; action_scale only on driven joints,
     plus role: policy|held); `action.joints` gives the driven joints in action order; `held_joints`
     says how the other joints are commanded on the robot (see hsim_locomotion/policy_node.py).
  3  motion-tracking policy (kind: tracking): drives every joint and plays the clip its ONNX file carries. Written
     when the ONNX metadata has `anchor_body_name` (mjlab's tracking exporter); `tracking` records the anchor body,
     its index in the ONNX body outputs, and the clip length.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_VERSION = 1          # all joints driven by the policy
MANIFEST_VERSION_SUBSET = 2   # policy drives a subset; the rest are held / commanded elsewhere
MANIFEST_VERSION_TRACKING = 3  # motion tracking: all joints, clip carried by the ONNX file
HELD_SOURCE_TOPIC = "/arm/joint_targets"

# mjlab G1 defaults, used only if the run's env.yaml can't be read.
DEFAULT_SIM_DT = 0.005
DEFAULT_DECIMATION = 4


def parse_value(raw: str):
    """mjlab (exporter_utils.list_to_csv_str) joins entries with ',' and the values
    inside one entry (e.g. a [min, max] clip range) with ';': '-inf;inf,-inf;inf'."""
    def scalar(s: str):
        try:
            return float(s)
        except ValueError:
            return s

    def entry(s: str):
        s = s.strip()
        return [scalar(v) for v in s.split(";")] if ";" in s else scalar(s)

    if "," not in raw:
        return entry(raw)
    return [entry(e) for e in raw.split(",")]


class _AnyTagLoader(yaml.SafeLoader):
    """Loads mjlab's dumped config while ignoring Python object tags."""


def _construct_any(loader, _suffix, node):
    """YAML constructor that reads any tagged node as plain data."""
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_AnyTagLoader.add_multi_constructor("", _construct_any)


def find_key(obj, key):
    """First value for `key` anywhere in a nested dict/list."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_key(v, key)
            if found is not None:
                return found
    return None


def motion_steps(model) -> int | None:
    """Clip length of a tracking ONNX: the first dimension of its bundled `joint_pos` buffer."""
    for init in model.graph.initializer:
        if init.name.split('.')[0] == 'joint_pos' and len(init.dims) == 2:   # exported as 'joint_pos.1'
            return int(init.dims[0])
    return None


def run_on_cpu(onnx_path: Path) -> dict:
    """Feed zeros through the model on CPU; return input/output shapes and timing."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds, inputs = {}, []
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        dtype = np.int64 if "int" in i.type else np.float32
        feeds[i.name] = np.zeros(shape, dtype=dtype)
        inputs.append({"name": i.name, "shape": shape, "type": i.type})
    outs = sess.run(None, feeds)
    outputs = [
        {"name": o.name, "shape": list(v.shape)} for o, v in zip(sess.get_outputs(), outs)
    ]
    for v in outs:
        if not np.all(np.isfinite(v)):
            raise RuntimeError("model produced NaN/Inf on zero input")

    import time
    t0 = time.perf_counter()
    for _ in range(200):
        sess.run(None, feeds)
    ms = (time.perf_counter() - t0) / 200 * 1e3
    return {"inputs": inputs, "outputs": outputs, "cpu_inference_ms": round(ms, 3)}


def main() -> int:
    """Write models/locomotion/<id>/ from a run: ONNX, manifest from its metadata, configs; check it on CPU."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="mjlab run dir, e.g. logs/rsl_rl/g1_velocity/<timestamp>")
    ap.add_argument("--policy-id", required=True, help="folder name under models/locomotion/")
    ap.add_argument("--onnx", type=Path, help="specific ONNX file (default: newest in run dir)")
    ap.add_argument("--out-dir", type=Path, help="output folder (default: models/locomotion/<policy-id>)")
    ap.add_argument("--gains-from-model", type=Path, help="take kp/kd per joint from this MuJoCo world's position "
                    "actuators instead of the ONNX metadata (Wisco: its PD runs in Python in training, so mjlab exports "
                    "kp=1, kd=0; tools/build_robot_world.py writes the real gains into the deploy world)")
    ap.add_argument("--action-joints", help="comma-separated driven joints in action order, for ONNX files exported "
                    "without `action_joint_names` metadata (e.g. Wisco: its training config's LOCOMOTION_ACTION_JOINTS)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    onnx_path = args.onnx or max(run_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, default=None)
    if onnx_path is None:
        print(f"No .onnx in {run_dir}. mjlab exports one on every checkpoint save.", file=sys.stderr)
        return 1

    model = onnx.load(str(onnx_path))
    meta = {p.key: parse_value(p.value) for p in model.metadata_props}
    required = ["joint_names", "joint_stiffness", "joint_damping", "default_joint_pos", "action_scale", "observation_names"]
    missing = [k for k in required if k not in meta]
    if missing:
        print(f"ONNX metadata is missing {missing}. Keys present: {sorted(meta)}", file=sys.stderr)
        return 1

    joint_names = meta["joint_names"]
    joint_names = [joint_names] if isinstance(joint_names, str) else joint_names
    n = len(joint_names)
    action_joints = args.action_joints.split(",") if args.action_joints else meta.get("action_joint_names")
    if isinstance(action_joints, str):
        action_joints = [action_joints]
    subset = action_joints is not None and list(action_joints) != list(joint_names)
    if subset:
        unknown = [j for j in action_joints if j not in joint_names]
        if unknown:
            print(f"action_joint_names not in joint_names: {unknown}", file=sys.stderr)
            return 1
    n_act = len(action_joints) if subset else n
    tracking = 'anchor_body_name' in meta
    if tracking and subset:
        print('a tracking policy must drive every joint', file=sys.stderr)
        return 1
    body_names = meta.get('body_names')
    body_names = [body_names] if isinstance(body_names, str) else body_names

    def per_joint(key, count=n):
        v = meta[key]
        v = [v] * count if isinstance(v, float) else v
        if len(v) != count:
            raise ValueError(f"{key} has {len(v)} values, expected {count}")
        return v

    scale_of = dict(zip(action_joints, per_joint("action_scale", n_act))) if subset else None

    if args.gains_from_model:
        import mujoco
        mm = mujoco.MjModel.from_xml_path(str(args.gains_from_model))
        kp_of, kd_of = {}, {}
        for i in range(mm.nu):
            jn = mm.joint(mm.actuator_trnid[i, 0]).name.split("/")[-1]
            kp_of[jn], kd_of[jn] = float(mm.actuator_gainprm[i, 0]), float(-mm.actuator_biasprm[i, 2])
        missing_g = [j for j in joint_names if j not in kp_of]
        if missing_g:
            print(f"--gains-from-model: no actuator for {missing_g}", file=sys.stderr)
            return 1
        meta["joint_stiffness"] = [round(kp_of[j], 3) for j in joint_names]
        meta["joint_damping"] = [round(kd_of[j], 3) for j in joint_names]

    params = run_dir / "params"
    env_cfg = None
    if (params / "env.yaml").exists():
        env_cfg = yaml.load((params / "env.yaml").read_text(), Loader=_AnyTagLoader)
    sim_dt = find_key(env_cfg, "timestep") if env_cfg else None
    decimation = find_key(env_cfg, "decimation") if env_cfg else None
    timing_source = "env.yaml" if (sim_dt and decimation) else "assumed mjlab G1 defaults"
    sim_dt = float(sim_dt or DEFAULT_SIM_DT)
    decimation = int(decimation or DEFAULT_DECIMATION)

    check = run_on_cpu(onnx_path)
    action_dims = [o["shape"][-1] for o in check["outputs"] if o["shape"]]
    if n_act not in action_dims:
        print(f"WARNING: no output has {n_act} values (one per driven joint). Outputs: {check['outputs']}", file=sys.stderr)

    out_dir = args.out_dir.resolve() if args.out_dir else REPO_ROOT / "models" / "locomotion" / args.policy_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, out_dir / "policy.onnx")
    for name in ("env.yaml", "agent.yaml"):
        if (params / name).exists():
            shutil.copy2(params / name, out_dir / name)

    manifest = {
        "manifest_version": (MANIFEST_VERSION_TRACKING if tracking else
                             MANIFEST_VERSION_SUBSET if subset else MANIFEST_VERSION),
        "policy_id": args.policy_id,
        "kind": "tracking" if tracking else "velocity",
        **({"tracking": {
            "anchor_body": meta["anchor_body_name"],
            "anchor_index": body_names.index(meta["anchor_body_name"]),
            "body_names": body_names,
            "motion_steps": motion_steps(model),
        }} if tracking else {}),
        # Checksum of policy.onnx, so a manifest and a policy file can be checked to belong together.
        "source": {"onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest()},
        "timing": {
            "sim_dt": sim_dt,
            "decimation": decimation,
            "policy_hz": round(1.0 / (sim_dt * decimation), 3),
            "source": timing_source,
        },
        "joints": [
            {
                "name": name,
                "default_pos": per_joint("default_joint_pos")[i],
                "kp": per_joint("joint_stiffness")[i],
                "kd": per_joint("joint_damping")[i],
                **({"action_scale": scale_of[name], "role": "policy"} if subset and name in scale_of
                   else {"role": "held"} if subset
                   else {"action_scale": per_joint("action_scale")[i]}),
            }
            for i, name in enumerate(joint_names)
        ],
        "action": {
            "type": "joint_position",
            "formula": "target = default_pos + action_scale * action",
            **({"joints": list(action_joints)} if subset else {}),
        },
        **({"held_joints": {
            "names": [j for j in joint_names if j not in scale_of],
            "source_topic": HELD_SOURCE_TOPIC,
            "behaviour": "default_pos until the first message on source_topic, then the latest commanded "
                         "target per joint (no timeout). In training these joints followed scripted random "
                         "targets (stand-in for the ROS arm controller).",
        }} if subset else {}),
        "observation": {
            "terms": [t] if isinstance(t := meta["observation_names"], str) else t,
            "scale": meta.get("observation_terms_scale"),
            "clip": meta.get("observation_terms_clip"),
            "history_length": meta.get("observation_terms_history_length"),
            "flatten_history_dim": meta.get("observation_terms_flatten_history_dim"),
        },
        "commands": [c] if isinstance(c := meta.get("command_names"), str) else c,
        "io": check,
        "notes": [
            "Floats in ONNX metadata are rounded to 3 decimals by mjlab; hsim_locomotion "
            "compares kp/kd here against the MuJoCo model and refuses to start if they differ by more than that.",
        ],
        "raw_metadata_keys": sorted(meta),
    }
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, width=120))

    print(f"Wrote {out_dir}/manifest.yaml")
    print(f"  {n} joints ({n_act} driven by the policy), policy {manifest['timing']['policy_hz']} Hz, "
          f"CPU inference {check['cpu_inference_ms']} ms, obs terms: {meta['observation_names']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
