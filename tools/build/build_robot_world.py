"""Build the deployment world for a robot trained in mjlab: the same thing assets/robots/g1/mjlab_1.6.0/ holds
for the G1 (mjlab `export-scene` output + the training physics options), so env_build.py and the ROS sim
can load it.

  .venv-mjlab/bin/python tools/build/build_robot_world.py wisco
  -> assets/robots/wisco_v1/mjlab_1.6.0/{scene.xml, wisco_velocity_flat.xml, assets/*.stl}

Why this isn't just export-scene for Wisco: the G1's mjlab actuators are MuJoCo position actuators, so the
exported XML already holds the PD loop. Wisco's (DcMotorActuator, AnkleLinkageActuator) compute the PD torque in
Python every physics step, so export-scene writes bare torque motors and plain MuJoCo would let the robot
collapse. Here each one becomes a MuJoCo position actuator with the training gains (robot_numbers.yaml via
robot_params.joint_params(): kp, kd) and the per-joint effort limit as forcerange.

Not reproduced in plain MuJoCo (training-only, Python side): the X-series torque-speed line, the ankle per-motor
clamp through the rod linkage (34 N*m per X4-36), and the 0-10 ms command delay. The joint-space effort limits
still apply. hsim_sim could add the ankle model (hardware_observations.md, Ankle torque).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[2]
ROBOTS = {
  "wisco": {
    "task": "Wisco-V1-Velocity-Flat",
    "import_path": ROOT / "training" / "locomotion" / "wisco_v1",
    "import_module": "wisco_mjlab",
    "params_path": ROOT / "robots" / "wisco_v1" / "model",
    "mesh_dir": ROOT / "robots" / "wisco_v1" / "meshes",
    "out": ROOT / "assets" / "robots" / "wisco_v1" / "mjlab_1.6.0",
    "world_name": "wisco_velocity_flat",
  },
}
PREFIX = "robot/"


def physics_options(task: str) -> str:
  """<option> line from the task's training config (export-scene doesn't write it, mjlab sets it in Python)."""
  from mjlab.tasks.registry import load_env_cfg
  mj = load_env_cfg(task).sim.mujoco
  integ = {"implicitfast": "implicitfast", "euler": "Euler", "rk4": "RK4", "implicit": "implicit"}
  get = lambda k, d: getattr(mj, k, d)
  return (f'<option timestep="{get("timestep", 0.005)}" integrator="{integ.get(str(get("integrator", "implicitfast")).lower(), "implicitfast")}" '
          f'cone="{get("cone", "pyramidal")}" jacobian="{get("jacobian", "auto")}" solver="{get("solver", "newton").capitalize()}" '
          f'iterations="{get("iterations", 10)}" tolerance="{get("tolerance", 1e-8)}" ls_iterations="{get("ls_iterations", 20)}" '
          f'ls_tolerance="{get("ls_tolerance", 0.01)}" ccd_iterations="{get("ccd_iterations", 50)}" '
          f'impratio="{get("impratio", 1)}" gravity="0 0 -9.81"/>')


def main(robot: str):
  """Export the robot's training scene from mjlab with the trained gains as position actuators."""
  r = ROBOTS[robot]
  sys.path.insert(0, str(r["import_path"]))
  sys.path.insert(0, str(r["params_path"]))
  __import__(r["import_module"])
  import mjlab.tasks  # noqa: F401
  from mjlab.scripts.export_scene import ExportSceneCfg, export_scene
  import robot_params as P

  out = r["out"]
  tmp = out.parent / "_export_tmp"
  export_scene(ExportSceneCfg(target=r["task"], output_dir=str(tmp)))
  out.mkdir(parents=True, exist_ok=True)
  (out / "assets").mkdir(exist_ok=True)
  for f in r["mesh_dir"].iterdir():
    if f.suffix.lower() in (".stl", ".obj", ".png"):
      shutil.copy2(f, out / "assets" / f.name)

  spec = mujoco.MjSpec.from_file(str(tmp / "scene.xml"))
  spec.meshdir = str(out / "assets")
  params = P.joint_params()
  done = []
  for act in spec.actuators:
    joint = act.target.removeprefix(PREFIX)
    p = params[joint]
    act.set_to_position(kp=float(p["kp"]), kv=float(p["kd"]))
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-float(p["effort"]), float(p["effort"])]
    done.append(joint)
  # init_state ctrl was written for torque motors (= 0 torque); for position actuators it must be the standing pose
  m = spec.compile()
  key = spec.keys[0]
  qpos = list(key.qpos)
  ctrl = []
  for i in range(m.nu):
    jid = m.actuator_trnid[i, 0]
    ctrl.append(qpos[m.jnt_qposadr[jid]])
  key.ctrl = ctrl
  xml = spec.to_xml()
  import re
  xml = re.sub(r'meshdir="[^"]*"', 'meshdir="assets"', xml)    # relative, so it loads anywhere (e.g. /hsim in Docker)
  (out / "scene.xml").write_text(xml)
  shutil.rmtree(tmp, ignore_errors=True)

  (out / f'{r["world_name"]}.xml').write_text(
    f'<!-- {robot} velocity-flat training world from mjlab 1.6.0, for deployment on CPU.\n'
    f'     scene.xml: mjlab `export-scene {r["task"]}` with the Python-side PD actuators turned into MuJoCo position\n'
    f'     actuators (kp/kd/effort from robot_numbers.yaml), written by tools/build/build_robot_world.py.\n'
    f'     Physics options restated from the task config (export-scene does not write them). -->\n'
    f'<mujoco model="{r["world_name"]}">\n  {physics_options(r["task"])}\n  <include file="scene.xml"/>\n</mujoco>\n')
  print(f"wrote {out} ({len(done)} position actuators)")


if __name__ == "__main__":
  main(sys.argv[1] if len(sys.argv) > 1 else "wisco")
