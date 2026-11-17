# Manipulation data generation: G1 + Dex1-1 at a RoboCasa counter

Standing, fixed-pelvis tabletop pick-and-place (v1: cups onto plates), scripted expert demos multiplied by
MimicGen-style generation. Output: raw episodes for ../groot/ (conversion to LeRobot v2 / GR00T).

```
assets/robots/g1_dex1/                  official Unitree G1 + Dex1-1 (unitree_ros, BSD-3) + built MJCF g1_dex1.xml
training/manipulation/sim/
  build_g1_dex1.py   URDF -> MJCF: fixed pelvis, legs/waist welded, arm + gripper actuators, coupled fingers, cameras, TCP sites
  catalog.yaml       scenes (robot pose, counter, reach band) and object sets per version (v1 cups/plates; v2, v3 planned)
  scene.py           compile kitchen + robot + all version objects -> build/<scene>_<version>.mjb (+ .json index)  [Mac only]
  env.py             ManipEnv: reset(seed) randomizes arm side, object, plate, poses, robot jitter, lighting; 20 Hz step
  ik.py              damped-least-squares IK, random restarts, collision-aware solve
  expert.py          scripted pick-place planner (orientation search, collision-checked joint trajectory) + player
  episode_io.py      raw episode writer (agreed format)
training/manipulation/datagen/
  mimicgen.py        source-demo segments -> new scenes (object/target-relative transform), IK + collision checks
  generate.py        parallel workers: scripted sources, then MimicGen; successes only; progress.jsonl, stats.json
  generate.sh        VM entry point: generate.sh <version> <n_episodes> [workers=3]
  setup_env.sh       VM one-time: ~/hsim_envs/datagen (Python 3.12, mujoco 3.11.0)
  preview.py         contact sheet + GIF of the ego camera
```

## Build (Mac) and run (VM)
```
# Mac: robot model + scene (needs the RoboCasa asset checkout in ~/hsim_third_party)
cd training/manipulation/sim
../../../.venv/bin/python build_g1_dex1.py
../../../.venv/bin/python scene.py --scene kitchen_l1_s1 --version v1      # -> build/kitchen_l1_s1_v1.mjb (1.05 GB)
# copy to the VM
rsync -a build/kitchen_l1_s1_v1.mjb build/kitchen_l1_s1_v1.json hsim-gpu:~/hsim_data/manip/scenes/
rsync -a --exclude build --exclude __pycache__ ../ hsim-gpu:~/robot/training/manipulation/
# VM
~/robot/training/manipulation/datagen/setup_env.sh                 # once
~/robot/training/manipulation/datagen/generate.sh v1 1000 3        # -> ~/hsim_data/manip/v1/raw/
```

## Design notes
- Robot: Unitree G1 29-dof + Dex1-1 (official model). Pelvis fixed at standing height (0.792 m, straight legs; per-episode
  jitter +/-3 cm xy, 0..-2 cm z, +/-5 deg yaw: the walking policy won't stop at an exact spot). Arms: 7 joints each,
  position-controlled (stiff, gravity compensated) at 20 Hz. Gripper: 0 = closed, 1 = open (94.8 mm).
- Camera: ego = the G1's real head D435 mount (torso, 47.6 deg down), 320x240, fovy 58 deg. Optional wrist cameras.
- Reach: the 0.92 m counter is high for a G1 (shoulder 1.08 m). Each arm works outboard of its own shoulder
  (robot frame x 0.27-0.46 m, |y| 0.12-0.30 m); object and plate are sampled in the active arm's band.
- Expert: side grasp from above (fingers pitched 0-30 deg down, approach yaw searched), TCP at 72 % of the cup height,
  vertical descend/lift/place/retreat, free-space moves in joint space; every planned step collision-checked.
- MimicGen: pick segment relative to the object, place segment relative to the plate (translation only for round
  objects), random source of the same arm, small position noise, successes only.
