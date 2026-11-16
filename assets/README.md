# assets

| Folder | What |
|---|---|
| `robots/wisco_v1/mjlab_1.6.0/` | Wisco V1 sim world: mjlab's exported scene with the training gains as MuJoCo position actuators (`tools/build_robot_world.py wisco`; source model in `robots/wisco_v1/`) |
| `robots/g1/mjlab_1.6.0/` | Unitree G1 training world from mjlab 1.6.0 |
| `robots/g1_dex1/` | G1 with Dex1-1 grippers, for manipulation (`training/manipulation/sim/build_g1_dex1.py`) |
| [`envs/`](envs/README.md) | environments (test rooms, RoboCasa kitchens): `env.yaml` sources here, built `world.mjb` + map + objects in `build/` (G1) or `build_wisco/` (Wisco V1), gitignored |
