# Wisco V1 pre-training checks

Checks that the robot model can be trained before any GPU time is spent: in plain MuJoCo on a laptop, with the same
actuator model mjlab uses.

| File | What |
|---|---|
| `run_checks.py` | eight checks, each PASS / FAIL: A1 the training model loads; A2 default pose (no self-collision, soles flat, centre of mass over the feet); A3 standing 10 s with zero action; A4 per-joint step responses; A5 joint directions, mirroring and range sweeps; A6 falls are detected; A7 both feet carry the weight; A8 simulation sizing and step time. Run a subset with `run_checks.py A3 A4`. Writes `results/<actuator>_delay<N>/` (report, JSON, plots; not kept in the repo) |
| `g1_baseline.py` | the same tests on the Unitree G1 with mjlab's own gains, as reference numbers for a robot known to train well |
| `sim_replica.py` | plain-MuJoCo copy of mjlab's actuator model (DC motor torque-speed limits, command delay) used by both |

Run after every model rebuild: `../../../.venv/bin/python run_checks.py --actuator dc --delay 2` from this folder
(see `robots/wisco_v1/README.md`, "Rebuild").
