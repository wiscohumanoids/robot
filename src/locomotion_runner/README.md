# locomotion_runner

**Status:** Stub gait, real interface (see `ARCHITECTURE.md`'s real-vs-stub table).
**Frequency:** 50 Hz.
**Subscribes:** `/cmd_vel` (`VelocityCommand`), `/robot_state` (`RobotState`).
**Publishes:** `/locomotion/observation` (`PolicyObservation`, for logging),
`/locomotion/joint_targets` (`JointTargets`, `source="locomotion"`).

Today's gait is a hand-written sinusoidal hip/knee-pitch swing on top of a
fixed nominal pose, gated by whether `/cmd_vel` exceeds a small threshold —
**not** a trained policy. The point of this package is that everything
*around* that gait (the 50 Hz rate, the message types, the observation
assembly) is exactly what a real policy needs, so dropping one in is a
one-function change.

## Dropping in a real berkeley_humanoid policy

See `locomotion_node.py`'s `_compute_action_onnx()`, marked:

```python
# === REAL POLICY GOES HERE (load ONNX) ===
```

Launch with:

```bash
ros2 run locomotion_runner locomotion_node --ros-args -p policy_onnx_path:=/path/to/policy.onnx
```

Requires `onnxruntime` (`pip install onnxruntime`) — not a hard dependency of
this package since the stub path doesn't need it.

**Read `INTEGRATION_POINTS.md`'s "Locomotion" section before doing this** —
there are two real gaps: (1) berkeley_humanoid trains a 29-DOF observation/
action space, this repo's canonical hardware is 23-DOF, and the 6-joint
pad/truncate mapping (`CANONICAL_TO_BERKELEY29` in this file) is unverified
against a real checkpoint; (2) the raw policy output in `berkeley_humanoid`
is a *correction* on top of a hard-coded nominal pose + gait, computed inside
`G1Env.step()` — that math is not part of the ONNX graph and must be ported
here too, which `_compute_action_onnx()` does not yet do (it currently
assumes the ONNX graph's raw output is directly usable as joint targets).
