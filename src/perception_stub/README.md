# perception_stub

**Layer:** 4 (perception & state estimation). **Status:** STUB.
**Frequency:** 30 Hz.
**Publishes:** `/object_poses` (`humanoid_interfaces/ObjectPoseArray`, frame
`map`), **the only publisher.**

## What is fake

There is no camera and no detector. Five objects are reported forever at fixed
positions, confidence 1.0:

- `cube_1` at the params `cube_x`, `cube_y`, `cube_z` (defaults 1.5, 0.3, 0.8 m in `map`);
- `ball_1`, `cup_1`, `box_1`, `bottle_1` from `scene.py`, one per cell of a 10 x 10
  grid (1 m cells) over the 10 m map, on a 0.8 m table.

| Object | Cell (col, row) | x, y (m) |
|---|---|---|
| `cube_1` | (6, 5) | 1.5, 0.3 |
| `ball_1` | (3, 7) | -1.5, 2.5 |
| `cup_1` | (8, 2) | 3.5, -2.5 |
| `box_1` | (7, 8) | 2.5, 3.5 |
| `bottle_1` | (2, 2) | -2.5, -2.5 |

## What replaces it

ORB-SLAM3 + FoundationPose (or similar) publishing the same `ObjectPoseArray`.
Contract points a real implementation must keep: poses **in the `map` frame**
(do the TF transform yourself so consumers don't need to), stable `object_id`s
across frames, >= 15 Hz. The camera's mounting frame should be added to the URDF
as a static edge: coordinate with whoever owns `g1_description`.

```bash
ros2 launch bringup full_stack.launch.py perception:=external
ros2 topic echo /object_poses --once
```
