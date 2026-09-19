# perception_stub

**Layer:** 4 -- perception & state estimation. **Status:** STUB.
**Frequency:** 30 Hz.
**Publishes:** `/object_poses` (`humanoid_interfaces/ObjectPoseArray`, frame
`map`) -- **the only publisher.**

## What is fake

There is no camera and no detector. One cube, `cube_1`, is reported forever at
a fixed position (params `cube_x`, `cube_y`, `cube_z`; defaults 1.5, 0.3, 0.8 m in
`map`), confidence 1.0.

## What replaces it

ORB-SLAM3 + FoundationPose (or similar) publishing the same `ObjectPoseArray`.
Contract points a real implementation must keep: poses **in the `map` frame**
(do the TF transform yourself so consumers don't need to), stable `object_id`s
across frames, >= 15 Hz. The camera's mounting frame should be added to the URDF
as a static edge -- coordinate with whoever owns `g1_description`.

```bash
ros2 launch bringup full_stack.launch.py perception:=external
ros2 topic echo /object_poses --once
```
