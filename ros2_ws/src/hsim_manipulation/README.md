# hsim_manipulation (Phase 5)

Manipulation skill node (ROS name `manipulation`): action server **/skills/manipulate**
(`hsim_interfaces/action/Manipulate`, proposed in `docs/runs/patches/F2_hsim_interfaces_manipulate.patch`).

For a goal `{instruction, object?, target?, max_duration_s?}` it runs the VLA policy (services/vla_server, GR00T) at
20 Hz: head-camera image + arm/gripper joint state + instruction -> 16-step chunk of absolute joint targets; publishes
the first 8 on `/arm/joint_targets`, asks again, until the perception check (object resting over the target) passes,
the time limit, or cancel. Legs and waist stay with hsim_locomotion.

```
ros2 launch hsim_manipulation manipulation.launch.py                      # needs the VLA server tunnel on the Mac
ros2 action send_goal /skills/manipulate hsim_interfaces/action/Manipulate "{instruction: 'pick up the mug and put it on the plate', object: mug, target: plate}"
```
Behavior tree: `pick(X)` + `place(Y)` in a plan become one goal (patch `docs/runs/patches/F2_behavior_manipulate.patch`).
Test (throwaway container, builds under /tmp): `docker run --rm -v $PWD:/repo:ro hsim/ros-jazzy:dev bash /repo/ros2_ws/src/hsim_manipulation/test/smoke_test.sh`

Needs before it works in the full sim (not this package): the G1 in hsim_sim must carry the Dex1-1 grippers and an ego
camera matching training (F1: D435 mount on the torso, pitched 47.6 deg down, 320x240), and the arm joints must be
released by the locomotion policy (legs+waist v2) so `/arm/joint_targets` is their only writer.
