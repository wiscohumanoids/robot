# hsim_manipulation

Manipulation skill node (ROS name `manipulation`): action server **/skills/manipulate**
(`hsim_interfaces/action/Manipulate`).

For a goal `{instruction, object?, target?, max_duration_s?}` it runs the VLA policy (services/vla_server, GR00T) at
20 Hz: head-camera image + arm/gripper joint state + instruction -> 16-step chunk of absolute joint targets; publishes
the first 8 on `/arm/joint_targets`, asks again, until the perception check (object resting over the target) passes,
the time limit, or cancel. Legs and waist stay with hsim_locomotion.

```
ros2 launch hsim_manipulation manipulation.launch.py                      # needs the VLA server tunnel on the Mac
ros2 action send_goal /skills/manipulate hsim_interfaces/action/Manipulate "{instruction: 'pick up the mug and put it on the plate', object: mug, target: plate}"
```
Behavior tree: `pick(X)` + `place(Y)` in a plan become one goal (`hsim_behavior`). The success check reads the object
memory, `/world/objects` (map frame).
Tests: `test/test_vla_client.py` (pytest, no ROS); `services/vla_server/run_vla_server.sh --mock` serves hold-still
chunks, so the node can be exercised without a model.

Needs before it works in the full sim (not this package): the G1 in hsim_sim must carry the Dex1-1 grippers and an ego
camera matching training (a D435-style mount on the torso, pitched 47.6° down, 320×240), and the arm joints must be
released by the locomotion policy (legs+waist, arms-free) so `/arm/joint_targets` is their only writer.
