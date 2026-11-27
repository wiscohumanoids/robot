# hsim_teleop — Mac viewer bridges: nodes `teleop` and `viewer_bridge`

Both start with `sim_stack.launch.py` (and so `autonomy.launch.py`) when `viewer:=true` (default).

**`teleop`**: the viewer's arrow keys → `/cmd_vel_teleop` (twist_mux prefers it over Nav2), and tasks
typed in the viewer's terminal → `/planner/command` (same as `tools/ask.sh`). Close the viewer and the
robot stops (twist_mux timeout). **R** in the viewer calls `/sim/reset`.
Docker Desktop doesn't reliably forward UDP from the Mac into the container, so the node says `HELO`
to the viewer (host.docker.internal:9873) 5 times a second and the viewer replies to that address;
UDP 9871 still works directly where forwarding does.

**`viewer_bridge`**: sends the viewer's perception panel what the robot knows (display only, nothing
from the simulator): its pose estimate, `/map`, `/scan`, stereo points, Nav2's `/plan`, object memory,
planner status. UDP to host.docker.internal:9872, map frame. See the module docstring for the format.

## The viewer (tools/sim/viewer.sh)
- **Main view:** a high camera follows the robot and slowly swings round behind it; walls or tall
  furniture between the camera and the robot are hidden (cutaway). Orbit/zoom with the mouse any time
  (it keeps your angle); **F** toggles swinging with the robot's heading.
- **Robot's view panel:** the same angle, on black: the robot glowing at its *own* estimated pose, its
  map (SLAM fills in live; new walls flash), lidar with a fading trail and a radar pulse, stereo point
  cloud colored by height, labeled object boxes, Nav2's path. **V** cycles: corner / side by side /
  robot's view only / off.
- **Tasks:** type in the viewer's terminal, e.g. `walk to the fridge`; progress and the robot's replies
  show at the top left of the window.

**Fallback, terminal keyboard** (launch with `viewer:=false` so only one `teleop` node exists):

    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args \
      -r __node:=teleop -r cmd_vel:=/cmd_vel_teleop -p stamped:=true
