# hsim_interfaces

Custom ROS types, only where no standard type fits:

| Type | Used by |
|---|---|
| `msg/Skill`, `msg/SkillPlan` | planner → behavior tree |
| `action/ExecutePlan` | planner (client) → behavior (server): run a plan, get progress and a plain-words failure reason |
| `srv/SetBodyPose` | **test-only** `/sim/set_body_pose`: move objects in the simulated world |
