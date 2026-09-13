# manipulation_runner

**Status:** Stub task execution, real action-server interface.
**Frequency:** 10 Hz internal loop while a goal is active; otherwise idle.
**Action server:** `execute_manipulation` (`humanoid_interfaces/ExecuteManipulation`).
**Publishes:** `/manipulation/joint_targets` (`JointTargets`, `source="manipulation"`)
— only while a goal is executing.

Today, any goal is accepted and runs a fixed three-phase scripted arm motion
(approach -> grasp -> retract) over ~3 seconds, then reports success. No
perception, no grasp planning, no learned policy — see the
`_execute_callback` docstring in `manipulation_node.py` for the exact
replacement point.

Test it without any other node running:

```bash
ros2 action send_goal /execute_manipulation humanoid_interfaces/action/ExecuteManipulation \
  "{task: 'pick', object_id: 'cube_1'}" --feedback
```

## Arbitration note

This node only publishes for the arm joint indices (13-22) — `wbc_stub`
treats `manipulation_runner`'s output as authoritative for arms **only when
recent**, and falls back to `locomotion_runner`'s arm targets (nominal hold
pose) otherwise. See `wbc_stub/README.md`.

## Dropping in real lerobot_alohamini inference

Read `INTEGRATION_POINTS.md`'s "Manipulation" section first — the transport
(ZMQ vs. ROS2), joint-count/order mismatch (AlohaMini is not the G1's arms),
and missing IK are all real gaps that must be resolved, not just a matter of
swapping a function body.
