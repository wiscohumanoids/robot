# wbc_stub

**Status:** Stub arbitration, standing in for a real whole-body controller.
**Frequency:** 500 Hz.
**Subscribes:** `/locomotion/joint_targets`, `/manipulation/joint_targets`
(both `JointTargets`).
**Publishes:** `/joint_command` (`JointCommand`, `effort_feedforward` always
zero).

## Arbitration rule (this is a stub, not physics)

- Leg + waist joints (canonical indices 0-12): **always** taken from
  `locomotion_runner`.
- Arm joints (canonical indices 13-22): taken from `manipulation_runner`
  **only if** a message was received within the last 0.3s
  (`MANIPULATION_STALENESS_S`); otherwise falls back to
  `locomotion_runner`'s arm values (its nominal hold pose).

This is a hand-picked priority rule, not a torque-consistent whole-body
solve. It does not check whether the resulting merged pose is
dynamically feasible, does not reason about contacts, and does not resolve
conflicts other than "which source wins per joint slice." A real WBC
(operational-space control, a QP over both target sets plus a dynamics
model and contact constraints) would replace this entire node without
changing its interface — that's the point of giving it a real
`JointTargets in -> JointCommand out` contract now.

## Why 500 Hz

Faster than either input (50 Hz locomotion, 10 Hz manipulation) so this node
never becomes the bottleneck; slower than `lowlevel_control`'s 1000 Hz
in-process real-time loop by design — `JointCommand` crosses a normal
cross-process DDS topic here, and `RESEARCH_NOTES.md`'s multi-rate section is
the reason that boundary sits at 500->1000 Hz across a topic rather than at
the 1kHz hardware-interface boundary itself (which is deliberately kept
in-process inside `controller_manager`).
