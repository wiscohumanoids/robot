UPDATED 2026-11-10: robot.urdf, robot.xml and model_report.json are now generated from the CAD export
(kept unchanged in original_from_team/) plus MyActuator datasheet values, the CAD IMU orientation, a ZED X optical
frame, transmissions and a "home" keyframe. See CHANGES.md. The original version of this README is
original_from_team/README.txt; the text below still describes the CAD export.

WISCOHUMANOIDS V2 MODEL  (built 2026-11-09 from "Legs V2 Full Assembly.step" + "FullAssmb.step")

FILES
  robot.urdf          31 revolute joints, 32 links + imu_link, head_camera, left_sole, right_sole
  robot.xml           MuJoCo scene: floating base, floor, torque actuators, IMU sensors,
                      keyframes "zero" (everything 0) and "stand" (arms slightly out)
  meshes/             <link>.stl visual, <link>_collision.stl convex hull (feet use a box)
  linkage.json/.py    ankle and waist rod geometry + joint <-> motor torque/velocity conversion
  model_report.json   per-joint limits, per-link mass, the pose each joint had in the CAD
  stand_test.py       5 s standing check

CONVENTIONS
  x forward, y left, z up, metres. Base link "pelvis", origin midway between the hip pitch axes.
  Zero pose = standing straight, arms down, feet flat. Pelvis origin is 0.7015 m above the floor.
  Every link frame sits at its joint pivot with axes parallel to the base at zero pose (all rpy = 0).
  Joint names, signs and ranges match the previous URDF.

WHAT IS MEASURED FROM CAD
  Joint axes and pivots (from the motor and bearing cylinders), link geometry, part volumes,
  centre-of-mass locations, inertia shapes, ankle/waist rod lengths and crank radii.
  Legs come from Legs V2 (not the older copy inside FullAssmb). Duplicate motors/bearings and the
  "wave" hands were left out; the YUBI gripper is the hand.

WHAT IS ESTIMATED  (change these when real numbers exist)
  Mass: motors use datasheet mass (X8-150 1.33, X6-60 0.82, X4-36 0.36, X4-10 0.33 kg).
        Everything else is volume x density: aluminium 2.70, fasteners/bearings steel 7.85,
        covers/fairings/hardstops plastic 1.25, sole 1.2, boards 1.9, gripper 1.4 g/cm3.
        Total 40.6 kg. No wiring. A motor's whole mass sits on the link that holds its body.
  Ankle rods and waist links are lumped into the shin and torso.
  MuJoCo-only joint values: armature (X8 0.05, X6 0.025, X4-36 0.039, X4-10 0.004 kg m2),
        damping, and friction loss (carried over from the old URDF).
  IMU and camera frames are at the CAD part locations with base-aligned axes.

ANKLES AND WAIST
  Modelled as two plain joints at the real pivots. Ankle: pitch then roll, roll axis 17.5 mm
  below the pitch axis, pitch axis 61.5 mm above the floor. Waist: yaw, then roll and pitch
  through one universal joint 174 mm above the pelvis origin.
  Joint effort/velocity limits for these four joints are the two-motor equivalents at zero pose.
  Use linkage.py to turn joint torque into the torque of each individual motor.
