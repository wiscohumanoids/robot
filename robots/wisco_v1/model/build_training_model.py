"""Build the mjlab training model from the V1 robot (robots/wisco_v1/robot.xml is never modified).

What changes, and why:
  - remove floor, light, motors, keyframes, sensors: mjlab adds its own terrain and its own
    position actuators (gains from robot_params.py), like the G1 model file it ships
  - head_yaw / head_pitch removed (head welded to the torso): the policy drives 29 joints
  - ankle pitch lower limit -> training value from ankle_linkage.json (hardware_observations.md, Ankle reach)
  - joint friction, viscous damping, armature -> from robot_numbers.yaml (robot_params.joint_params)
  - link masses -> robot_numbers.yaml masses.overrides where given (inertia scaled with the mass)
  - collision geoms renamed <body>_collision, feet left/right_foot_collision (mjlab matches
    collision settings by name, as in the G1 config)
  - add sites left_foot / right_foot at the sole centre (foot height / slip rewards)
  - add the sensors the velocity task reads: imu_ang_vel, imu_lin_vel, imu_lin_acc,
    imu_upvector, root_angmom (same names as mjlab's G1 model)
Also writes train_model_info.json: policy joint order, default pose, pelvis height.

Run:  python build_training_model.py
"""
import json
import sys

import mujoco
import numpy as np

import robot_params as P

sys.dont_write_bytecode = True
INFO_OUT = P.HERE / "train_model_info.json"


def build():
    """robot.xml stripped to what mjlab needs: no actuators, sensors, keys, floor or lights; training limits and
    sites.
    """
    spec = mujoco.MjSpec.from_file(str(P.SOURCE_XML))
    spec.modelname = "wisco_v1_train"

    for a in list(spec.actuators):
        spec.delete(a)
    for s in list(spec.sensors):
        spec.delete(s)
    for k in list(spec.keys):
        spec.delete(k)
    for g in list(spec.worldbody.geoms):           # the floor
        spec.delete(g)
    for li in list(spec.worldbody.lights):
        spec.delete(li)

    for name in P.FIXED_JOINTS:
        spec.delete(spec.joint(name))

    lower = P.ankle_pitch_lower()
    for side in ("left", "right"):
        j = spec.joint(f"{side}_ankle_pitch")
        j.range = [lower, j.range[1]]

    # joint friction / viscous damping / armature from robot_numbers.yaml (via robot_params)
    for name, p in P.joint_params().items():
        j = spec.joint(name)
        j.frictionloss = p["frictionloss"]
        j.damping = [p["damping"], 0.0, 0.0]          # MuJoCo 3.11: [linear, quadratic, cubic]
        j.armature = p["armature"]

    # measured / better masses (robot_numbers.yaml masses.overrides): mass replaced, inertia scaled by the same factor
    for link, o in P.load_mass_overrides().items():
        b = spec.body(link)
        f = o["mass_kg"] / b.mass
        b.mass = o["mass_kg"]
        b.inertia = [x * f for x in b.inertia]
        if "com" in o:
            b.ipos = o["com"]

    # collision geom names
    renamed = {}
    for g in spec.geoms:
        if g.classname.name != "collision":
            continue
        body = g.parent.name
        new = {"left_ankle_roll_link": "left_foot_collision",
               "right_ankle_roll_link": "right_foot_collision"}.get(body, f"{body}_collision")
        if new in renamed.values():
            raise RuntimeError(f"two collision geoms on {body}")
        renamed[g.name or "(unnamed)"] = new
        g.name = new

    # foot sites at the sole centre (same place as the source model's left_sole / right_sole)
    for side in ("left", "right"):
        sole = spec.site(f"{side}_sole")
        spec.body(f"{side}_ankle_roll_link").add_site(name=f"{side}_foot", pos=sole.pos)

    # Policy IMU: same point as the real board ("imu" site, board orientation from CAD) but aligned with the
    # pelvis, so the policy reads angular velocity etc. in pelvis axes. On the robot, the IMU driver applies the
    # fixed board->pelvis rotation (robots/wisco_v1/robot.urdf imu_link), so a re-mounted board never needs retraining.
    imu = spec.site("imu")
    spec.body("pelvis").add_site(name="imu_policy", pos=imu.pos)
    S, O = mujoco.mjtSensor, mujoco.mjtObj
    spec.add_sensor(name="imu_ang_vel", type=S.mjSENS_GYRO, objtype=O.mjOBJ_SITE, objname="imu_policy")
    spec.add_sensor(name="imu_lin_vel", type=S.mjSENS_VELOCIMETER, objtype=O.mjOBJ_SITE, objname="imu_policy")
    spec.add_sensor(name="imu_lin_acc", type=S.mjSENS_ACCELEROMETER, objtype=O.mjOBJ_SITE, objname="imu_policy")
    spec.add_sensor(name="imu_upvector", type=S.mjSENS_FRAMEZAXIS, objtype=O.mjOBJ_SITE, objname="imu_policy")
    spec.add_sensor(name="root_angmom", type=S.mjSENS_SUBTREEANGMOM, objtype=O.mjOBJ_BODY, objname="pelvis")
    return spec, renamed


def pelvis_height(model):
    """Pelvis z that puts the lowest foot collision point exactly on the floor in DEFAULT_POSE."""
    d = mujoco.MjData(model)
    d.qpos[2] = 1.0
    d.qpos[3] = 1.0
    for name, q in P.DEFAULT_POSE.items():
        d.qpos[model.jnt_qposadr[model.joint(name).id]] = q
    mujoco.mj_forward(model, d)
    lowest = np.inf
    for gname in P.FOOT_GEOMS:
        g = model.geom(gname).id
        R = d.geom_xmat[g].reshape(3, 3)
        h = model.geom_size[g]
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * h
        lowest = min(lowest, (corners @ R.T + d.geom_xpos[g])[:, 2].min())
    return 1.0 - lowest


def main():
    """Write model/wisco_v1_train.xml and train_model_info.json."""
    spec, renamed = build()
    xml = spec.to_xml()                            # compiles with meshes resolved against robots/wisco_v1/
    # point the saved file at robots/wisco_v1/meshes relative to this folder (works on the VM too)
    assert xml.count('meshdir="meshes/"') == 1
    P.TRAIN_XML.write_text(xml.replace('meshdir="meshes/"', 'meshdir="../meshes/"'))
    model = mujoco.MjModel.from_xml_path(str(P.TRAIN_XML))      # reload: proves the file stands alone

    joints = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    params = P.joint_params()
    assert set(joints) == set(params), (set(joints) ^ set(params))
    z = pelvis_height(model)
    info = {
        "source_model": str(P.SOURCE_XML.relative_to(P.HERE.parents[2])),
        "policy_joints_in_model_order": joints,
        "num_policy_joints": len(joints),
        "fixed_joints": list(P.FIXED_JOINTS),
        "default_pose": {j: params[j]["default"] for j in joints},
        "default_pelvis_height": round(z, 5),
        "ankle_pitch_lower_training": P.ankle_pitch_lower(),
        "total_mass_kg": round(float(model.body_subtreemass[1]), 3),
        "collision_geoms_renamed": renamed,
        "sensors": [model.sensor(i).name for i in range(model.nsensor)],
    }
    INFO_OUT.write_text(json.dumps(info, indent=1))
    print(f"wrote {P.TRAIN_XML.name}: {len(joints)} policy joints, {model.ngeom} geoms, "
          f"mass {info['total_mass_kg']} kg, sensors {info['sensors']}")
    print(f"default pose pelvis height {z:.4f} m (straight-leg stand: 0.7015 m)")
    print(f"wrote {INFO_OUT.name}")


if __name__ == "__main__":
    main()
