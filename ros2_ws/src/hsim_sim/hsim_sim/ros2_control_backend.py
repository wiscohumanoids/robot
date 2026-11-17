"""Files for the ros2_control robot backend: the trained MuJoCo scene run by the upstream `mujoco_ros2_control` plugin.

    ros2 run hsim_sim ros2_control_backend <scene.xml> <out_dir> [--update-rate HZ]

From the scene a walking policy trained in (mjlab's exported MJCF) this writes, into `out_dir`:
  scene.xml         the same model and physics with mjlab's "robot/" name prefix removed (so the ros2_control joint
                    names are the robot's joint names) and an IMU (`imu_quat`, `imu_gyro`, `imu_accel`) on the site
                    the policy's IMU sensors use
  robot.urdf        a description holding only the `<ros2_control>` block: one `position` command interface and
                    position/velocity/effort state per actuated joint, the IMU, and the plugin's settings
  controllers.yaml  controller_manager: joint_state_broadcaster, imu_sensor_broadcaster, and a forward position
                    controller (`joint_targets_controller`) over the same joints
  joints.yaml       joint order and standing targets, for joint_target_bridge

Every joint keeps its MuJoCo `position` actuator, so a position command is written straight to the actuator's ctrl
and MuJoCo computes kp * (target - q) - kd * dq at every physics step: the same PD law and rate the policy was trained
with, and the law the drives run in MyActuator mode 5 (position + kp + kd per cycle). The plugin rejects any other
actuator type for a position interface at startup, so a scene whose actuators changed cannot silently run another law.
"""

import argparse
import os
import re

import mujoco
import yaml

PREFIX = 'robot/'
# Who publishes the sim node's topics on this backend (the interface contract lists them as owners)
NODES = {
    'mujoco_ros2_control_node': '/clock and /sim/ground_truth/odom (the plugin\'s own node), reset_world, set_pause',
    'joint_state_broadcaster': '/joint_states',
    'imu_sensor_broadcaster': '/imu (remapped from /imu_sensor_broadcaster/imu)',
}
IMU_SENSOR = 'imu_ang_vel'   # the policy's gyro; its site is where the IMU sits


def _strip(xml, prefix):
    """Remove `prefix` from every attribute value that starts with it: element names and the references to them
    (geom meshes, sensor sites, actuator joints...). Renaming MjSpec elements alone would leave the references stale.
    Default class names keep it: "robot/main" would become MuJoCo's reserved "main" class.
    """
    return re.sub(r'(?<!class=)"' + re.escape(prefix), '"', xml)


def _absolute_asset_dirs(spec, scene):
    """Point meshdir/texturedir at the original scene's folders, so the copy written elsewhere still finds them."""
    base = os.path.dirname(os.path.abspath(scene))
    for attr in ('meshdir', 'texturedir'):
        d = getattr(spec, attr)
        setattr(spec, attr, d if os.path.isabs(d) else os.path.join(base, d))


def robot_joints(model):
    """(joint name, actuator id) for every hinge joint driven by a position actuator, in joint order."""
    act_of_joint = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)
                    if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
    out = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or j not in act_of_joint:
            continue
        a = act_of_joint[j]
        if model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE or model.actuator_biasprm[a, 1] >= 0:
            raise ValueError(f'joint {model.joint(j).name}: actuator {model.actuator(a).name} is not a position '
                             f'actuator (kp, kd); the backend needs one per joint')
        out.append((model.joint(j).name, a))
    return out


def convert(scene, prefix=PREFIX):
    """The plugin-ready MjSpec and its compiled model: prefix removed, IMU sensors added on the policy IMU site."""
    spec = mujoco.MjSpec.from_file(scene)
    _absolute_asset_dirs(spec, scene)
    spec = mujoco.MjSpec.from_string(_strip(spec.to_xml(), prefix))
    model = spec.compile()
    gyro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, IMU_SENSOR.removeprefix(prefix))
    if gyro < 0:
        raise ValueError(f'{scene}: no "{IMU_SENSOR}" sensor, so the IMU site is unknown')
    site = model.site(int(model.sensor_objid[gyro])).name
    spec.add_sensor(name='imu_quat', type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site)
    spec.add_sensor(name='imu_gyro', type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site)
    spec.add_sensor(name='imu_accel', type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site)
    return spec, spec.compile()


def free_joint(model):
    """The floating base's joint name (None for a fixed-base model)."""
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return model.joint(j).name
    return None


def urdf(model, joints, scene_path, keyframe, headless=True):
    """A URDF holding the `<ros2_control>` block for the plugin (one dummy link per joint keeps the URDF valid)."""
    lines = ['<?xml version="1.0"?>', '<robot name="mujoco_backend">', '  <link name="mujoco_backend_root"/>']
    for name, a in joints:
        j = model.joint(name)
        lo, hi = (float(v) for v in model.jnt_range[j.id])
        effort = float(abs(model.actuator_forcerange[a]).max()) or 1000.0
        limited = bool(model.jnt_limited[j.id])
        lines += [f'  <link name="mujoco_backend_{name}"/>',
                  f'  <joint name="{name}" type="{"revolute" if limited else "continuous"}">',
                  '    <parent link="mujoco_backend_root"/>', f'    <child link="mujoco_backend_{name}"/>',
                  '    <axis xyz="0 0 1"/>',
                  f'    <limit lower="{lo}" upper="{hi}" effort="{effort}" velocity="100.0"/>' if limited else
                  f'    <limit effort="{effort}" velocity="100.0"/>',
                  '  </joint>']
    hw = {'mujoco_model': scene_path, 'headless': str(headless).lower(), 'initial_keyframe': keyframe,
          'sim_speed_factor': '1.0'}
    base = free_joint(model)
    if base:
        hw.update(odom_free_joint_name=base, odom_topic='/sim/ground_truth/odom')
    lines += ['  <ros2_control name="MujocoSystem" type="system">', '    <hardware>',
              '      <plugin>mujoco_ros2_control/MujocoSystemInterface</plugin>']
    lines += [f'      <param name="{k}">{v}</param>' for k, v in hw.items()]
    lines.append('    </hardware>')
    qpos0 = _keyframe_qpos(model, keyframe)
    for name, _ in joints:
        q0 = float(qpos0[model.jnt_qposadr[model.joint(name).id]])
        lines += [f'    <joint name="{name}">', '      <command_interface name="position"/>',
                  f'      <state_interface name="position"><param name="initial_value">{q0}</param></state_interface>',
                  '      <state_interface name="velocity"/>', '      <state_interface name="effort"/>', '    </joint>']
    lines += ['    <sensor name="imu">']
    lines += [f'      <state_interface name="{q}.{ax}"/>' for q, axes in
              (('orientation', 'xyzw'), ('angular_velocity', 'xyz'), ('linear_acceleration', 'xyz')) for ax in axes]
    lines += ['      <param name="mujoco_type">imu</param>', '      <param name="mujoco_sensor_name">imu</param>',
              '    </sensor>', '  </ros2_control>', '</robot>', '']
    return '\n'.join(lines)


def _keyframe_qpos(model, keyframe):
    """qpos of the named keyframe (the model's default pose when it has none of that name)."""
    k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    return model.key_qpos[k] if k >= 0 else model.qpos0


def standing_targets(model, joints, keyframe):
    """Each joint's standing target: the keyframe's actuator ctrl, else its keyframe position."""
    k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    qpos = _keyframe_qpos(model, keyframe)
    if k >= 0 and model.nu:
        return [float(model.key_ctrl[k][a]) for _, a in joints]
    return [float(qpos[model.jnt_qposadr[model.joint(n).id]]) for n, _ in joints]


def controllers(joints, update_rate, imu_frame='imu_link'):
    """controller_manager parameters for the backend (each list its own object: ROS parameter files reject the YAML
    aliases a shared list would be written as).
    """
    def names():
        """A fresh list of the joint names."""
        return [n for n, _ in joints]
    return {
        'controller_manager': {'ros__parameters': {'update_rate': int(update_rate)}},
        'joint_state_broadcaster': {'ros__parameters': {
            'type': 'joint_state_broadcaster/JointStateBroadcaster', 'joints': names(),
            'interfaces': ['position', 'velocity', 'effort']}},
        'imu_sensor_broadcaster': {'ros__parameters': {
            'type': 'imu_sensor_broadcaster/IMUSensorBroadcaster', 'sensor_name': 'imu', 'frame_id': imu_frame}},
        'joint_targets_controller': {'ros__parameters': {
            'type': 'forward_command_controller/ForwardCommandController', 'joints': names(),
            'interface_name': 'position'}},
    }


def build(scene, out_dir, update_rate=200, keyframe='init_state', headless=True, prefix=PREFIX):
    """Write scene.xml, robot.urdf, controllers.yaml and joints.yaml into `out_dir`; return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    spec, model = convert(scene, prefix)
    joints = robot_joints(model)
    paths = {k: os.path.join(out_dir, f) for k, f in
             (('scene', 'scene.xml'), ('urdf', 'robot.urdf'), ('controllers', 'controllers.yaml'),
              ('joints', 'joints.yaml'))}
    with open(paths['scene'], 'w') as f:
        f.write(spec.to_xml())
    with open(paths['urdf'], 'w') as f:
        f.write(urdf(model, joints, paths['scene'], keyframe, headless))
    with open(paths['controllers'], 'w') as f:
        yaml.safe_dump(controllers(joints, update_rate), f, sort_keys=False)
    with open(paths['joints'], 'w') as f:
        yaml.safe_dump({'joints': [n for n, _ in joints],
                        'standing': standing_targets(model, joints, keyframe)}, f, sort_keys=False)
    return paths


def main():
    """CLI: write the backend files for a scene."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('scene')
    ap.add_argument('out_dir')
    ap.add_argument('--update-rate', type=float, default=200.0, help='controller_manager rate, Hz')
    ap.add_argument('--keyframe', default='init_state')
    a = ap.parse_args()
    for k, p in build(a.scene, a.out_dir, a.update_rate, a.keyframe).items():
        print(f'{k}: {p}')


if __name__ == '__main__':
    main()
