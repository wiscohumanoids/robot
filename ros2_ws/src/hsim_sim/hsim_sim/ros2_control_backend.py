"""Files for the ros2_control robot backend: the trained MuJoCo world run by the upstream `mujoco_ros2_control`
plugin, under the robot's own controllers.

    ros2 run hsim_sim ros2_control_backend <world> <out_dir> [--update-rate HZ]

`world` is either the scene a walking policy trained in (mjlab's exported MJCF, the bare training floor) or an
environment's plugin-ready build, `assets/envs/<env>/<build dir>/world_ros2_control/world.xml`, written by
tools/env_build.py next to the env's world.mjb (same model, same qpos layout; MJCF with its assets beside it, since
the plugin bundles its own MuJoCo version and a compiled .mjb only loads in the version that wrote it). This
writes, into `out_dir`:

  scene.xml         (MJCF input only) the same model and physics with mjlab's "robot/" name prefix removed, so the
                    ros2_control joint names are the robot's joint names, and an IMU (`imu_quat`, `imu_gyro`,
                    `imu_accel`) on the site the policy's IMU sensors use. An env build already has both.
  robot.urdf        a description holding only the `<ros2_control>` block: one `position` command interface and
                    position/velocity/effort state per actuated joint, the IMU, the plugin's settings, and the robot's
                    floating base as ground-truth odometry
  controllers.yaml  the controller manager and its controllers, the robot's own:
                      leg_targets, arm_targets   JointTargetController (wisco_ethercat_ros2_hw): the walking policy's
                                                 /locomotion/joint_targets with hold, ramp-in, joint limits and a
                                                 timeout, one instance for the legs/waist/head and one for the arms
                      joint_state_broadcaster    /joint_states
                      imu_sensor_broadcaster     /imu
  plugins.yaml      mujoco_ros2_control plugins: GroundTruthPlugin (hsim_mujoco_plugins), the world's state for the
                    simulated camera and the Mac viewer

Every joint keeps its MuJoCo `position` actuator, so a position command becomes the actuator's ctrl and MuJoCo
computes kp * (target - q) - kd * dq at every physics step: the PD law and rate the policy was trained with, and the
law the drives run in MyActuator mode 5. Those gains live in the model, so the controllers write positions only, and
hold (rather than damp) when targets stop. The plugin rejects any other actuator type for a position interface at
startup, so a model whose actuators changed cannot silently run another law.
"""

import argparse
import os
import re

import mujoco
import yaml

PREFIX = 'robot/'
# Who publishes the sim node's topics on this backend (the interface contract lists them as owners)
NODES = {
    'mujoco_ros2_control_node': '/clock, /sim/ground_truth/odom and (GroundTruthPlugin) /sim/ground_truth/state; '
                                'reset_world, set_pause',
    'joint_state_broadcaster': '/joint_states',
    'imu_sensor_broadcaster': '/imu (remapped from /imu_sensor_broadcaster/imu)',
}
IMU_SENSOR = 'imu_ang_vel'   # the policy's gyro; its site is where the IMU sits
IMU_NAMES = ('imu_quat', 'imu_gyro', 'imu_accel')
ARM_WORDS = ('shoulder', 'elbow', 'forearm', 'wrist')   # arm joints go to arm_targets, the rest to leg_targets
UNLIMITED = 10.0             # rad: the position bound given to the controllers for a joint without limits


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
    """The plugin-ready MjSpec and its compiled model: prefix removed, IMU sensors added on the policy IMU site.
    tools/env_build.py builds an env's world_ros2_control.mjb from this spec.
    """
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


def free_joint(model, joints=None):
    """The robot's floating-base joint: the free joint of the body tree its actuated joints hang from (an env's loose
    objects have free joints too). None for a fixed-base model.
    """
    joints = joints if joints is not None else robot_joints(model)
    if joints:
        body = int(model.jnt_bodyid[model.joint(joints[0][0]).id])
        root = int(model.body_rootid[body])
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and int(model.jnt_bodyid[j]) == root:
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
    base = free_joint(model, joints)
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


def groups(joints):
    """{controller name: [joint names]}: the arms in arm_targets, everything else (legs, waist, head) in leg_targets."""
    names = [n for n, _ in joints]
    arms = [n for n in names if any(w in n for w in ARM_WORDS)]
    out = {'leg_targets': [n for n in names if n not in arms]}
    if arms:
        out['arm_targets'] = arms
    return out


def limits(model, names):
    """(position_min, position_max) per joint from the model (UNLIMITED for joints without limits)."""
    lo, hi = [], []
    for n in names:
        j = model.joint(n).id
        limited = bool(model.jnt_limited[j])
        lo.append(float(model.jnt_range[j][0]) if limited else -UNLIMITED)
        hi.append(float(model.jnt_range[j][1]) if limited else UNLIMITED)
    return lo, hi


def controllers(model, joints, update_rate, imu_frame='imu_link', topic='/locomotion/joint_targets',
                timeout=0.1, ramp_time=0.5):
    """Controller manager parameters for the backend. Each list is its own object: ROS parameter files reject the YAML
    aliases a shared list would be written as. `timeout` counts control-loop (here simulated) time, five policy ticks;
    `ramp_time` is short because the robot starts standing.
    """
    out = {
        'controller_manager': {'ros__parameters': {'update_rate': int(update_rate)}},
        'joint_state_broadcaster': {'ros__parameters': {
            'type': 'joint_state_broadcaster/JointStateBroadcaster', 'joints': [n for n, _ in joints],
            'interfaces': ['position', 'velocity', 'effort']}},
        'imu_sensor_broadcaster': {'ros__parameters': {
            'type': 'imu_sensor_broadcaster/IMUSensorBroadcaster', 'sensor_name': 'imu', 'frame_id': imu_frame}},
    }
    for name, names in groups(joints).items():
        lo, hi = limits(model, names)
        out[name] = {'ros__parameters': {
            # no gain_interfaces (the default, none): an empty list in a parameter file has no type ROS accepts
            'type': 'wisco_ethercat_ros2_hw/JointTargetController', 'topic': topic, 'joints': list(names),
            'position_min': lo, 'position_max': hi, 'limit_margin': 0.0,
            'timeout': float(timeout), 'ramp_time': float(ramp_time), 'timeout_action': 'hold'}}
    return out


def plugins(viewer_host='host.docker.internal', viewer_port=9870, viewer_rate=0.0, state_rate=20.0):
    """mujoco_ros2_control plugin parameters (read by the plugin's node inside the ros2_control process)."""
    return {'/**': {'ros__parameters': {'mujoco_plugins': {'ground_truth': {
        'type': 'hsim_mujoco_plugins/GroundTruthPlugin', 'state_topic': '/sim/ground_truth/state',
        'state_rate': float(state_rate), 'viewer_host': viewer_host, 'viewer_port': int(viewer_port),
        'viewer_rate': float(viewer_rate)}}}}}


def _plugin_ready(model):
    """The model already has the backend's IMU and unprefixed robot joints (an env's world_ros2_control build)."""
    return all(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, s) >= 0 for s in IMU_NAMES) and \
        bool(robot_joints(model))


def load(world, prefix=PREFIX):
    """(path the plugin loads or None, compiled model, spec to write or None) for a world: a plugin-ready build is
    used as is; a training scene (MJCF) is converted. A .mjb must be plugin-ready.
    """
    if world.endswith('.mjb'):
        model = mujoco.MjModel.from_binary_path(world)
        if not _plugin_ready(model):
            raise ValueError(f'{world}: not a plugin-ready world (rebuild it with tools/env_build.py)')
        return world, model, None
    model = mujoco.MjModel.from_xml_path(world)
    if _plugin_ready(model):
        return world, model, None
    spec, model = convert(world, prefix)
    return None, model, spec


def build(world, out_dir, update_rate=200, keyframe='init_state', headless=True, prefix=PREFIX,
          viewer_rate=0.0, viewer_host='host.docker.internal', viewer_port=9870):
    """Write robot.urdf, controllers.yaml, plugins.yaml (and scene.xml for an MJCF input) into `out_dir`; return the
    paths, plus 'world' (what the plugin loads) and 'controllers_to_spawn' (names in start order).
    """
    os.makedirs(out_dir, exist_ok=True)
    path, model, spec = load(world, prefix)
    joints = robot_joints(model)
    paths = {k: os.path.join(out_dir, f) for k, f in
             (('urdf', 'robot.urdf'), ('controllers', 'controllers.yaml'), ('plugins', 'plugins.yaml'))}
    if spec is not None:
        path = paths['scene'] = os.path.join(out_dir, 'scene.xml')
        with open(path, 'w') as f:
            f.write(spec.to_xml())
    with open(paths['urdf'], 'w') as f:
        f.write(urdf(model, joints, path, keyframe, headless))
    ctrl = controllers(model, joints, update_rate)
    with open(paths['controllers'], 'w') as f:
        yaml.safe_dump(ctrl, f, sort_keys=False)
    with open(paths['plugins'], 'w') as f:
        yaml.safe_dump(plugins(viewer_host, viewer_port, viewer_rate), f, sort_keys=False)
    paths['world'] = path
    paths['controllers_to_spawn'] = ['joint_state_broadcaster', 'imu_sensor_broadcaster'] + list(groups(joints))
    return paths


def main():
    """CLI: write the backend files for a world."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('world')
    ap.add_argument('out_dir')
    ap.add_argument('--update-rate', type=float, default=200.0, help='controller_manager rate, Hz')
    ap.add_argument('--keyframe', default='init_state')
    a = ap.parse_args()
    for k, p in build(a.world, a.out_dir, a.update_rate, a.keyframe).items():
        print(f'{k}: {p}')


if __name__ == '__main__':
    main()
