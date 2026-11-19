"""Files for the ethercat_sim robot backend: the robot's EtherCAT hardware simulated (hsim_mujoco_plugins'
VirtualRobotSystem: simulated CiA402 drives in MyActuator mode 5, the master's EcCiA402Drive logic, MuJoCo as the
robot), under the controller chain the robot computer runs.

    ros2 run hsim_sim ethercat_sim_backend <world> <out_dir> [--update-rate HZ] [--profile wisco.yaml]

`world` is as for ros2_control_backend.py (a training scene, converted the same way, or an env's
world_ros2_control/world.xml). The robot profile's `ethercat_sim` section names its two-motor mechanisms
(config/robots/<robot>.yaml); a robot without one has a drive on every joint. This writes, into `out_dir`:

  scene.xml         (MJCF input only) as ros2_control_backend.py
  robot.urdf        the `<ros2_control>` block for VirtualRobotSystem: one hardware joint per MOTOR (the joint's own
                    name for a motor on the joint; <mechanism>_motor_a / _b for the two motors of an ankle or the
                    waist), with the drive interfaces EcCiA402Drive exports, and the IMU
  controllers.yaml  the controller manager and the robot computer's controllers:
                      parallel_joints            ParallelLinkageController: the two-motor joints in joint space
                                                 (MyActuator mode 5 through the linkage)
                      leg_targets, arm_targets   JointTargetController: the walking policy's /locomotion/joint_targets
                                                 to the drives as position + kp + kd (the trained gains, from the
                                                 model's actuators); ankle and waist roll/pitch through parallel_joints
                      joint_states               ChainedJointStateBroadcaster: /joint_states in the robot's joints
                      drive_states               joint_state_broadcaster on the motors: /drive_states/joint_states
                                                 and /drive_states/dynamic_joint_states (status word, mode, error code)
                      imu_sensor_broadcaster     /imu

The same JointTargetController / ParallelLinkageController configuration runs on the RK3588
(wisco_ethercat_ros2_hw/config/wisco_joint_targets.yaml, wisco_parallel_joints.yaml); here the gains come from the
simulated model and the timeout holds rather than damps, as on the ros2_control backend. The profile's
`ethercat_sim.command_mode`, if set, replaces the file's ParallelLinkageController command mode (Wisco runs `torque`:
the file's `pd_in_drive` loses the ankles' gain coupling, and the robot falls about 2 s after it stops walking).
"""

import argparse
import os

import yaml

from hsim_sim import ros2_control_backend as rc

NODE = 'virtual_robot'          # VirtualRobotSystem's node: /clock, ground truth, reset_world, set_pause
# Who publishes the sim node's topics on this backend (the interface contract lists them as owners)
NODES = {
    NODE: '/clock, /sim/ground_truth/odom, /sim/ground_truth/state; reset_world, set_pause',
    'joint_states': '/joint_states (ChainedJointStateBroadcaster)',
    'imu_sensor_broadcaster': '/imu (remapped from /imu_sensor_broadcaster/imu)',
}
DRIVE_STATES = ['position', 'velocity', 'effort', 'mode_of_operation', 'status_word', 'error_code']
DRIVE_COMMANDS = ['position', 'velocity', 'effort', 'kp', 'kd', 'mode_of_operation', 'reset_fault', 'max_torque']
PERMILLE_PER_PEAK = 3000.0      # the drive's per-mille unit: peak torque = 3000 per mille (only sets the resolution)


def gains(model, actuator):
    """(kp, kd) of a MuJoCo position actuator: the PD the policy was trained with."""
    return float(model.actuator_gainprm[actuator, 0]), float(-model.actuator_biasprm[actuator, 2])


def mechanisms(ethercat, joints):
    """[{name, joints: [j0, j1], motors: [a, b], directions, peak}] from the profile's ethercat_sim section and the
    ParallelLinkageController configuration it names; [] for a robot with a drive on every joint.
    """
    if not ethercat or not ethercat.get('parallel_joints'):
        return []
    cfg = yaml.safe_load(open(ethercat['parallel_joints']))['parallel_joints']['ros__parameters']
    linkage = yaml.safe_load(open(ethercat['linkage_file']))       # JSON is YAML
    names = {n for n, _ in joints}
    out = []
    for mech in cfg['mechanisms']:
        js = list(linkage[mech]['joints'])
        if not set(js) <= names:
            raise ValueError(f'{mech}: joints {js} are not actuated joints of the model')
        out.append({'name': mech, 'joints': js, 'motors': list(cfg[mech]['motors']),
                    'directions': [float(d) for d in cfg[mech].get('motor_directions', [1.0, 1.0])],
                    'peak': float(ethercat['linkage_motor_peak_nm'][mech])})
    return out


def physics_timestep(model, update_rate):
    """The physics step for a bus cycle of 1/update_rate: the model's when the cycle is a whole number of steps, the
    cycle itself when it is shorter (0 = keep the model's). Anything else cannot run in lockstep.
    """
    period, dt = 1.0 / update_rate, float(model.opt.timestep)
    n = period / dt
    if n < 1.0 - 1e-9:
        return period
    if abs(n - round(n)) > 1e-6:
        raise ValueError(f'update_rate {update_rate} Hz: a bus cycle of {period} s is not a whole number of the '
                         f"model's {dt} s physics steps")
    return 0.0


def motors(model, joints, mechs):
    """[(motor name, {joint params})] in bus order: the motors of single joints (joint order), then the mechanisms'."""
    in_mech = {j for m in mechs for j in m['joints']}
    out = []
    for name, a in joints:
        if name in in_mech:
            continue
        peak = float(abs(model.actuator_forcerange[a]).max()) or 100.0
        out.append((name, {'peak_effort_nm': peak, 'effort_nm_per_permille': peak / PERMILLE_PER_PEAK}))
    for m in mechs:
        for crank, (motor, d) in enumerate(zip(m['motors'], m['directions'])):
            out.append((motor, {'peak_effort_nm': m['peak'], 'effort_nm_per_permille': m['peak'] / PERMILLE_PER_PEAK,
                                'mechanism': m['name'], 'crank': crank, 'direction': d}))
    return out


def urdf(model, joints, mechs, scene_path, keyframe, update_rate, linkage_file='', viewer_rate=0.0,
         viewer_host='host.docker.internal', viewer_port=9870, state_rate=20.0):
    """A URDF holding the `<ros2_control>` block for VirtualRobotSystem (one dummy link per motor keeps it valid)."""
    table = motors(model, joints, mechs)
    lines = ['<?xml version="1.0"?>', '<robot name="virtual_robot">', '  <link name="virtual_robot_root"/>']
    for name, _ in table:
        lines += [f'  <link name="virtual_robot_{name}"/>', f'  <joint name="{name}" type="continuous">',
                  '    <parent link="virtual_robot_root"/>', f'    <child link="virtual_robot_{name}"/>',
                  '    <axis xyz="0 0 1"/>', '  </joint>']
    hw = {'mujoco_model': scene_path, 'keyframe': keyframe, 'control_period': 1.0 / update_rate,
          'physics_timestep': physics_timestep(model, update_rate), 'real_time_factor': 1.0,
          'hold_until_released': 'true', 'settle_cycles': 3, 'mode_of_operation': 5, 'node_name': NODE,
          'state_topic': '/sim/ground_truth/state', 'state_rate': state_rate, 'viewer_host': viewer_host,
          'viewer_port': viewer_port, 'viewer_rate': viewer_rate}
    if mechs:
        hw['linkage_file'] = linkage_file
    base = rc.free_joint(model, joints)
    if base:
        hw.update(free_joint=base, odom_topic='/sim/ground_truth/odom')
    lines += ['  <ros2_control name="VirtualRobot" type="system">', '    <hardware>',
              '      <plugin>hsim_mujoco_plugins/VirtualRobotSystem</plugin>']
    lines += [f'      <param name="{k}">{v}</param>' for k, v in hw.items()]
    lines.append('    </hardware>')
    for name, params in table:
        lines.append(f'    <joint name="{name}">')
        lines += [f'      <command_interface name="{c}"/>' for c in DRIVE_COMMANDS]
        lines += [f'      <state_interface name="{s}"/>' for s in DRIVE_STATES]
        lines += [f'      <param name="{k}">{v}</param>' for k, v in params.items()]
        lines.append('    </joint>')
    lines += ['    <sensor name="imu">']
    lines += [f'      <state_interface name="{q}.{ax}"/>' for q, axes in
              (('orientation', 'xyzw'), ('angular_velocity', 'xyz'), ('linear_acceleration', 'xyz')) for ax in axes]
    lines += [f'      <param name="{k}_sensor">{s}</param>' for k, s in zip(('quat', 'gyro', 'accel'), rc.IMU_NAMES)]
    lines += ['    </sensor>', '  </ros2_control>', '</robot>', '']
    return '\n'.join(lines)


def controllers(model, joints, mechs, update_rate, ethercat=None, imu_frame='imu_link',
                topic='/locomotion/joint_targets', timeout=0.1, ramp_time=0.5):
    """Controller manager parameters: the robot computer's chain, gains from the model's actuators."""
    act = dict(joints)
    chained = [j for m in mechs for j in m['joints']]
    out = {
        'controller_manager': {'ros__parameters': {'update_rate': int(update_rate)}},
        'joint_states': {'ros__parameters': {
            'type': 'wisco_ethercat_ros2_hw/ChainedJointStateBroadcaster', 'joints': [n for n, _ in joints],
            **({'chained_joints': list(chained), 'chain_prefix': 'parallel_joints'} if chained else {})}},
        'drive_states': {'ros__parameters': {
            'type': 'joint_state_broadcaster/JointStateBroadcaster', 'use_local_topics': True}},
        'imu_sensor_broadcaster': {'ros__parameters': {
            'type': 'imu_sensor_broadcaster/IMUSensorBroadcaster', 'sensor_name': 'imu', 'frame_id': imu_frame}},
    }
    if mechs:
        cfg = yaml.safe_load(open(ethercat['parallel_joints']))['parallel_joints']['ros__parameters']
        cfg['linkage_file'] = ethercat['linkage_file']
        if ethercat.get('command_mode'):                    # the profile's choice over the robot computer's file
            cfg['command_mode'] = ethercat['command_mode']
        for m in mechs:
            kp, kd = zip(*(gains(model, act[j]) for j in m['joints']))
            cfg[m['name']].update(kp=list(kp), kd=list(kd))
        out['parallel_joints'] = {'ros__parameters': cfg}
    for name, names in rc.groups(joints).items():
        lo, hi = rc.limits(model, names)
        kp, kd = zip(*(gains(model, act[n]) for n in names))
        mine = [j for j in chained if j in names]
        out[name] = {'ros__parameters': {
            'type': 'wisco_ethercat_ros2_hw/JointTargetController', 'topic': topic, 'joints': list(names),
            **({'chained_joints': mine, 'chain_prefix': 'parallel_joints'} if mine else {}),
            'gain_interfaces': ['kp', 'kd'], 'kp': list(kp), 'kd': list(kd),
            'position_min': lo, 'position_max': hi, 'limit_margin': 0.0,
            'timeout': float(timeout), 'ramp_time': float(ramp_time), 'timeout_action': 'hold'}}
    return out


def build(world, out_dir, update_rate=200, keyframe='init_state', prefix=rc.PREFIX, ethercat=None,
          viewer_rate=0.0, viewer_host='host.docker.internal', viewer_port=9870):
    """Write robot.urdf, controllers.yaml (and scene.xml for an MJCF input) into `out_dir`; return the paths, plus
    'world' (what the plugin loads), 'controllers_to_spawn' (names in start order: one spawner, so the chain comes up
    bottom first) and 'plugin_node'.
    """
    os.makedirs(out_dir, exist_ok=True)
    path, model, spec = rc.load(world, prefix)
    joints = rc.robot_joints(model)
    mechs = mechanisms(ethercat, joints)
    paths = {k: os.path.join(out_dir, f) for k, f in (('urdf', 'robot.urdf'), ('controllers', 'controllers.yaml'))}
    if spec is not None:
        path = paths['scene'] = os.path.join(out_dir, 'scene.xml')
        with open(path, 'w') as f:
            f.write(spec.to_xml())
    with open(paths['urdf'], 'w') as f:
        f.write(urdf(model, joints, mechs, path, keyframe, update_rate, (ethercat or {}).get('linkage_file', ''),
                     viewer_rate, viewer_host, viewer_port))
    ctrl = controllers(model, joints, mechs, update_rate, ethercat)
    with open(paths['controllers'], 'w') as f:
        yaml.safe_dump(ctrl, f, sort_keys=False)
    paths['world'] = path
    paths['controllers_to_spawn'] = (['parallel_joints'] if mechs else []) + list(rc.groups(joints)) + \
        ['joint_states', 'drive_states', 'imu_sensor_broadcaster']
    paths['plugin_node'] = '/' + NODE
    return paths


def main():
    """CLI: write the backend files for a world."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('world')
    ap.add_argument('out_dir')
    ap.add_argument('--update-rate', type=float, default=200.0, help='bus cycle / controller_manager rate, Hz')
    ap.add_argument('--keyframe', default='init_state')
    ap.add_argument('--profile', help="robot profile YAML (its ethercat_sim section: the two-motor mechanisms)")
    a = ap.parse_args()
    ethercat = yaml.safe_load(open(a.profile)).get('ethercat_sim') if a.profile else None
    for k, p in build(a.world, a.out_dir, a.update_rate, a.keyframe, ethercat=ethercat).items():
        print(f'{k}: {p}')


if __name__ == '__main__':
    main()
