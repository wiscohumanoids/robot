"""MuJoCo simulation node (node name: `sim`). Replaces sim_stub.

Steps the exact world the policy trained in (mjlab's exported scene + training
physics options) in real time. MuJoCo's position actuators are the joint PD loop.

Publishes (all stamped with sim time). Robot sensors are realistic (see sensors.py):
  /clock                    every physics step
  /joint_states             29 G1 joints, names without mjlab's "robot/" prefix (encoders)
  /imu                      pelvis IMU: noisy, biased gyro/accel + onboard attitude filter
                            (tilt accurate, heading drifts; starts at yaw 0, no magnetometer)
  /scan                     head lidar (multi-beam, flattened to 2D), frame lidar_link, 10 Hz
  /battery                  sensor_msgs/BatteryState, drained by motor power, 1 Hz
  /sim/viewer/image_raw     a camera following the robot, for Foxglove (render_hz, off by default)
Privileged (EVALUATION ONLY - graders, tests, and explicitly-labeled stubs; never the robot stack):
  /sim/ground_truth/odom    true pelvis pose; twist = body-frame velocity
  /sim/ground_truth/state   std_msgs/Float64MultiArray [time, qpos...] at 20 Hz
Service:
  /sim/reset                std_srvs/Trigger: put the robot back in the standing keyframe
  /sim/set_body_pose        TEST-ONLY: move a free body in the world (e.g. drop an obstacle)
Side channel (not ROS, for the Mac viewer): qpos streamed over UDP to
viewer_host:viewer_port at viewer_hz, so tools/viewer.sh can show the robot
in a native MuJoCo window. Packet: b"HSIM", uint32 nq, float64 time, float64[nq] qpos.
Subscribes:
  /locomotion/joint_targets joint position targets by name (until the first one arrives,
                            the actuators hold the standing keyframe)

It never reads /cmd_vel: the robot only moves because the policy moves its joints.
"""

import socket
import struct
import threading
import time

import mujoco
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import BatteryState, Image, Imu, JointState, LaserScan
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from hsim_interfaces.srv import SetBodyPose

from hsim_sim.sensors import Battery, ImuModel, Lidar, yaw_of

PREFIX = 'robot/'
# Lidar mount on the torso, about where the real G1's head lidar sits (~1.2 m up)
LIDAR_MOUNT = np.array([0.06, 0.0, 0.42])


def to_time(t: float) -> Time:
    # whole nanoseconds first, then split: t = 5.9999999999 must become 6.000000000, not 5.0
    sec, nsec = divmod(int(round(t * 1e9)), 1_000_000_000)
    return Time(sec=sec, nanosec=nsec)


def wxyz_to_ros(q):
    return float(q[1]), float(q[2]), float(q[3]), float(q[0])


class MujocoSim(Node):
    def __init__(self):
        super().__init__('sim')
        self.declare_parameter('scene', '/hsim/assets/robots/g1/mjlab_1.6.0/g1_velocity_flat.xml')
        self.declare_parameter('keyframe', 'init_state')
        self.declare_parameter('reset_on_fall', True)
        self.declare_parameter('fall_height', 0.45)
        self.declare_parameter('render_hz', 0.0)
        self.declare_parameter('viewer_host', 'host.docker.internal')
        self.declare_parameter('viewer_port', 9870)
        self.declare_parameter('viewer_hz', 50.0)
        self.declare_parameter('render_width', 640)
        self.declare_parameter('render_height', 480)
        self.declare_parameter('seed', 0)
        self.declare_parameter('lidar_hz', 10.0)
        self.declare_parameter('battery_start', 1.0)
        self.declare_parameter('battery_drain_multiplier', 1.0)
        self.declare_parameter('ground_truth_state_hz', 20.0)
        # robot profile (config/robots/<robot>.yaml via the launch file); defaults = G1
        self.declare_parameter('root_body', 'pelvis')
        self.declare_parameter('torso_body', 'torso_link')
        self.declare_parameter('lidar_mount', LIDAR_MOUNT.tolist())   # lidar origin in the torso body, m
        # Lockstep with the walking policy: at every control boundary (policy period, 20 ms) physics waits until a new
        # /locomotion/joint_targets has arrived (up to lockstep_timeout of wall time), like the robot's real-time
        # control loop that never misses a cycle. Without it, a busy CPU (RTAB-Map, rendering, perception) made the
        # policy miss ticks and physics ran 40-90 ms on stale targets (training: <= 10 ms) -> rocking and stumbling.
        # When the CPU can't keep up, sim time now slows down instead. Only active once targets are arriving.
        self.declare_parameter('lockstep', True)
        self.declare_parameter('control_period', 0.02)
        self.declare_parameter('lockstep_timeout', 0.1)
        # physics may run this far past a boundary before it waits (the policy was trained with 0-10 ms of command
        # delay); one 5 ms step lets the ROS round trip overlap with physics instead of stalling it
        self.declare_parameter('lockstep_grace', 0.005)

        scene = self.get_parameter('scene').value
        # .mjb = a built environment (tools/env_build.py); .xml = a plain MJCF world
        self.m = (mujoco.MjModel.from_binary_path(scene) if scene.endswith('.mjb')
                  else mujoco.MjModel.from_xml_path(scene))
        self.d = mujoco.MjData(self.m)
        self.key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, self.get_parameter('keyframe').value)
        self.fall_height = self.get_parameter('fall_height').value
        self.reset_on_fall = self.get_parameter('reset_on_fall').value

        # The robot's joints = hinge joints that have an actuator, mapped by name (actuator
        # order != joint order). Unactuated hinges (e.g. cabinet doors in an environment) are
        # part of the world, not the robot, and are not published on /joint_states.
        act_of_joint = {int(self.m.actuator_trnid[a, 0]): a for a in range(self.m.nu)}
        joints = [j for j in range(self.m.njnt)
                  if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and j in act_of_joint]
        self.joint_names = [self.m.joint(j).name.removeprefix(PREFIX) for j in joints]
        self.qpos_idx = np.array([self.m.jnt_qposadr[j] for j in joints])
        self.qvel_idx = np.array([self.m.jnt_dofadr[j] for j in joints])
        self.act_of_name = {self.m.joint(j).name.removeprefix(PREFIX): act_of_joint[j] for j in joints}
        # joint velocity slot for each actuator, for motor power (battery)
        self.act_dof = np.array([self.m.jnt_dofadr[self.m.actuator_trnid[a, 0]] for a in range(self.m.nu)])

        def sensor(name):
            s = self.m.sensor(PREFIX + name)
            return slice(s.adr[0], s.adr[0] + s.dim[0])
        self.s_gyro, self.s_vel, self.s_acc = sensor('imu_ang_vel'), sensor('imu_lin_vel'), sensor('imu_lin_acc')
        self.pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + self.get_parameter('root_body').value)
        self.torso = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + self.get_parameter('torso_body').value)
        if self.pelvis < 0 or self.torso < 0:
            raise RuntimeError(f'root/torso body not found in {scene} (check the robot profile)')
        self.lidar_mount = np.array(self.get_parameter('lidar_mount').value, dtype=float)

        # Move the robot's own geometry to groups 4/5 so ray-based sensors don't see the robot.
        # Groups only affect drawing and ray filtering, not physics.
        for g in range(self.m.ngeom):
            if self.m.body(self.m.geom_bodyid[g]).name.startswith(PREFIX):
                self.m.geom_group[g] = 4 if self.m.geom_group[g] <= 2 else 5

        rng = np.random.default_rng(self.get_parameter('seed').value)
        self.imu = ImuModel(rng, self.m.opt.timestep)
        self.lidar = Lidar(self.m, rng)
        self.battery = Battery(start_soc=self.get_parameter('battery_start').value,
                               drain_multiplier=self.get_parameter('battery_drain_multiplier').value)

        self.lock = threading.Lock()
        self.reset()

        self.pub_clock = self.create_publisher(Clock, '/clock', 10)
        self.pub_js = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_imu = self.create_publisher(Imu, '/imu', 10)
        self.pub_gt = self.create_publisher(Odometry, '/sim/ground_truth/odom', 10)
        self.pub_img = self.create_publisher(Image, '/sim/viewer/image_raw', 2)
        self.pub_scan = self.create_publisher(LaserScan, '/scan', 5)
        self.pub_batt = self.create_publisher(BatteryState, '/battery', 5)
        self.pub_state = self.create_publisher(Float64MultiArray, '/sim/ground_truth/state', 5)
        self.batt_next = self.state_next = 0.0
        self.state_period = 1.0 / max(self.get_parameter('ground_truth_state_hz').value, 1e-6)
        self.create_subscription(JointState, '/locomotion/joint_targets', self.on_targets, 10)
        self.create_service(Trigger, '/sim/reset', self.on_reset)
        self.create_service(SetBodyPose, '/sim/set_body_pose', self.on_set_body_pose)
        self.unknown_warned = set()

        self.viewer_addr, self.viewer_sock = None, None
        viewer_hz = self.get_parameter('viewer_hz').value
        if viewer_hz > 0:
            host, port = self.get_parameter('viewer_host').value, self.get_parameter('viewer_port').value
            try:
                self.viewer_addr = (socket.gethostbyname(host), port)
                self.viewer_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.viewer_period, self.viewer_next = 1.0 / viewer_hz, 0.0
            except OSError as e:
                self.get_logger().warn(f'viewer stream off: cannot resolve {host} ({e})')

        # Two sims would fight over /clock and the robot; refuse to run as the second one.
        self.single_sim_timer = self.create_timer(2.0, self.check_single_sim)

        self.dt = self.m.opt.timestep
        self.wall_start, self.sim_start = time.monotonic(), self.d.time
        self.lockstep = bool(self.get_parameter('lockstep').value)
        self.ctrl_period = float(self.get_parameter('control_period').value)
        self.lockstep_timeout = float(self.get_parameter('lockstep_timeout').value)
        self.lockstep_grace = float(self.get_parameter('lockstep_grace').value)
        self.target_seq = self.used_seq = 0
        self.next_boundary = None      # sim time of the next control boundary (set when targets start arriving)
        self.wait_since = None
        self.lockstep_waits = self.lockstep_timeouts = 0
        self.create_timer(self.dt, self.step)
        self.stats_t, self.stats_steps = time.monotonic(), 0

        if self.get_parameter('render_hz').value > 0:
            threading.Thread(target=self.render_loop, daemon=True).start()
        if self.get_parameter('lidar_hz').value > 0:
            threading.Thread(target=self.lidar_loop, daemon=True).start()
        self.get_logger().info(
            f'MuJoCo {mujoco.__version__} sim: {scene} | dt={self.dt}s, {len(self.joint_names)} joints, '
            f'integrator={mujoco.mjtIntegrator(self.m.opt.integrator).name}')

    def reset(self):
        """Back to the standing keyframe. Sim time keeps running forward: /clock must
        never jump backwards, or every node on sim time gets confused."""
        with self.lock:
            t = self.d.time
            mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
            self.d.time = t
            mujoco.mj_forward(self.m, self.d)
            self.imu.reset(self.d.xquat[self.pelvis])     # like a power cycle of the IMU

    def check_single_sim(self):
        self.destroy_timer(self.single_sim_timer)
        if self.count_publishers('/clock') > 1:
            self.get_logger().fatal('another node is already publishing /clock (a second sim?). '
                                    'Stop it first: tools/ros.sh stop. Exiting.')
            raise SystemExit(1)

    def on_reset(self, _req, resp):
        self.reset()
        self.get_logger().info('reset to standing keyframe (service call)')
        resp.success, resp.message = True, 'reset'
        return resp

    def on_set_body_pose(self, req, resp):
        b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, req.body)
        if b < 0 or self.m.body_jntnum[b] != 1 or self.m.jnt_type[self.m.body_jntadr[b]] != mujoco.mjtJoint.mjJNT_FREE:
            resp.success, resp.message = False, f'"{req.body}" is not a free body in this world'
            return resp
        if req.body.startswith(PREFIX):
            resp.success, resp.message = False, 'refusing to teleport the robot'
            return resp
        j = self.m.body_jntadr[b]
        qa, va = self.m.jnt_qposadr[j], self.m.jnt_dofadr[j]
        p, o = req.pose.position, req.pose.orientation
        with self.lock:
            self.d.qpos[qa:qa + 7] = [p.x, p.y, p.z, o.w or 1.0, o.x, o.y, o.z]
            self.d.qvel[va:va + 6] = 0.0
            mujoco.mj_forward(self.m, self.d)
        self.get_logger().info(f'test hook: moved {req.body} to ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})')
        resp.success, resp.message = True, 'moved'
        return resp

    def stream_to_viewer(self):
        if self.viewer_sock is None or self.d.time < self.viewer_next:
            return
        self.viewer_next = self.d.time + self.viewer_period
        packet = b'HSIM' + struct.pack('<Id', self.m.nq, self.d.time) + self.d.qpos.astype('<f8').tobytes()
        try:
            self.viewer_sock.sendto(packet, self.viewer_addr)
        except OSError:
            pass  # viewer not running; that's fine

    def on_targets(self, msg: JointState):
        with self.lock:
            for name, pos in zip(msg.name, msg.position):
                a = self.act_of_name.get(name)
                if a is None:
                    if name not in self.unknown_warned:
                        self.get_logger().warn(f'target for unknown joint "{name}" ignored')
                        self.unknown_warned.add(name)
                    continue
                self.d.ctrl[a] = pos
            self.target_seq += 1

    def _policy_ready(self) -> bool:
        """Lockstep gate: False = hold physics at this control boundary until the policy's new targets arrive."""
        if not self.lockstep or self.target_seq == 0:
            return True
        if self.next_boundary is None:
            self.next_boundary = self.d.time + self.ctrl_period
            self.used_seq = self.target_seq
            return True
        if self.target_seq != self.used_seq and self.d.time + 1e-9 >= self.next_boundary:
            pass                                      # new targets are in: move on to the next boundary
        elif self.d.time + 1e-9 < self.next_boundary + self.lockstep_grace:
            return True
        if self.target_seq == self.used_seq:          # nothing new since the last boundary: wait (bounded)
            now = time.monotonic()
            if self.wait_since is None:
                self.wait_since = now
                self.lockstep_waits += 1
            if now - self.wait_since < self.lockstep_timeout:
                return False
            self.lockstep_timeouts += 1               # policy gone or stuck: don't freeze the world
        self.used_seq = self.target_seq
        self.wait_since = None
        while self.next_boundary <= self.d.time + 1e-9:
            self.next_boundary += self.ctrl_period
        return True

    def step(self):
        # Catch up if the timer fell behind, but never run ahead of wall time.
        target_sim = self.sim_start + (time.monotonic() - self.wall_start)
        steps = 0
        while self.d.time + self.dt <= target_sim + 1e-9 and steps < 8:
            if not self._policy_ready():
                break          # waiting for the policy; the next timer call catches up to wall time
            with self.lock:
                mujoco.mj_step(self.m, self.d)
                self.battery.step(self.d.actuator_force, self.d.qvel[self.act_dof], self.dt)
            steps += 1
            self.publish()
        self.stream_to_viewer()
        if target_sim - self.d.time > 0.1:   # > 100 ms behind: drop the backlog instead of spiralling
            self.wall_start, self.sim_start = time.monotonic(), self.d.time
        self.stats_steps += steps

        if self.d.xpos[self.pelvis][2] < self.fall_height and self.reset_on_fall:
            self.get_logger().warn('robot fell, resetting to the standing keyframe')
            self.reset()

        now = time.monotonic()
        if now - self.stats_t > 10.0:
            rtf = self.stats_steps * self.dt / (now - self.stats_t)
            if rtf < 0.9:
                self.get_logger().warn(f'real-time factor {rtf:.2f}' + (
                    f' (lockstep: waited for the policy {self.lockstep_waits}x, {self.lockstep_timeouts} timeouts)'
                    if self.lockstep else ''))
            self.stats_t, self.stats_steps = now, 0
            self.lockstep_waits = self.lockstep_timeouts = 0

    def publish(self):
        d = self.d
        stamp = to_time(d.time)
        self.pub_clock.publish(Clock(clock=stamp))

        js = JointState()
        js.header.stamp = stamp
        js.name = self.joint_names
        js.position = d.qpos[self.qpos_idx].tolist()
        js.velocity = d.qvel[self.qvel_idx].tolist()
        js.effort = [float(d.actuator_force[self.act_of_name[n]]) for n in self.joint_names]
        self.pub_js.publish(js)

        # IMU (site aligned with the pelvis): what the sensor + its onboard filter report
        q_est, gyro, acc = self.imu.measure(d.sensordata[self.s_gyro], d.sensordata[self.s_acc])
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = 'imu_link'
        imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w = wxyz_to_ros(q_est)
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = map(float, gyro)
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = map(float, acc)
        imu.orientation_covariance = [1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-2]
        imu.angular_velocity_covariance = [self.imu.gyro_noise ** 2 if i in (0, 4, 8) else 0.0 for i in range(9)]
        imu.linear_acceleration_covariance = [self.imu.accel_noise ** 2 if i in (0, 4, 8) else 0.0 for i in range(9)]
        self.pub_imu.publish(imu)

        if d.time >= self.batt_next:
            self.batt_next = d.time + 1.0
            b = BatteryState()
            b.header.stamp = stamp
            b.voltage, b.percentage = self.battery.voltage, float(self.battery.soc)
            b.current = -float(self.battery.power_w / self.battery.voltage)
            b.capacity = b.design_capacity = 9.0
            b.charge = float(9.0 * self.battery.soc)
            b.present = True
            b.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
            self.pub_batt.publish(b)

        if d.time >= self.state_next:
            self.state_next = d.time + self.state_period
            self.pub_state.publish(Float64MultiArray(data=[float(d.time), *map(float, d.qpos)]))

        q = wxyz_to_ros(d.xquat[self.pelvis])

        gt = Odometry()
        gt.header.stamp = stamp
        gt.header.frame_id = 'odom'
        gt.child_frame_id = 'base_link'
        p = d.xpos[self.pelvis]
        gt.pose.pose.position.x, gt.pose.pose.position.y, gt.pose.pose.position.z = map(float, p)
        o = gt.pose.pose.orientation
        o.x, o.y, o.z, o.w = q
        v, w = d.sensordata[self.s_vel], d.sensordata[self.s_gyro]
        gt.twist.twist.linear.x, gt.twist.twist.linear.y, gt.twist.twist.linear.z = map(float, v)
        gt.twist.twist.angular.x, gt.twist.twist.angular.y, gt.twist.twist.angular.z = map(float, w)
        self.pub_gt.publish(gt)

    def render_loop(self):
        """Off-screen camera that follows the robot. Uses its own MjData copy so
        rendering never blocks physics for longer than a state copy."""
        w, h = self.get_parameter('render_width').value, self.get_parameter('render_height').value
        period = 1.0 / self.get_parameter('render_hz').value
        try:
            renderer = mujoco.Renderer(self.m, height=h, width=w)
        except Exception as e:  # no GL available: keep simulating without images
            self.get_logger().warn(f'rendering disabled ({e}); set MUJOCO_GL=osmesa or egl')
            return
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = self.pelvis
        cam.distance, cam.elevation, cam.azimuth = 3.0, -15.0, 135.0
        opt = mujoco.MjvOption()
        opt.geomgroup[4] = 1                      # draw the robot (its geoms live in group 4)
        view = mujoco.MjData(self.m)
        while rclpy.ok():
            t0 = time.monotonic()
            with self.lock:
                view.qpos[:], view.qvel[:], view.time = self.d.qpos, self.d.qvel, self.d.time
            mujoco.mj_forward(self.m, view)
            renderer.update_scene(view, camera=cam, scene_option=opt)
            pixels = renderer.render()
            img = Image()
            img.header.stamp = to_time(view.time)
            img.header.frame_id = 'sim_viewer'
            img.height, img.width, img.encoding, img.step = h, w, 'rgb8', w * 3
            img.data = pixels.tobytes()
            self.pub_img.publish(img)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))


    def lidar_loop(self):
        """Head lidar on its own thread with its own state copy: a scan takes ~10-20 ms in a
        busy kitchen, and physics must not wait for it."""
        period = 1.0 / self.get_parameter('lidar_hz').value
        view = mujoco.MjData(self.m)
        inc = 2 * np.pi / self.lidar.n_az
        while rclpy.ok():
            t0 = time.monotonic()
            with self.lock:
                view.qpos[:], view.time = self.d.qpos, self.d.time
            mujoco.mj_kinematics(self.m, view)
            R_torso = view.xmat[self.torso].reshape(3, 3)
            origin = view.xpos[self.torso] + R_torso @ self.lidar_mount
            ranges = self.lidar.scan(view, origin, yaw_of(view.xquat[self.pelvis]))
            msg = LaserScan()
            msg.header.stamp = to_time(view.time)
            msg.header.frame_id = 'lidar_link'
            msg.angle_min, msg.angle_increment = -np.pi, inc
            msg.angle_max = -np.pi + inc * (self.lidar.n_az - 1)
            msg.scan_time, msg.time_increment = period, 0.0
            msg.range_min, msg.range_max = self.lidar.range_min, self.lidar.range_max
            msg.ranges = ranges.astype(np.float32).tolist()
            self.pub_scan.publish(msg)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))


def main():
    rclpy.init()
    node = MujocoSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
