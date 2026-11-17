// VirtualRobotSystem: the robot's EtherCAT hardware, simulated. A ros2_control hardware plugin that stands in for
// ethercat_driver_ros2 on the robot computer, with MuJoCo as the robot.
//
//   controllers ──command interfaces──► Cia402DriveMaster ──RxPDO──► RmdxMockSlave (one per motor) ──torque──► MuJoCo
//   (the robot's)  ◄──state interfaces── (EcCiA402Drive's   ◄─TxPDO── (CiA402 state machine,     ◄──q, q̇───
//                                         processData)                mode 5 PD in the drive)
//
// What the controllers see is what they see on the robot: one hardware joint per MOTOR, with the interfaces
// EcCiA402Drive exports (state position, velocity, effort, mode_of_operation, status_word, error_code; command
// position, velocity, effort, kp, kd, mode_of_operation, reset_fault, max_torque), scaled to SI units from the
// drive's counts and per-mille values, one bus cycle late in each direction, through a CiA402 state machine the
// master enables at startup. Each drive runs its servo law at every physics step (mode 5 by default: kp, kd from
// the bus, PD in the drive) on the joint MuJoCo simulates. MuJoCo's own position actuators for these joints are
// switched off: the drives' torques are the only actuation.
//
// Joints whose motors sit on the joint (hips, knees, ...) are one drive each. A two-motor mechanism (Wisco's ankles,
// waist roll/pitch: robots/<robot>/linkage.json) is two drives whose motor angles come from the serial joints'
// angles through the rod geometry (parallel_linkage.hpp, the controller's own math), and whose torques reach the
// joints as J^T tau: the controllers then need ParallelLinkageController on top, as on the robot.
//
// Simulated time: write() steps the physics one control period (whole physics steps), publishes /clock and paces
// itself to `real_time_factor`. Run the controller manager with use_sim_time, as with mujoco_ros2_control. Its loop
// sleeps on the ROS clock and starts before this plugin is loaded, so until the loop first calls write() a wall
// timer ticks the (frozen) world's time instead.
//
// Also published (from the `virtual_robot` node): /sim/ground_truth/state and the viewer stream
// (ground_truth_stream.hpp), /sim/ground_truth/odom (the floating base, body-frame twist); the IMU as a sensor
// component's state interfaces (imu_sensor_broadcaster). Services: ~/reset_world (keyframe; simulated time keeps
// running forward), ~/set_pause (write() blocks while paused, so the controller manager and /clock stop, as with
// mujoco_ros2_control). The world starts frozen at the keyframe (time runs, physics does not) until the first
// unpause (`hold_until_released`), and stays frozen for `settle_cycles` after a reset, while the bus carries the
// controllers' new targets to the drives.
//
// <hardware> parameters: mujoco_model (MJCF; hsim_sim/ethercat_sim_backend.py writes it), keyframe, control_period
// (s; a whole number of physics steps), physics_timestep (0 = the model's), real_time_factor, hold_until_released,
// settle_cycles, mode_of_operation (0x6060 when the command interface is unset; 5), linkage_file, free_joint,
// odom_topic, node_name, state_topic, state_rate, viewer_host, viewer_port, viewer_rate.
// <joint> parameters (one per motor): counts_per_rev, effort_nm_per_permille, peak_effort_nm; for a motor of a
// two-motor mechanism also mechanism (linkage.json name), crank (0 or 1), direction (+1 / -1, as
// ParallelLinkageController's motor_directions). Any other joint is the MuJoCo hinge of that name.
// <sensor> (one, optional): the IMU; parameters quat_sensor, gyro_sensor, accel_sensor (MuJoCo sensor names).
#ifndef HSIM_MUJOCO_PLUGINS__VIRTUAL_ROBOT_SYSTEM_HPP_
#define HSIM_MUJOCO_PLUGINS__VIRTUAL_ROBOT_SYSTEM_HPP_

#include <mujoco/mujoco.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "hsim_mujoco_plugins/ground_truth_stream.hpp"
#include "mujoco_ros2_control_msgs/srv/reset_world.hpp"
#include "mujoco_ros2_control_msgs/srv/set_pause.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rosgraph_msgs/msg/clock.hpp"
#include "wisco_ethercat_ros2_hw/mock/cia402_master.hpp"
#include "wisco_ethercat_ros2_hw/mock/rmdx_mock_slave.hpp"
#include "wisco_ethercat_ros2_hw/parallel_linkage.hpp"

namespace hsim_mujoco_plugins
{

class VirtualRobotSystem : public hardware_interface::SystemInterface
{
public:
  ~VirtualRobotSystem() override;
  CallbackReturn on_init(const hardware_interface::HardwareComponentInterfaceParams & params) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  using Drive = wisco_ethercat_ros2_hw::mock::RmdxMockSlave;
  using Master = wisco_ethercat_ros2_hw::mock::Cia402DriveMaster;
  using Vec2 = wisco_ethercat_ros2_hw::linkage::Vec2;
  using Mat2 = wisco_ethercat_ros2_hw::linkage::Mat2;

  struct Motor                                   // one drive on the bus
  {
    Drive drive;
    Master master;
    wisco_ethercat_ros2_hw::mock::DriveCommands cmd;
    wisco_ethercat_ros2_hw::mock::DriveStates st;
    wisco_ethercat_ros2_hw::mock::TxPdo inbox;
    int linkage = -1;                            // -1: on its own joint (dof); else linkages_[linkage], crank
    int crank = 0;
    int qadr = 0, dof = 0;
  };

  struct Linkage                                 // a two-motor mechanism over two serial MuJoCo joints
  {
    std::string name;
    wisco_ethercat_ros2_hw::linkage::Mechanism mech;
    std::array<int, 2> qadr{}, dof{}, motor{{-1, -1}};
    Vec2 offset{}, direction{{1.0, 1.0}};       // crank angle at the joint zero pose; motor reading sign
    Vec2 crank{};                                // latest crank angles (branch reference for the next solve)
    Mat2 jacobian{};
    bool ok = false;
  };

  // Crank angles and J of linkage l at the current MuJoCo state; false when the rods cannot close there.
  bool solve_linkage(Linkage & l);
  // Each drive's state now (before a bus cycle) and one physics step's motor torques -> qfrc_applied.
  void measure_drives();
  void apply_drive_torques();
  void publish(double t);
  void pace();
  void on_reset(
    const std::shared_ptr<mujoco_ros2_control_msgs::srv::ResetWorld::Request> req,
    std::shared_ptr<mujoco_ros2_control_msgs::srv::ResetWorld::Response> res);
  void on_pause(
    const std::shared_ptr<mujoco_ros2_control_msgs::srv::SetPause::Request> req,
    std::shared_ptr<mujoco_ros2_control_msgs::srv::SetPause::Response> res);

  mjModel * model_ = nullptr;
  mjData * data_ = nullptr;
  int keyframe_ = -1;
  int substeps_ = 1;
  double control_period_ = 0.005, real_time_factor_ = 1.0;
  std::vector<Motor> motors_;
  std::vector<Linkage> linkages_;
  bool have_frame_ = false;
  int settle_cycles_ = 3, frozen_cycles_ = 0;

  // IMU (sensor component), floating base
  int imu_quat_ = -1, imu_gyro_ = -1, imu_accel_ = -1;
  std::array<double, 10> imu_{};                 // orientation xyzw, angular velocity xyz, linear acceleration xyz
  int base_body_ = -1;

  // ROS side (own node, on the controller manager's executor)
  rclcpp::Node::SharedPtr node_;
  rclcpp::Executor::WeakPtr executor_;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Service<mujoco_ros2_control_msgs::srv::ResetWorld>::SharedPtr reset_srv_;
  rclcpp::Service<mujoco_ros2_control_msgs::srv::SetPause>::SharedPtr pause_srv_;
  rclcpp::TimerBase::SharedPtr startup_clock_;   // ticks /clock until the control loop calls write()
  std::atomic<bool> writing_{false};
  GroundTruthStream ground_truth_;
  nav_msgs::msg::Odometry odom_;

  // threads: write() (control loop) and the service callbacks (executor)
  std::mutex data_mutex_;                        // model_, data_, drives
  std::mutex pause_mutex_;
  std::condition_variable pause_cv_;
  bool paused_ = false;
  std::atomic<bool> released_{false};
  std::chrono::steady_clock::time_point wall_anchor_;
  double sim_anchor_ = 0.0;
  std::atomic<bool> anchored_{false};
  uint64_t unsolved_ = 0;
};

}  // namespace hsim_mujoco_plugins

#endif  // HSIM_MUJOCO_PLUGINS__VIRTUAL_ROBOT_SYSTEM_HPP_
