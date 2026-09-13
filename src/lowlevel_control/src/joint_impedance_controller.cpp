#include "lowlevel_control/joint_impedance_controller.hpp"

#include <algorithm>
#include <cmath>

#include "hardware_interface/loaned_command_interface.hpp"
#include "hardware_interface/loaned_state_interface.hpp"
#include "rclcpp/rclcpp.hpp"

namespace lowlevel_control
{

using humanoid_interfaces::msg::JointCommand;
using humanoid_interfaces::msg::RobotState;
using humanoid_interfaces::msg::SafetyStatus;

namespace
{
// Fallback defaults if config/controllers.yaml doesn't override them --
// CANONICAL_JOINT_ORDER, see g1_description/g1_23dof.urdf.xacro and
// humanoid_interfaces/config/canonical_joint_order.yaml. Gains mirror the
// position_pid values bipedal_nav originally used per joint group (legs
// stiff, knee highest, waist/hip-yaw softer, arms soft, wrist softest) --
// reused here as impedance PD gains instead, per ARCHITECTURE.md's
// "Sim/hardware symmetry" rationale for standardizing on a raw effort
// interface.
const std::vector<std::string> kDefaultJoints = {
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
  "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
  "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint",
};

const std::vector<double> kDefaultKp = {
  1500, 1500, 500, 1500, 1500, 1500,     // left leg (hip_yaw softer)
  1500, 1500, 500, 1500, 1500, 1500,     // right leg
  500,                                   // waist_yaw
  100, 100, 100, 100, 50,                // left arm (wrist_roll softest)
  100, 100, 100, 100, 50,                // right arm
};

const std::vector<double> kDefaultKd = {
  150, 150, 50, 150, 150, 150,
  150, 150, 50, 150, 150, 150,
  50,
  10, 10, 10, 10, 5,
  10, 10, 10, 10, 5,
};

const std::vector<std::string> kImuInterfaceSuffixes = {
  "orientation.x", "orientation.y", "orientation.z", "orientation.w",
  "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
  "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z",
};
constexpr const char * kImuSensorName = "imu_imu";
}  // namespace

controller_interface::CallbackReturn JointImpedanceController::on_init()
{
  auto node = get_node();
  node->declare_parameter<std::vector<std::string>>("joints", kDefaultJoints);
  node->declare_parameter<std::vector<double>>("kp", kDefaultKp);
  node->declare_parameter<std::vector<double>>("kd", kDefaultKd);
  node->declare_parameter<double>("max_effort", 0.0);

  joint_names_ = node->get_parameter("joints").as_string_array();
  kp_ = node->get_parameter("kp").as_double_array();
  kd_ = node->get_parameter("kd").as_double_array();
  max_effort_ = node->get_parameter("max_effort").as_double();

  if (joint_names_.size() != kp_.size() || joint_names_.size() != kd_.size())
  {
    RCLCPP_ERROR(
      node->get_logger(),
      "joints (%zu), kp (%zu), kd (%zu) size mismatch -- fix config/controllers.yaml",
      joint_names_.size(), kp_.size(), kd_.size());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
JointImpedanceController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint : joint_names_)
  {
    config.names.push_back(joint + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointImpedanceController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint : joint_names_)
  {
    config.names.push_back(joint + "/position");
    config.names.push_back(joint + "/velocity");
    config.names.push_back(joint + "/effort");
  }
  // See g1_description/g1_23dof.urdf.xacro's <sensor name="imu_imu"> block --
  // if the active hardware plugin doesn't export these (e.g. an
  // EthercatHardwareInterface build that hasn't wired an IMU yet), activation
  // will fail loudly here rather than silently reading garbage, which is the
  // correct failure mode for a safety-relevant sensor.
  for (const auto & suffix : kImuInterfaceSuffixes)
  {
    config.names.push_back(std::string(kImuSensorName) + "/" + suffix);
  }
  return config;
}

namespace
{
template<typename InterfaceT>
InterfaceT & find_interface(std::vector<InterfaceT> & interfaces, const std::string & full_name)
{
  auto it = std::find_if(
    interfaces.begin(), interfaces.end(),
    [&full_name](const InterfaceT & iface) { return iface.get_name() == full_name; });
  if (it == interfaces.end())
  {
    throw std::runtime_error("JointImpedanceController: interface not found: " + full_name);
  }
  return *it;
}
}  // namespace

controller_interface::CallbackReturn JointImpedanceController::on_configure(
  const rclcpp_lifecycle::State &)
{
  auto node = get_node();

  command_sub_ = node->create_subscription<JointCommand>(
    "/joint_command", rclcpp::SystemDefaultsQoS(),
    [this](const JointCommand::SharedPtr msg) { command_buffer_.writeFromNonRT(msg); });

  safety_sub_ = node->create_subscription<SafetyStatus>(
    "/safety_status", rclcpp::SystemDefaultsQoS(),
    [this](const SafetyStatus::SharedPtr msg) { is_safe_.store(msg->is_safe); });

  state_pub_ = std::make_shared<realtime_tools::RealtimePublisher<RobotState>>(
    node->create_publisher<RobotState>("/robot_state", rclcpp::SystemDefaultsQoS()));

  RCLCPP_INFO(
    node->get_logger(),
    "JointImpedanceController configured for %zu joints, tau = tau_ff + Kp(q_d-q) + Kd(qd_d-qd)",
    joint_names_.size());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn JointImpedanceController::on_activate(
  const rclcpp_lifecycle::State &)
{
  effort_command_.clear();
  position_state_.clear();
  velocity_state_.clear();
  effort_state_.clear();
  imu_state_.clear();

  try
  {
    for (const auto & joint : joint_names_)
    {
      effort_command_.emplace_back(find_interface(command_interfaces_, joint + "/effort"));
      position_state_.emplace_back(find_interface(state_interfaces_, joint + "/position"));
      velocity_state_.emplace_back(find_interface(state_interfaces_, joint + "/velocity"));
      effort_state_.emplace_back(find_interface(state_interfaces_, joint + "/effort"));
    }
    for (const auto & suffix : kImuInterfaceSuffixes)
    {
      imu_state_.emplace_back(
        find_interface(state_interfaces_, std::string(kImuSensorName) + "/" + suffix));
    }
  }
  catch (const std::runtime_error & e)
  {
    RCLCPP_ERROR(get_node()->get_logger(), "on_activate failed: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  // Start "disabled" (is_safe_ defaults true, but with no JointCommand
  // received yet update() holds zero effort -- see update()'s null-check).
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn JointImpedanceController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  effort_command_.clear();
  position_state_.clear();
  velocity_state_.clear();
  effort_state_.clear();
  imu_state_.clear();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type JointImpedanceController::update(
  const rclcpp::Time & time, const rclcpp::Duration & /*period*/)
{
  const bool safe = is_safe_.load();

  auto * command_shared_ptr = command_buffer_.readFromRT();
  const JointCommand * command =
    (command_shared_ptr && *command_shared_ptr) ? command_shared_ptr->get() : nullptr;

  for (size_t i = 0; i < joint_names_.size(); ++i)
  {
    double tau = 0.0;
    if (safe && command != nullptr && i < command->position_target.size())
    {
      const double q = position_state_[i].get().get_value();
      const double qd = velocity_state_[i].get().get_value();
      const double q_d = command->position_target[i];
      const double qd_d = command->velocity_target[i];
      const double tau_ff = command->effort_feedforward[i];
      tau = tau_ff + kp_[i] * (q_d - q) + kd_[i] * (qd_d - qd);
      if (max_effort_ > 0.0)
      {
        tau = std::clamp(tau, -max_effort_, max_effort_);
      }
    }
    // safe == false, or no JointCommand received yet: tau stays 0.0 -- this
    // is the E-stop / not-yet-armed behaviour required by
    // INTEGRATION_POINTS.md "Safety". Per-joint effort limits from the URDF
    // are enforced one layer down by whichever SystemInterface is active
    // (verified in mujoco_system.cpp's effort-command clamp against
    // <limit effort="...">), so this controller does not duplicate that clamp
    // except for the optional global max_effort_ override above.
    effort_command_[i].get().set_value(tau);
  }

  if (state_pub_ && state_pub_->trylock())
  {
    auto & msg = state_pub_->msg_;
    msg.header.stamp = time;
    msg.joint_positions.resize(joint_names_.size());
    msg.joint_velocities.resize(joint_names_.size());
    msg.joint_efforts.resize(joint_names_.size());
    for (size_t i = 0; i < joint_names_.size(); ++i)
    {
      msg.joint_positions[i] = position_state_[i].get().get_value();
      msg.joint_velocities[i] = velocity_state_[i].get().get_value();
      msg.joint_efforts[i] = effort_state_[i].get().get_value();
    }

    // imu_state_ order: orientation x,y,z,w (0-3), angular_velocity x,y,z (4-6),
    // linear_acceleration x,y,z (7-9) -- see kImuInterfaceSuffixes.
    const double qx = imu_state_[0].get().get_value();
    const double qy = imu_state_[1].get().get_value();
    const double qz = imu_state_[2].get().get_value();
    const double qw = imu_state_[3].get().get_value();
    msg.imu_orientation.x = qx;
    msg.imu_orientation.y = qy;
    msg.imu_orientation.z = qz;
    msg.imu_orientation.w = qw;
    msg.imu_angular_velocity.x = imu_state_[4].get().get_value();
    msg.imu_angular_velocity.y = imu_state_[5].get().get_value();
    msg.imu_angular_velocity.z = imu_state_[6].get().get_value();
    msg.imu_linear_acceleration.x = imu_state_[7].get().get_value();
    msg.imu_linear_acceleration.y = imu_state_[8].get().get_value();
    msg.imu_linear_acceleration.z = imu_state_[9].get().get_value();

    // gravity_vector = world gravity (0,0,-9.81) expressed in the IMU/pelvis
    // body frame, i.e. R(q)^T * (0,0,-9.81) with q = (qw,qx,qy,qz). Computed
    // directly (no tf2 dependency) to avoid any allocation in this
    // real-time-called path -- see RESEARCH_NOTES.md section 4.
    constexpr double g = 9.81;
    // R^T * [0,0,-g] = -g * (third ROW of R) = -g * [2(qx*qz+qw*qy), 2(qy*qz-qw*qx), qw^2-qx^2-qy^2+qz^2]
    msg.gravity_vector.x = -g * (2.0 * (qx * qz + qw * qy));
    msg.gravity_vector.y = -g * (2.0 * (qy * qz - qw * qx));
    msg.gravity_vector.z = -g * (qw * qw - qx * qx - qy * qy + qz * qz);

    state_pub_->unlockAndPublish();
  }

  return controller_interface::return_type::OK;
}

}  // namespace lowlevel_control

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  lowlevel_control::JointImpedanceController, controller_interface::ControllerInterface)
