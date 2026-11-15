#ifndef LOWLEVEL_CONTROL__JOINT_IMPEDANCE_CONTROLLER_HPP_
#define LOWLEVEL_CONTROL__JOINT_IMPEDANCE_CONTROLLER_HPP_

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "hardware_interface/loaned_command_interface.hpp"
#include "hardware_interface/loaned_state_interface.hpp"
#include "realtime_tools/realtime_buffer.h"
#include "realtime_tools/realtime_publisher.h"

#include "humanoid_interfaces/msg/joint_command.hpp"
#include "humanoid_interfaces/msg/robot_state.hpp"
#include "humanoid_interfaces/msg/safety_status.hpp"

namespace lowlevel_control
{

// STATUS: REAL. See lowlevel_control/README.md and RESEARCH_NOTES.md sections
// 2 ("ros2_control SystemInterface architecture"; this is the matching
// controller-side lifecycle) and 4 ("PREEMPT-RT") for why JointCommand and
// SafetyStatus are handed to update() via realtime-safe buffers rather than
// processed directly in their ROS subscription callbacks.
//
// FREQUENCY: whatever controller_manager's update_rate parameter says
// (config/controllers.yaml in this repo sets it to 1000).
//
// tau = tau_ff + Kp*(q_d - q) + Kd*(qd_d - qd), per joint, CANONICAL_JOINT_ORDER.
class JointImpedanceController : public controller_interface::ControllerInterface
{
public:
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  std::vector<std::string> joint_names_;
  std::vector<double> kp_;
  std::vector<double> kd_;
  // Per-joint torque limit [Nm], index-aligned with joint_names_; a value <= 0 disables the
  // clamp for that joint. Enforced HERE, not by the hardware plugin: the MuJoCo bridge's
  // URDF-limit clamp is dead code (has_effort_limits is never set), so without this the
  // sim applies unbounded torque. Values come from the URDF <limit effort="...">.
  std::vector<double> effort_limits_;
  double max_effort_{0.0};  // optional extra global clamp on top of effort_limits_; 0.0 = off

  // Cached pointers into exported command/state interfaces, indexed same as joint_names_.
  std::vector<std::reference_wrapper<hardware_interface::LoanedCommandInterface>> effort_command_;
  std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>> position_state_;
  std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>> velocity_state_;
  std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>> effort_state_;

  // IMU state interfaces (sensor "imu_imu" in g1_description's <ros2_control>
  // block: see that file's comments for the exact naming convention this
  // depends on). Order: orientation x,y,z,w; angular_velocity x,y,z;
  // linear_acceleration x,y,z.
  std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>> imu_state_;

  realtime_tools::RealtimeBuffer<std::shared_ptr<humanoid_interfaces::msg::JointCommand>>
    command_buffer_;
  rclcpp::Subscription<humanoid_interfaces::msg::JointCommand>::SharedPtr command_sub_;

  std::atomic<bool> is_safe_{true};
  rclcpp::Subscription<humanoid_interfaces::msg::SafetyStatus>::SharedPtr safety_sub_;

  std::shared_ptr<realtime_tools::RealtimePublisher<humanoid_interfaces::msg::RobotState>>
    state_pub_;
};

}  // namespace lowlevel_control

#endif  // LOWLEVEL_CONTROL__JOINT_IMPEDANCE_CONTROLLER_HPP_
