// Forwards Float64MultiArray commands to the joints, with the checks the
// stock forward controllers lack: a command timeout, per-joint magnitude
// limits, soft position limits, NaN rejection and a stop on deactivate.
#ifndef WISCO_ETHERCAT_ROS2_HW__SAFE_COMMAND_CONTROLLER_HPP_
#define WISCO_ETHERCAT_ROS2_HW__SAFE_COMMAND_CONTROLLER_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "wisco_ethercat_ros2_hw/safety_logic.hpp"

namespace wisco_ethercat_ros2_hw
{

class SafeCommandController : public controller_interface::ControllerInterface
{
public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  struct Command
  {
    std::vector<double> values;
    // Steady clock, so sim time or a wall-clock jump cannot keep a command fresh.
    int64_t received_ns = 0;
  };

  static int64_t steady_now_ns();
  void write_zero();

  std::vector<std::string> joints_;
  std::string interface_name_;
  safety::Mode mode_ = safety::Mode::kVelocity;
  std::vector<safety::JointLimits> limits_;
  double soft_margin_ = 0.1;
  int64_t timeout_ns_ = 100000000;

  realtime_tools::RealtimeBuffer<std::shared_ptr<Command>> command_buffer_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;

  std::vector<std::reference_wrapper<hardware_interface::LoanedCommandInterface>> commands_;
  std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>> positions_;
  std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>> velocities_;
};

}  // namespace wisco_ethercat_ros2_hw

#endif  // WISCO_ETHERCAT_ROS2_HW__SAFE_COMMAND_CONTROLLER_HPP_
