// ArmController: the arm stack's RT loop (spec Mode P / Mode T) as a
// ros2_control controller. Runs arm_law.hpp every cycle with Pinocchio
// gravity from the robot description plus the calibration file.
//
// Topics (Float64MultiArray; layouts are mirrored in arm_stack/hw/ros_client.py):
//   ~/command (in)   [mode (0 P, 1 T), flags (1 gravity_ff, 2 ignore_soft_limits),
//                     friction_ff_gain, q[n], qd[n], kp[n], kd[n], tau_ff[n],
//                     torque_clamp[n]]
//                    NaN = not set. A command older than watchdog_s (50 ms)
//                    drops the arm into gravity comp + damping in CST.
//                    Optionally followed by a payload block [kg, x, y, z]: a
//                    point mass fixed in payload_frame (parameter, default
//                    tool0), added to g(q) from then on (Mode P, Mode T and the
//                    watchdog fallback) until a command carries another one;
//                    kg <= 0 clears it, NaN kg leaves it. Commands without the
//                    block (the original length) leave it as it is. Reset to
//                    none on activation.
//   ~/state (out)    [t (s, steady), mode, fallback, limit_hit, clamped,
//                     q[n], qd[n], tau_measured[n], tau_out[n], q_target[n]]
//                    every cycle, through a realtime publisher (try-lock; the
//                    non-RT thread publishes, the loop never blocks).
//
// Interfaces per joint: command position, effort, torque_offset,
// mode_of_operation; state position, velocity, effort.
//
// drive_feedback (parameter, default false) also claims the state interfaces
// status_word (0x6041) and mode_of_operation (0x6061 display) per joint:
//   * while any drive is not OPERATION_ENABLED (fault, quick stop, still
//     enabling) the arm runs the fallback (gravity comp + damping, CST) and
//     the disabled joint's CSP target tracks its actual position, so
//     re-enabling never steps the drive to a stale target;
//   * until the drive reports a requested CSP <-> CST switch in 0x6061, the
//     torque goes in 0x60B2 (added in both modes per CiA402), so a drive that
//     switches a few cycles late never runs a cycle without it, and the CSP
//     ease-in waits for the drive to be in CSP.
// It needs the drive's ros2_control joint to export those state interfaces
// (0x6041 / 0x6061 with state_interface: in the slave config); tested against
// test/mock/ (test_arm_controller_mock.cpp).
//
// CSP re-entry (always): the target eases in from the actual position over
// blend_s (arm::ease_in) before the max_velocity rate limit, so a drive that
// differentiates 0x607A for velocity feedforward sees no velocity step.
// Blends and eases count control periods (update()'s period), not wall time.
#ifndef WISCO_ETHERCAT_ROS2_HW__ARM_CONTROLLER_HPP_
#define WISCO_ETHERCAT_ROS2_HW__ARM_CONTROLLER_HPP_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "wisco_ethercat_ros2_hw/arm_law.hpp"

namespace wisco_ethercat_ros2_hw
{

class GravityModel;

constexpr size_t kCommandHeader = 3;
constexpr size_t kCommandBlocks = 6;
constexpr size_t kPayloadBlock = 4;   // optional [kg, x, y, z] after the per-joint blocks
constexpr size_t kStateHeader = 5;
constexpr size_t kStateBlocks = 5;

// Smallest 0x6072 the controller writes, as a fraction of the joint's
// configured max torque (0x6072 = 0 means the stall-current limit).
constexpr double kMinTorqueCapFraction = 0.01;

class ArmController : public controller_interface::ControllerInterface
{
public:
  ArmController();
  ~ArmController() override;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State &) override;
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override;
  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  struct Command
  {
    arm::LawCommand law;
    int64_t received_ns = 0;
    bool has_payload = false;          // the optional payload block was sent
    double payload[kPayloadBlock] = {0.0, 0.0, 0.0, 0.0};
  };
  using Ref = std::reference_wrapper<hardware_interface::LoanedCommandInterface>;
  using StateRef = std::reference_wrapper<hardware_interface::LoanedStateInterface>;

  static int64_t steady_now_ns();

  std::vector<std::string> joints_;
  arm::LawConfig cfg_;
  std::unique_ptr<GravityModel> gravity_;
  double blend_s_ = 0.02;

  realtime_tools::RealtimeBuffer<std::shared_ptr<Command>> command_buffer_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
  std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Float64MultiArray>> state_pub_;
  std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>> rt_state_;

  std::vector<Ref> cmd_pos_, cmd_eff_, cmd_off_, cmd_mode_, cmd_max_;
  std::vector<StateRef> st_pos_, st_vel_, st_eff_, st_status_, st_mode_;
  bool drive_feedback_ = false;
  // write_max_torque (parameter, default false): also claims <joint>/max_torque
  // (0x6072) and writes the command's torque clamp there, so the drive's CSP
  // loop cannot exceed it. Needs 0x6072 mapped as a command interface.
  bool write_max_torque_ = false;
  bool payload_ok_ = false;                         // payload_frame found in the model
  double payload_[kPayloadBlock] = {0.0, 0.0, 0.0, 0.0};   // applied now (RT thread only)

  // preallocated per-cycle buffers
  std::vector<double> q_, qd_, tau_meas_, g_, tau_switch_, q_target_prev_;
  std::vector<double> q_switch_;     // per joint: where the CSP target eases in from
  std::vector<double> since_ease_;  // per joint: s of control cycles since that ease started
  std::vector<bool> enabled_;
  arm::LawOutput out_;
  int last_mode_ = -1;
  double since_switch_ = 0.0;        // s of control cycles since the last P <-> T switch
};

}  // namespace wisco_ethercat_ros2_hw

#endif  // WISCO_ETHERCAT_ROS2_HW__ARM_CONTROLLER_HPP_
