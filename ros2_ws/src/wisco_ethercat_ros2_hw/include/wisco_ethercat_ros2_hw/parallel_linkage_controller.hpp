// ParallelLinkageController: the robot's "PR mode" for its parallel two-motor joints (ankles, waist).
//
// The walking policy and the controllers above it work in the URDF's serial joints (left_ankle_pitch,
// left_ankle_roll, ...), exactly as in simulation. The hardware has two motors per mechanism, driving it through
// push rods. This chainable controller sits between them:
//
//   upstream controller ──references──► ParallelLinkageController ──commands──► motor A, motor B (EtherCAT drives)
//      (joint space)     ◄───states────   (parallel_linkage.hpp)    ◄─states──
//
//   exports, per joint:  references  <controller>/<joint>/position, velocity, effort
//                        states      <controller>/<joint>/position, velocity, effort
//   claims, per motor:   commands    position [, velocity, effort, kp, kd] (pd_in_drive) or effort (torque)
//                        states      position, velocity [, effort]
//
// Command modes (`command_mode`):
//   pd_in_drive  motor targets through the linkage, joint velocity and feedforward torque through J, and per-motor
//                PD gains from the joint gains (the diagonal of J^-T K J^-1): what MyActuator mode 5 (position + kp +
//                kd per cycle) runs. The diagonal drops a coupling term; the controller reports its size.
//   torque       the joint-space PD computed here every cycle and sent as motor torques (exact; needs a fast,
//                steady bus: see infra/rk3588)
//
// Motor angles: each motor's position, as the hardware reports it after calibration, must be 0 with the joint at its
// zero pose (foot flat, torso upright) and grow in `motor_directions` times its crank axis in linkage.json. The CAD
// crank offsets at the zero pose are added here.
//
// Joint limits (`<mechanism>.position_min / position_max`, the URDF limits): references are clamped into them, and a
// solved pose more than `limit_margin` outside them is treated as a bad reading (the rods admit other, physically
// impossible configurations; a wrong calibration or direction lands there).
//
// Safety: when the joint angles cannot be solved (a motor reading the rods cannot produce, or a pose outside the
// limits) or a target is out of reach, pd_in_drive holds the motors where they are (position = measured, no
// feedforward) and torque mode sends zero torque; the failure is counted and logged. With no upstream (not chained), ~/commands (std_msgs/Float64MultiArray, joint positions in the order of
// `mechanisms`, two per mechanism) sets the position references; until a command arrives the joints hold.
#ifndef WISCO_ETHERCAT_ROS2_HW__PARALLEL_LINKAGE_CONTROLLER_HPP_
#define WISCO_ETHERCAT_ROS2_HW__PARALLEL_LINKAGE_CONTROLLER_HPP_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "controller_interface/chainable_controller_interface.hpp"
#include "hardware_interface/loaned_command_interface.hpp"
#include "hardware_interface/loaned_state_interface.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "wisco_ethercat_ros2_hw/parallel_linkage.hpp"

namespace wisco_ethercat_ros2_hw
{

class ParallelLinkageController : public controller_interface::ChainableControllerInterface
{
public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

  // counters, for tests and diagnostics
  uint64_t solve_failures() const { return solve_failures_; }
  uint64_t reach_failures() const { return reach_failures_; }
  double max_stiffness_coupling() const { return max_coupling_; }

protected:
  std::vector<hardware_interface::StateInterface::SharedPtr> on_export_state_interfaces_list() override;
  std::vector<hardware_interface::CommandInterface::SharedPtr> on_export_reference_interfaces_list() override;
  bool on_set_chained_mode(bool chained_mode) override;
  controller_interface::return_type update_reference_from_subscribers(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  controller_interface::return_type update_and_write_commands(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  enum class Mode { kPdInDrive, kTorque };
  enum Ref { kPos = 0, kVel = 1, kEff = 2, kRefs = 3 };     // reference / exported-state layout per joint

  struct Unit                                            // one mechanism: two joints, two motors
  {
    std::string name;
    std::array<std::string, 2> joints, motors;
    std::array<double, 2> direction{1.0, 1.0};
    linkage::Vec2 offset{};                              // crank angle at the joint zero pose (CAD)
    linkage::Vec2 kp{}, kd{};                            // joint-space gains
    linkage::Vec2 lower{}, upper{};                      // joint limits
    linkage::Mechanism mech;
    linkage::Vec2 q_prev{};                              // last solved joint angles (Newton warm start)
    linkage::Vec2 hold_motor{};                          // last good motor targets (hardware frame)
    bool solved{false};
  };

  bool read_unit(std::size_t u, linkage::MotorState2 & crank, linkage::JointState2 & joint);
  void write_hold(std::size_t u, const linkage::Vec2 & motor_hw);
  std::size_t cmd_index(std::size_t u, int motor, const std::string & iface) const;

  std::vector<Unit> units_;
  Mode mode_ = Mode::kPdInDrive;
  std::vector<std::string> motor_cmd_ifaces_;            // per motor, in this order
  bool motor_effort_state_ = true;
  double limit_margin_ = 0.05;
  std::vector<double> exported_state_values_;            // per joint: position, velocity, effort

  realtime_tools::RealtimeBuffer<std::shared_ptr<std::vector<double>>> command_buffer_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;

  uint64_t solve_failures_ = 0, reach_failures_ = 0;
  double max_coupling_ = 0.0;
};

}  // namespace wisco_ethercat_ros2_hw

#endif  // WISCO_ETHERCAT_ROS2_HW__PARALLEL_LINKAGE_CONTROLLER_HPP_
