// JointTargetController: receives a policy's joint targets (sensor_msgs/JointState by name, e.g. the walking
// policy's /locomotion/joint_targets from the central computer) and hands them to the drives in the real-time loop,
// with the safety the topic alone does not give:
//
//   HOLD   on activation, hold the measured pose until targets arrive
//   RAMP   blend linearly from the measured pose to the (moving) targets over `ramp_time` (Unitree's deployment does
//          the same over 2 s before its policy runs)
//   TRACK  pass the targets through, clamped to the joint limits (URDF, minus `limit_margin`), with each joint's kp and
//          kd: the trained policy's, from its manifest.yaml, so the drives run the PD the policy was trained with
//          (joints the policy does not train, like Wisco's head, take `gain_override_*`)
//   SAFE   no new message for `timeout` of control-loop time (central computer gone, DDS stall): `damp` (kp 0, kd `damping_kd`, Unitree's
//          exit command) or `hold` (stay at the measured pose); when messages return, RAMP again
//
// Joints listed in `chained_joints` are the two-motor ones (ankles, waist roll/pitch): their position goes to
// `<chain_prefix>/<joint>/position`, exported by ParallelLinkageController, which owns their gains; they hold
// (never damp) in SAFE. Other joints get `<joint>/position` and, if `gain_interfaces` names them (MyActuator mode 5),
// `<joint>/kp`, `<joint>/kd`.
//
// Service ~/hold (std_srvs/Trigger): back to HOLD at the measured pose, ignoring messages received before the
// request, then RAMP again from there when new ones arrive. For after the robot is moved by something else (the
// simulator's world reset, an operator), so the controller never pulls it back to the pose it held before.
//
// One instance per policy output: locomotion → legs, waist, head; a manipulation policy's targets would get their own
// instance (or the ArmController) for the arms. Joints a message leaves out keep their last target.
#ifndef WISCO_ETHERCAT_ROS2_HW__JOINT_TARGET_CONTROLLER_HPP_
#define WISCO_ETHERCAT_ROS2_HW__JOINT_TARGET_CONTROLLER_HPP_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace wisco_ethercat_ros2_hw
{

class JointTargetController : public controller_interface::ControllerInterface
{
public:
  enum class Phase { kHold, kRamp, kTrack, kSafe };

  controller_interface::CallbackReturn on_init() override;
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::return_type update(const rclcpp::Time & time, const rclcpp::Duration & period) override;

  Phase phase() const { return phase_; }
  uint64_t clamped() const { return clamped_; }
  static const char * phase_name(Phase p);

private:
  struct Targets
  {
    std::vector<double> position;   // per controller joint; NaN = not in the message
    int64_t received_ns = 0;
  };
  struct Joint
  {
    std::string name;
    bool chained = false;
    double kp = 0.0, kd = 0.0, lower = 0.0, upper = 0.0;
    double target = 0.0;            // last commanded target (policy frame, clamped)
    double ramp_from = 0.0;         // measured position when the ramp began
    std::size_t cmd_pos = 0, cmd_kp = SIZE_MAX, cmd_kd = SIZE_MAX, state_pos = 0;
  };

  static int64_t steady_now_ns();
  std::string position_name(const Joint & j) const;
  void hold_measured();
  void write(std::size_t i, double position, double kp, double kd);
  void publish_status();

  std::vector<Joint> joints_;
  std::string chain_prefix_ = "parallel_joints";
  std::vector<std::string> gain_ifaces_;
  int64_t timeout_ns_ = 50000000;
  double ramp_time_ = 2.0, damping_kd_ = 8.0;
  bool damp_on_timeout_ = true;

  Phase phase_ = Phase::kHold;
  double ramp_elapsed_ = 0.0;
  int64_t last_seen_ns_ = 0;        // receive time of the newest message the loop has taken
  int64_t hold_floor_ns_ = 0;       // messages received before the last ~/hold request do not count
  double since_fresh_ = 0.0;        // control-loop time since the newest message, s (update periods)
  uint64_t clamped_ = 0, cycles_ = 0;

  realtime_tools::RealtimeBuffer<std::shared_ptr<Targets>> buffer_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr hold_service_;
  std::atomic<int64_t> hold_request_ns_{0};   // set by ~/hold, taken by the real-time loop
  std::shared_ptr<realtime_tools::RealtimePublisher<diagnostic_msgs::msg::DiagnosticArray>> status_;
};

}  // namespace wisco_ethercat_ros2_hw

#endif  // WISCO_ETHERCAT_ROS2_HW__JOINT_TARGET_CONTROLLER_HPP_
