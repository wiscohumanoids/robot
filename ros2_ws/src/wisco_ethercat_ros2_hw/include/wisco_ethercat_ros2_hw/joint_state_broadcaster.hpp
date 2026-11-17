// ChainedJointStateBroadcaster: /joint_states in the robot's URDF joints when some of them are not hardware joints.
//
// The upstream joint_state_broadcaster publishes the hardware's joints. On Wisco the ankles and the waist's roll and
// pitch are two motors each behind push rods: the hardware (the EtherCAT drives, or the simulated robot) has
// left_ankle_motor_a / _b, ... and the serial joints the policy, leg odometry and every other /joint_states reader
// use (left_ankle_pitch, left_ankle_roll, ...) exist only as ParallelLinkageController's exported states. This
// broadcaster reads the joints from wherever they live and publishes them in one message, in `joints` order:
//
//   joints           the published joints, in message order
//   chained_joints   those read from the chain controller: `<chain_prefix>/<joint>/position` (velocity, effort)
//                    instead of `<joint>/position` (same names as JointTargetController's)
//   chain_prefix     the chain controller's name (default parallel_joints)
//   topic            default /joint_states
//
// One complete message per update (the controller manager rate, or this controller's update_rate), stamped with
// the controller manager's time. Load it after the chain controller (its states must be exported).
#ifndef WISCO_ETHERCAT_ROS2_HW__JOINT_STATE_BROADCASTER_HPP_
#define WISCO_ETHERCAT_ROS2_HW__JOINT_STATE_BROADCASTER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace wisco_ethercat_ros2_hw
{

class ChainedJointStateBroadcaster : public controller_interface::ControllerInterface
{
public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::return_type update(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  std::vector<std::string> joints_, sources_;     // published names; the interface prefix each is read from
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>> realtime_;
};

}  // namespace wisco_ethercat_ros2_hw

#endif  // WISCO_ETHERCAT_ROS2_HW__JOINT_STATE_BROADCASTER_HPP_
