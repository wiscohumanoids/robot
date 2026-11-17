// ChainedJointStateBroadcaster: see include/wisco_ethercat_ros2_hw/joint_state_broadcaster.hpp.
#include "wisco_ethercat_ros2_hw/joint_state_broadcaster.hpp"

#include <algorithm>
#include <limits>

#include "pluginlib/class_list_macros.hpp"

namespace wisco_ethercat_ros2_hw
{

using controller_interface::CallbackReturn;
using controller_interface::InterfaceConfiguration;
using controller_interface::interface_configuration_type;

namespace
{
constexpr const char * kInterfaces[] = {"position", "velocity", "effort"};
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
}  // namespace

CallbackReturn ChainedJointStateBroadcaster::on_init()
{
  auto_declare<std::vector<std::string>>("joints", {});
  auto_declare<std::vector<std::string>>("chained_joints", {});
  auto_declare<std::string>("chain_prefix", "parallel_joints");
  auto_declare<std::string>("topic", "/joint_states");
  return CallbackReturn::SUCCESS;
}

InterfaceConfiguration ChainedJointStateBroadcaster::command_interface_configuration() const
{
  return {interface_configuration_type::NONE, {}};
}

InterfaceConfiguration ChainedJointStateBroadcaster::state_interface_configuration() const
{
  InterfaceConfiguration cfg{interface_configuration_type::INDIVIDUAL, {}};
  for (const auto & s : sources_) {
    for (const char * i : kInterfaces) {cfg.names.push_back(s + "/" + i);}
  }
  return cfg;
}

CallbackReturn ChainedJointStateBroadcaster::on_configure(const rclcpp_lifecycle::State &)
{
  auto node = get_node();
  joints_ = node->get_parameter("joints").as_string_array();
  const auto chained = node->get_parameter("chained_joints").as_string_array();
  const auto prefix = node->get_parameter("chain_prefix").as_string();
  if (joints_.empty()) {
    RCLCPP_ERROR(node->get_logger(), "joints: none given");
    return CallbackReturn::ERROR;
  }
  for (const auto & c : chained) {
    if (std::find(joints_.begin(), joints_.end(), c) == joints_.end()) {
      RCLCPP_ERROR(node->get_logger(), "chained joint %s is not in joints", c.c_str());
      return CallbackReturn::ERROR;
    }
  }
  sources_.clear();
  for (const auto & j : joints_) {
    const bool is_chained = std::find(chained.begin(), chained.end(), j) != chained.end();
    sources_.push_back(is_chained ? prefix + "/" + j : j);
  }
  publisher_ = node->create_publisher<sensor_msgs::msg::JointState>(
    node->get_parameter("topic").as_string(), rclcpp::SystemDefaultsQoS());
  realtime_ = std::make_shared<realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>>(publisher_);
  auto & msg = realtime_->msg_;
  msg.name = joints_;
  msg.position.assign(joints_.size(), 0.0);
  msg.velocity.assign(joints_.size(), 0.0);
  msg.effort.assign(joints_.size(), 0.0);
  RCLCPP_INFO(node->get_logger(), "%zu joints (%zu through %s) on %s", joints_.size(), chained.size(),
    prefix.c_str(), publisher_->get_topic_name());
  return CallbackReturn::SUCCESS;
}

controller_interface::return_type ChainedJointStateBroadcaster::update(
  const rclcpp::Time & time, const rclcpp::Duration &)
{
  if (realtime_ && realtime_->trylock()) {
    auto & msg = realtime_->msg_;
    msg.header.stamp = time;
    for (std::size_t i = 0; i < joints_.size(); ++i) {
      msg.position[i] = state_interfaces_[3 * i].get_optional().value_or(kNaN);
      msg.velocity[i] = state_interfaces_[3 * i + 1].get_optional().value_or(kNaN);
      msg.effort[i] = state_interfaces_[3 * i + 2].get_optional().value_or(kNaN);
    }
    realtime_->unlockAndPublish();
  }
  return controller_interface::return_type::OK;
}

}  // namespace wisco_ethercat_ros2_hw

PLUGINLIB_EXPORT_CLASS(wisco_ethercat_ros2_hw::ChainedJointStateBroadcaster, controller_interface::ControllerInterface)
