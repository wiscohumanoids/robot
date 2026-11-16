// Copyright 2026 WiscoHumanoids
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace wisco
{

/// Minimal ROS 2 API node that awaits torque commands.
///
/// Subscribes to /effort_controller/commands (std_msgs/Float64MultiArray) and
/// holds the most recently received torque commands. Intended as the command
/// surface for the eventual Cyclic Synchronous Torque (CST) implementation;
/// the hardware is currently configured for Cyclic Synchronous Velocity (CSV).
class ExampleRosNode : public rclcpp::Node
{
public:
  explicit ExampleRosNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("example_ros_node", options)
  {
    torque_command_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/effort_controller/commands", 10,
      std::bind(&ExampleRosNode::torque_command_callback, this, std::placeholders::_1));

    // Heartbeat timer: periodically re-log the latest awaited torque commands
    // so the API state is observable.
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&ExampleRosNode::print_latest_torque_commands, this));

    RCLCPP_INFO(
      this->get_logger(),
      "example_ros_node API initialized. Awaiting torque commands on /effort_controller/commands.");
  }

  /// Return the most recently received torque commands (the API surface).
  const std::vector<double> & get_torque_commands() const { return latest_torque_commands_; }

private:
  void torque_command_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    latest_torque_commands_.assign(msg->data.begin(), msg->data.end());
    RCLCPP_INFO(
      this->get_logger(), "Received torque commands: [%s]",
      vector_to_str(latest_torque_commands_).c_str());
  }

  void print_latest_torque_commands()
  {
    RCLCPP_INFO(
      this->get_logger(), "Awaited torque commands: [%s]",
      vector_to_str(latest_torque_commands_).c_str());
  }

  std::string vector_to_str(const std::vector<double> & v)
  {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
      out += std::to_string(v[i]);
      if (i + 1 < v.size()) {
        out += ", ";
      }
    }
    return out.empty() ? "none (awaiting first command)" : out;
  }

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr torque_command_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::vector<double> latest_torque_commands_;
};

}  // namespace wisco

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<wisco::ExampleRosNode>());
  rclcpp::shutdown();
  return 0;
}
