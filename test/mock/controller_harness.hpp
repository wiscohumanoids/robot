// Runs a real ros2_control controller (ArmController, SafeCommandController)
// against MockEthercatSystem, one read -> update -> write cycle at a time,
// the order controller_manager uses. Test-only.
//
// The controller goes through its normal lifecycle (init -> configure ->
// activate) and gets loaned interfaces from the system's exported ones, as
// the resource manager would hand them out. Commands on the controller's
// topic are delivered by calling the subscription's handle_message() (the
// entry point the executor uses), so the real callback and its checks run,
// without DDS timing in the test.
#ifndef WISCO_ETHERCAT_ROS2_HW__MOCK__CONTROLLER_HARNESS_HPP_
#define WISCO_ETHERCAT_ROS2_HW__MOCK__CONTROLLER_HARNESS_HPP_

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "hardware_interface/loaned_command_interface.hpp"
#include "hardware_interface/loaned_state_interface.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "mock_ethercat_system.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace wisco_ethercat_ros2_hw
{
namespace mock
{

struct JointSpec
{
  std::string name;
  double effort_nm_per_permille = 0.015;
  double peak_effort_nm = 10.0;
  int max_torque_permille = 1000;   // 0x6072
  double inertia = 0.05;
  double initial_position = 0.0;
  int direction = 1;
  double zero_offset = 0.0;
  double counts_per_rev = 131072.0;
};

class ControllerHarness
{
public:
  explicit ControllerHarness(
    const std::vector<JointSpec> & joints,
    const std::unordered_map<std::string, std::string> & hw_params = {})
  {
    hardware_interface::HardwareInfo info;
    info.name = "EtherCATSystem";
    info.type = "system";
    info.hardware_plugin_name = "mock/MockEthercatSystem";
    info.hardware_parameters = hw_params;
    for (const auto & j : joints) {
      hardware_interface::ComponentInfo c;
      c.name = j.name;
      c.type = "joint";
      c.parameters["effort_nm_per_permille"] = std::to_string(j.effort_nm_per_permille);
      c.parameters["peak_effort_nm"] = std::to_string(j.peak_effort_nm);
      c.parameters["max_torque_permille"] = std::to_string(j.max_torque_permille);
      c.parameters["inertia"] = std::to_string(j.inertia);
      c.parameters["initial_position"] = std::to_string(j.initial_position);
      c.parameters["direction"] = std::to_string(j.direction);
      c.parameters["zero_offset"] = std::to_string(j.zero_offset);
      c.parameters["counts_per_rev"] = std::to_string(j.counts_per_rev);
      info.joints.push_back(c);
    }
    if (hw_.on_init(info) != hardware_interface::CallbackReturn::SUCCESS) {
      throw std::runtime_error("MockEthercatSystem::on_init failed");
    }
    state_ifaces_ = hw_.export_state_interfaces();
    command_ifaces_ = hw_.export_command_interfaces();
  }

  MockEthercatSystem & hw() {return hw_;}

  // Run the bus without a controller (drives enable under the master logic).
  void run_bus(int cycles)
  {
    for (int k = 0; k < cycles; ++k) {
      hw_.read(now(), period());
      hw_.write(now(), period());
      t_ns_ += period_ns_;
    }
  }

  // init + configure the controller, wire its interfaces, activate.
  // Returns false if any lifecycle step fails.
  bool load(
    std::shared_ptr<controller_interface::ControllerInterface> controller,
    const std::string & name, const std::vector<rclcpp::Parameter> & params,
    const std::string & command_topic_suffix)
  {
    controller_ = controller;
    auto options = rclcpp::NodeOptions()
      .parameter_overrides(params)
      .allow_undeclared_parameters(true)
      .automatically_declare_parameters_from_overrides(true);
    controller_interface::ControllerInterfaceParams ci_params;
    ci_params.controller_name = name;
    ci_params.robot_description = "";
    ci_params.update_rate = 1000;
    ci_params.controller_manager_update_rate = 1000;
    ci_params.node_namespace = "";
    ci_params.node_options = options;
    if (controller_->init(ci_params) != controller_interface::return_type::OK) {
      return false;
    }
    if (controller_->configure().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE) {
      return false;
    }
    std::vector<hardware_interface::LoanedCommandInterface> cmds;
    for (const auto & want : controller_->command_interface_configuration().names) {
      auto * ci = find(command_ifaces_, want);
      if (!ci) {return false;}
      cmds.emplace_back(*ci);
    }
    std::vector<hardware_interface::LoanedStateInterface> states;
    for (const auto & want : controller_->state_interface_configuration().names) {
      auto * si = find(state_ifaces_, want);
      if (!si) {return false;}
      states.emplace_back(*si);
    }
    controller_->assign_interfaces(std::move(cmds), std::move(states));
    if (controller_->get_node()->activate().id() !=
      lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE)
    {
      return false;
    }
    subscription_ = find_subscription(command_topic_suffix);
    next_tick_ = std::chrono::steady_clock::now();
    return subscription_ != nullptr;
  }

  // Deliver one message to the controller's command subscription.
  void send(const std::vector<double> & data)
  {
    auto msg = std::make_shared<std_msgs::msg::Float64MultiArray>();
    msg->data = data;
    std::shared_ptr<void> erased = msg;
    rclcpp::MessageInfo info;
    subscription_->handle_message(erased, info);
  }

  // Keep re-sending `data` every `every` cycles (a streaming client), or stop.
  void stream(const std::vector<double> & data, int every = 5)
  {
    stream_ = data;
    stream_every_ = every;
    send(data);
  }
  void stop_stream() {stream_.clear();}

  // read -> update -> write, n times. Paced at the 1 kHz period in wall time
  // (the controllers time their command watchdogs on the steady clock),
  // unless set_realtime(false).
  void cycle(int n = 1)
  {
    for (int k = 0; k < n; ++k) {
      if (realtime_) {
        const auto t = std::chrono::steady_clock::now();
        next_tick_ += std::chrono::nanoseconds(period_ns_);
        if (next_tick_ < t - std::chrono::milliseconds(20)) {next_tick_ = t;}   // fell behind
        std::this_thread::sleep_until(next_tick_);
      }
      if (!stream_.empty() && (++stream_count_ % stream_every_) == 0) {send(stream_);}
      hw_.read(now(), period());
      if (controller_->update(now(), period()) != controller_interface::return_type::OK) {
        throw std::runtime_error("controller update failed");
      }
      hw_.write(now(), period());
      t_ns_ += period_ns_;
    }
  }

  void deactivate()
  {
    controller_->get_node()->deactivate();
    controller_->release_interfaces();
  }

  void set_realtime(bool on) {realtime_ = on;}
  // Sleep in wall time while the bus keeps running (a pause with the loop alive).
  void wait_ms(int ms) {cycle(ms);}

  rclcpp::Time now() const {return rclcpp::Time(t_ns_, RCL_STEADY_TIME);}
  rclcpp::Duration period() const {return rclcpp::Duration::from_nanoseconds(period_ns_);}

private:
  template<typename T>
  static T * find(std::vector<T> & v, const std::string & full_name)
  {
    for (auto & i : v) {
      if (i.get_name() == full_name) {return &i;}
    }
    return nullptr;
  }

  rclcpp::SubscriptionBase::SharedPtr find_subscription(const std::string & suffix)
  {
    rclcpp::SubscriptionBase::SharedPtr found;
    controller_->get_node()->get_node_base_interface()->for_each_callback_group(
      [&](rclcpp::CallbackGroup::SharedPtr group) {
        auto s = group->find_subscription_ptrs_if(
          [&](const rclcpp::SubscriptionBase::SharedPtr & sub) {
            const std::string topic = sub->get_topic_name();
            return topic.size() >= suffix.size() &&
            topic.compare(topic.size() - suffix.size(), suffix.size(), suffix) == 0;
          });
        if (s) {found = s;}
      });
    return found;
  }

  MockEthercatSystem hw_;
  std::vector<hardware_interface::StateInterface> state_ifaces_;
  std::vector<hardware_interface::CommandInterface> command_ifaces_;
  std::shared_ptr<controller_interface::ControllerInterface> controller_;
  rclcpp::SubscriptionBase::SharedPtr subscription_;
  std::vector<double> stream_;
  int stream_every_ = 5;
  int stream_count_ = 0;
  int64_t t_ns_ = 1000000000;
  int64_t period_ns_ = 1000000;   // 1 kHz, the arm's update_rate
  bool realtime_ = true;
  std::chrono::steady_clock::time_point next_tick_ = std::chrono::steady_clock::now();
};

}  // namespace mock
}  // namespace wisco_ethercat_ros2_hw

#endif  // WISCO_ETHERCAT_ROS2_HW__MOCK__CONTROLLER_HARNESS_HPP_
