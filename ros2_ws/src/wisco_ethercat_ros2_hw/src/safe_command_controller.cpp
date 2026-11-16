#include "wisco_ethercat_ros2_hw/safe_command_controller.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>

#include "controller_interface/helpers.hpp"

namespace wisco_ethercat_ros2_hw
{

int64_t SafeCommandController::steady_now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

controller_interface::CallbackReturn SafeCommandController::on_init()
{
  try {
    // auto_declare: on Humble the controller node already declares parameters
    // found in the params file, so a plain declare_parameter would throw.
    auto_declare<std::vector<std::string>>("joints", std::vector<std::string>{});
    auto_declare<std::string>("interface_name", "velocity");
    auto_declare<double>("command_timeout", 0.1);
    auto_declare<double>("soft_margin", 0.1);
    auto_declare<std::vector<double>>("max_command", std::vector<double>{});
    auto_declare<std::vector<double>>("position_min", std::vector<double>{});
    auto_declare<std::vector<double>>("position_max", std::vector<double>{});
    auto_declare<std::vector<double>>("damping", std::vector<double>{});
  } catch (const std::exception & e) {
    std::fprintf(stderr, "SafeCommandController: exception in on_init: %s\n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn SafeCommandController::on_configure(
  const rclcpp_lifecycle::State &)
{
  auto node = get_node();
  auto logger = node->get_logger();
  joints_ = node->get_parameter("joints").as_string_array();
  interface_name_ = node->get_parameter("interface_name").as_string();
  const double timeout = node->get_parameter("command_timeout").as_double();
  soft_margin_ = node->get_parameter("soft_margin").as_double();
  const auto max_command = node->get_parameter("max_command").as_double_array();
  const auto position_min = node->get_parameter("position_min").as_double_array();
  const auto position_max = node->get_parameter("position_max").as_double_array();
  const auto damping = node->get_parameter("damping").as_double_array();

  if (joints_.empty()) {
    RCLCPP_ERROR(logger, "'joints' is empty");
    return controller_interface::CallbackReturn::ERROR;
  }
  if (interface_name_ == "velocity") {
    mode_ = safety::Mode::kVelocity;
  } else if (interface_name_ == "effort") {
    mode_ = safety::Mode::kEffort;
  } else {
    RCLCPP_ERROR(
      logger, "interface_name must be 'velocity' or 'effort', got '%s'", interface_name_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  if (!(timeout > 0.0) || !std::isfinite(timeout)) {
    RCLCPP_ERROR(logger, "command_timeout must be a positive number of seconds");
    return controller_interface::CallbackReturn::ERROR;
  }
  const size_t n = joints_.size();
  if (max_command.size() != n || position_min.size() != n || position_max.size() != n ||
    damping.size() != n)
  {
    RCLCPP_ERROR(
      logger,
      "max_command, position_min, position_max and damping need one entry per joint (%zu); "
      "got %zu, %zu, %zu, %zu", n, max_command.size(), position_min.size(), position_max.size(),
      damping.size());
    return controller_interface::CallbackReturn::ERROR;
  }
  limits_.clear();
  for (size_t i = 0; i < n; ++i) {
    if (!(max_command[i] >= 0.0) || !(position_min[i] < position_max[i]) || !(damping[i] >= 0.0)) {
      RCLCPP_ERROR(logger, "bad limits for joint '%s'", joints_[i].c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    limits_.push_back({max_command[i], position_min[i], position_max[i], damping[i]});
  }
  timeout_ns_ = static_cast<int64_t>(timeout * 1e9);

  subscription_ = node->create_subscription<std_msgs::msg::Float64MultiArray>(
    "~/commands", rclcpp::SystemDefaultsQoS(),
    [this, n](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
      if (msg->data.size() != n) {
        RCLCPP_WARN_THROTTLE(
          get_node()->get_logger(), *get_node()->get_clock(), 2000,
          "ignoring command with %zu values; expected %zu (one per joint, in 'joints' order)",
          msg->data.size(), n);
        return;
      }
      auto command = std::make_shared<Command>();
      command->values = msg->data;
      command->received_ns = steady_now_ns();
      command_buffer_.writeFromNonRT(command);
    });

  RCLCPP_INFO(
    logger, "%zu joint(s), %s commands, timeout %.3f s", n, interface_name_.c_str(), timeout);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
SafeCommandController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint : joints_) {
    config.names.push_back(joint + "/" + interface_name_);
  }
  return config;
}

controller_interface::InterfaceConfiguration
SafeCommandController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint : joints_) {
    config.names.push_back(joint + "/position");
  }
  for (const auto & joint : joints_) {
    config.names.push_back(joint + "/velocity");
  }
  return config;
}

controller_interface::CallbackReturn SafeCommandController::on_activate(
  const rclcpp_lifecycle::State &)
{
  commands_.clear();
  positions_.clear();
  velocities_.clear();
  const bool ok =
    controller_interface::get_ordered_interfaces(
    command_interfaces_, joints_, interface_name_, commands_) &&
    controller_interface::get_ordered_interfaces(
    state_interfaces_, joints_, "position", positions_) &&
    controller_interface::get_ordered_interfaces(
    state_interfaces_, joints_, "velocity", velocities_);
  if (!ok) {
    RCLCPP_ERROR(get_node()->get_logger(), "could not get the expected interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }
  // Drop any command received before activation: the joints start held.
  command_buffer_.writeFromNonRT(std::shared_ptr<Command>());
  write_zero();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn SafeCommandController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  write_zero();
  command_buffer_.writeFromNonRT(std::shared_ptr<Command>());
  commands_.clear();
  positions_.clear();
  velocities_.clear();
  return controller_interface::CallbackReturn::SUCCESS;
}

void SafeCommandController::write_zero()
{
  for (auto & command : commands_) {
    command.get().set_value(0.0);
  }
}

controller_interface::return_type SafeCommandController::update(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  const std::shared_ptr<Command> command = *command_buffer_.readFromRT();
  const bool fresh = command && (steady_now_ns() - command->received_ns) <= timeout_ns_;
  for (size_t i = 0; i < commands_.size(); ++i) {
    const double requested = command ? command->values[i] : 0.0;
    const double out = safety::safe_command(
      mode_, requested, fresh, positions_[i].get().get_value(), velocities_[i].get().get_value(),
      limits_[i], soft_margin_);
    commands_[i].get().set_value(out);
  }
  return controller_interface::return_type::OK;
}

}  // namespace wisco_ethercat_ros2_hw

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
  wisco_ethercat_ros2_hw::SafeCommandController, controller_interface::ControllerInterface)
