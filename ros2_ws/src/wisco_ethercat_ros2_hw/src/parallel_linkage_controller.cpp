// ParallelLinkageController: see include/wisco_ethercat_ros2_hw/parallel_linkage_controller.hpp.
#include "wisco_ethercat_ros2_hw/parallel_linkage_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "pluginlib/class_list_macros.hpp"
#include "wisco_ethercat_ros2_hw/parallel_linkage_yaml.hpp"

namespace wisco_ethercat_ros2_hw
{

using controller_interface::CallbackReturn;
using controller_interface::InterfaceConfiguration;
using controller_interface::interface_configuration_type;
using controller_interface::return_type;

namespace
{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
const char * kRefNames[] = {"position", "velocity", "effort"};
double finite_or(double v, double fallback) { return std::isfinite(v) ? v : fallback; }
}  // namespace

CallbackReturn ParallelLinkageController::on_init()
{
  try {
    auto_declare<std::string>("linkage_file", "");
    auto_declare<std::vector<std::string>>("mechanisms", std::vector<std::string>{});
    auto_declare<std::string>("command_mode", "pd_in_drive");
    auto_declare<std::vector<std::string>>(
      "motor_command_interfaces", std::vector<std::string>{"position", "velocity", "effort"});
    auto_declare<bool>("motor_effort_state", true);
    auto_declare<double>("limit_margin", 0.05);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "parameter declaration failed: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn ParallelLinkageController::on_configure(const rclcpp_lifecycle::State &)
{
  auto node = get_node();
  const auto file = node->get_parameter("linkage_file").as_string();
  const auto names = node->get_parameter("mechanisms").as_string_array();
  const auto mode = node->get_parameter("command_mode").as_string();
  if (mode == "pd_in_drive") {
    mode_ = Mode::kPdInDrive;
    motor_cmd_ifaces_ = node->get_parameter("motor_command_interfaces").as_string_array();
    if (std::find(motor_cmd_ifaces_.begin(), motor_cmd_ifaces_.end(), "position") == motor_cmd_ifaces_.end()) {
      RCLCPP_ERROR(node->get_logger(), "pd_in_drive needs a 'position' motor command interface");
      return CallbackReturn::ERROR;
    }
  } else if (mode == "torque") {
    mode_ = Mode::kTorque;
    motor_cmd_ifaces_ = {"effort"};
  } else {
    RCLCPP_ERROR(node->get_logger(), "command_mode '%s': pd_in_drive or torque", mode.c_str());
    return CallbackReturn::ERROR;
  }
  motor_effort_state_ = node->get_parameter("motor_effort_state").as_bool();
  limit_margin_ = node->get_parameter("limit_margin").as_double();
  if (file.empty() || names.empty()) {
    RCLCPP_ERROR(node->get_logger(), "set linkage_file and mechanisms");
    return CallbackReturn::ERROR;
  }

  units_.clear();
  for (const auto & name : names) {
    Unit u;
    u.name = name;
    try {
      const auto loaded = linkage::load_mechanism(file, name);
      u.joints = {loaded.joints[0], loaded.joints[1]};
      u.mech.set_geometry(loaded.geometry);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(node->get_logger(), "%s", e.what());
      return CallbackReturn::ERROR;
    }
    auto get = [&](const std::string & key, auto fallback) {
        const std::string p = name + "." + key;
        if (!node->has_parameter(p)) {node->declare_parameter(p, fallback);}
        return node->get_parameter(p);
      };
    const auto motors = get("motors", std::vector<std::string>{}).as_string_array();
    const auto dirs = get("motor_directions", std::vector<double>{1.0, 1.0}).as_double_array();
    const auto kp = get("kp", std::vector<double>{}).as_double_array();
    const auto kd = get("kd", std::vector<double>{}).as_double_array();
    const auto lo = get("position_min", std::vector<double>{}).as_double_array();
    const auto hi = get("position_max", std::vector<double>{}).as_double_array();
    if (motors.size() != 2 || dirs.size() != 2 || kp.size() != 2 || kd.size() != 2 || lo.size() != 2 ||
      hi.size() != 2)
    {
      RCLCPP_ERROR(node->get_logger(),
        "%s: motors, motor_directions, kp, kd, position_min and position_max need 2 values each", name.c_str());
      return CallbackReturn::ERROR;
    }
    if (!(lo[0] < hi[0] && lo[1] < hi[1])) {
      RCLCPP_ERROR(node->get_logger(), "%s: position_min must be below position_max", name.c_str());
      return CallbackReturn::ERROR;
    }
    u.lower = {lo[0], lo[1]};
    u.upper = {hi[0], hi[1]};
    for (int i = 0; i < 2; ++i) {
      if (std::fabs(std::fabs(dirs[i]) - 1.0) > 1e-9) {
        RCLCPP_ERROR(node->get_logger(), "%s.motor_directions: +1 or -1", name.c_str());
        return CallbackReturn::ERROR;
      }
    }
    u.motors = {motors[0], motors[1]};
    u.direction = {dirs[0], dirs[1]};
    u.kp = {kp[0], kp[1]};
    u.kd = {kd[0], kd[1]};
    if (!u.mech.motor_angles({0.0, 0.0}, u.offset)) {
      RCLCPP_ERROR(node->get_logger(), "%s: the linkage does not close at the zero pose", name.c_str());
      return CallbackReturn::ERROR;
    }
    RCLCPP_INFO(
      node->get_logger(), "%s: joints %s, %s <- motors %s, %s (crank offsets at zero %.4f, %.4f rad)",
      name.c_str(), u.joints[0].c_str(), u.joints[1].c_str(), u.motors[0].c_str(), u.motors[1].c_str(),
      u.offset[0], u.offset[1]);
    units_.push_back(u);
  }

  const std::size_t n_joints = 2 * units_.size();
  exported_state_values_.assign(n_joints * kRefs, kNaN);
  reference_interfaces_.assign(n_joints * kRefs, kNaN);

  command_buffer_.writeFromNonRT(nullptr);
  subscription_ = node->create_subscription<std_msgs::msg::Float64MultiArray>(
    "~/commands", rclcpp::SystemDefaultsQoS(),
    [this, n_joints](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
      if (msg->data.size() != n_joints) {
        RCLCPP_WARN_THROTTLE(
          get_node()->get_logger(), *get_node()->get_clock(), 2000,
          "~/commands: %zu values, expected %zu joint positions", msg->data.size(), n_joints);
        return;
      }
      command_buffer_.writeFromNonRT(std::make_shared<std::vector<double>>(msg->data));
    });
  return CallbackReturn::SUCCESS;
}

InterfaceConfiguration ParallelLinkageController::command_interface_configuration() const
{
  InterfaceConfiguration c{interface_configuration_type::INDIVIDUAL, {}};
  for (const auto & u : units_) {
    for (const auto & m : u.motors) {
      for (const auto & i : motor_cmd_ifaces_) {c.names.push_back(m + "/" + i);}
    }
  }
  return c;
}

InterfaceConfiguration ParallelLinkageController::state_interface_configuration() const
{
  InterfaceConfiguration c{interface_configuration_type::INDIVIDUAL, {}};
  for (const auto & u : units_) {
    for (const auto & m : u.motors) {
      c.names.push_back(m + "/position");
      c.names.push_back(m + "/velocity");
      if (motor_effort_state_) {c.names.push_back(m + "/effort");}
    }
  }
  return c;
}

std::vector<hardware_interface::StateInterface::SharedPtr>
ParallelLinkageController::on_export_state_interfaces_list()
{
  std::vector<hardware_interface::StateInterface::SharedPtr> out;
  for (std::size_t u = 0; u < units_.size(); ++u) {
    for (int j = 0; j < 2; ++j) {
      for (int r = 0; r < kRefs; ++r) {
        out.push_back(std::make_shared<hardware_interface::StateInterface>(
            get_node()->get_name(), units_[u].joints[j] + "/" + kRefNames[r],
            &exported_state_values_[(2 * u + j) * kRefs + r]));
      }
    }
  }
  return out;
}

std::vector<hardware_interface::CommandInterface::SharedPtr>
ParallelLinkageController::on_export_reference_interfaces_list()
{
  std::vector<hardware_interface::CommandInterface::SharedPtr> out;
  for (std::size_t u = 0; u < units_.size(); ++u) {
    for (int j = 0; j < 2; ++j) {
      for (int r = 0; r < kRefs; ++r) {
        out.push_back(std::make_shared<hardware_interface::CommandInterface>(
            get_node()->get_name(), units_[u].joints[j] + "/" + kRefNames[r],
            &reference_interfaces_[(2 * u + j) * kRefs + r]));
      }
    }
  }
  return out;
}

bool ParallelLinkageController::on_set_chained_mode(bool) { return true; }

std::size_t ParallelLinkageController::cmd_index(std::size_t u, int motor, const std::string & iface) const
{
  const auto it = std::find(motor_cmd_ifaces_.begin(), motor_cmd_ifaces_.end(), iface);
  if (it == motor_cmd_ifaces_.end()) {return std::numeric_limits<std::size_t>::max();}
  return (2 * u + motor) * motor_cmd_ifaces_.size() + static_cast<std::size_t>(it - motor_cmd_ifaces_.begin());
}

bool ParallelLinkageController::read_unit(std::size_t u, linkage::MotorState2 & crank, linkage::JointState2 & joint)
{
  Unit & unit = units_[u];
  const std::size_t per_motor = motor_effort_state_ ? 3 : 2;
  for (int m = 0; m < 2; ++m) {
    const std::size_t base = (2 * u + m) * per_motor;
    const double d = unit.direction[m];
    crank.position[m] = d * state_interfaces_[base].get_value() + unit.offset[m];
    crank.velocity[m] = d * state_interfaces_[base + 1].get_value();
    crank.effort[m] = motor_effort_state_ ? d * state_interfaces_[base + 2].get_value() : 0.0;
    if (!std::isfinite(crank.position[m]) || !std::isfinite(crank.velocity[m])) {return false;}
  }
  if (!linkage::to_joint(unit.mech, crank, unit.q_prev, joint)) {return false;}
  for (int j = 0; j < 2; ++j) {           // another closure of the rods, not a pose the joint can be in
    if (joint.position[j] < unit.lower[j] - limit_margin_ || joint.position[j] > unit.upper[j] + limit_margin_) {
      return false;
    }
  }
  unit.q_prev = joint.position;
  unit.solved = true;
  return true;
}

void ParallelLinkageController::write_hold(std::size_t u, const linkage::Vec2 & motor_hw)
{
  for (int m = 0; m < 2; ++m) {
    const auto set = [&](const char * iface, double v) {
        const std::size_t i = cmd_index(u, m, iface);
        if (i < command_interfaces_.size()) {(void)command_interfaces_[i].set_value(v);}
      };
    if (mode_ == Mode::kTorque) {
      set("effort", 0.0);
      continue;
    }
    set("position", motor_hw[m]);
    set("velocity", 0.0);
    set("effort", 0.0);
    // kp / kd keep their last values: holding needs the drive's stiffness
  }
}

CallbackReturn ParallelLinkageController::on_activate(const rclcpp_lifecycle::State &)
{
  const std::size_t per_motor = motor_effort_state_ ? 3 : 2;
  for (std::size_t u = 0; u < units_.size(); ++u) {
    linkage::MotorState2 crank;
    linkage::JointState2 joint;
    units_[u].q_prev = {0.0, 0.0};
    if (!read_unit(u, crank, joint)) {
      RCLCPP_ERROR(get_node()->get_logger(), "%s: cannot solve the joint angles from the motor readings "
        "(calibration or motor_directions?)", units_[u].name.c_str());
      return CallbackReturn::ERROR;
    }
    for (int m = 0; m < 2; ++m) {
      units_[u].hold_motor[m] = state_interfaces_[(2 * u + m) * per_motor].get_value();
    }
    for (int j = 0; j < 2; ++j) {                  // hold where the joints are until a command arrives
      reference_interfaces_[(2 * u + j) * kRefs + kPos] = joint.position[j];
      reference_interfaces_[(2 * u + j) * kRefs + kVel] = 0.0;
      reference_interfaces_[(2 * u + j) * kRefs + kEff] = 0.0;
    }
    write_hold(u, units_[u].hold_motor);
  }
  command_buffer_.writeFromNonRT(nullptr);
  solve_failures_ = reach_failures_ = 0;
  max_coupling_ = 0.0;
  return CallbackReturn::SUCCESS;
}

CallbackReturn ParallelLinkageController::on_deactivate(const rclcpp_lifecycle::State &)
{
  const std::size_t per_motor = motor_effort_state_ ? 3 : 2;
  for (std::size_t u = 0; u < units_.size(); ++u) {
    linkage::Vec2 here{};
    for (int m = 0; m < 2; ++m) {here[m] = state_interfaces_[(2 * u + m) * per_motor].get_value();}
    write_hold(u, here);
  }
  return CallbackReturn::SUCCESS;
}

return_type ParallelLinkageController::update_reference_from_subscribers(const rclcpp::Time &, const rclcpp::Duration &)
{
  const auto cmd = command_buffer_.readFromRT();
  if (cmd && *cmd) {
    for (std::size_t k = 0; k < (*cmd)->size(); ++k) {
      reference_interfaces_[k * kRefs + kPos] = (**cmd)[k];
      reference_interfaces_[k * kRefs + kVel] = 0.0;
      reference_interfaces_[k * kRefs + kEff] = 0.0;
    }
  }
  return return_type::OK;
}

return_type ParallelLinkageController::update_and_write_commands(const rclcpp::Time &, const rclcpp::Duration &)
{
  const std::size_t per_motor = motor_effort_state_ ? 3 : 2;
  for (std::size_t u = 0; u < units_.size(); ++u) {
    Unit & unit = units_[u];
    linkage::MotorState2 crank;
    linkage::JointState2 joint;
    if (!read_unit(u, crank, joint)) {
      ++solve_failures_;
      RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "%s: motor readings the linkage cannot produce; holding the motors", unit.name.c_str());
      linkage::Vec2 here{};
      for (int m = 0; m < 2; ++m) {here[m] = state_interfaces_[(2 * u + m) * per_motor].get_value();}
      write_hold(u, here);
      continue;
    }
    for (int j = 0; j < 2; ++j) {
      double * s = &exported_state_values_[(2 * u + j) * kRefs];
      s[kPos] = joint.position[j];
      s[kVel] = joint.velocity[j];
      s[kEff] = joint.effort[j];
    }

    linkage::JointCommand2 cmd;
    for (int j = 0; j < 2; ++j) {
      const double * r = &reference_interfaces_[(2 * u + j) * kRefs];
      cmd.position[j] = std::clamp(finite_or(r[kPos], joint.position[j]), unit.lower[j], unit.upper[j]);
      cmd.velocity[j] = finite_or(r[kVel], 0.0);
      cmd.effort[j] = finite_or(r[kEff], 0.0);
    }
    cmd.kp = unit.kp;
    cmd.kd = unit.kd;

    if (mode_ == Mode::kTorque) {
      linkage::Vec2 tau{};
      if (!linkage::pd_motor_torque(unit.mech, cmd, joint, crank.position, tau)) {
        ++reach_failures_;
        write_hold(u, unit.hold_motor);
        continue;
      }
      for (int m = 0; m < 2; ++m) {
        (void)command_interfaces_[cmd_index(u, m, "effort")].set_value(unit.direction[m] * tau[m]);
      }
      continue;
    }

    linkage::MotorCommand2 out;
    if (!linkage::to_motor(unit.mech, cmd, joint.position, crank.position, out)) {
      ++reach_failures_;
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "%s: target (%.3f, %.3f) out of the linkage's reach; holding the last reachable target",
        unit.name.c_str(), cmd.position[0], cmd.position[1]);
      write_hold(u, unit.hold_motor);
      continue;
    }
    max_coupling_ = std::max(max_coupling_, out.stiffness_coupling);
    for (int m = 0; m < 2; ++m) {
      const double d = unit.direction[m];
      const double target_hw = d * (out.position[m] - unit.offset[m]);
      unit.hold_motor[m] = target_hw;
      const auto set = [&](const char * iface, double v) {
          const std::size_t i = cmd_index(u, m, iface);
          if (i < command_interfaces_.size()) {(void)command_interfaces_[i].set_value(v);}
        };
      set("position", target_hw);
      set("velocity", d * out.velocity[m]);
      set("effort", d * out.effort[m]);
      set("kp", out.kp[m]);
      set("kd", out.kd[m]);
    }
  }
  return return_type::OK;
}

}  // namespace wisco_ethercat_ros2_hw

PLUGINLIB_EXPORT_CLASS(
  wisco_ethercat_ros2_hw::ParallelLinkageController, controller_interface::ChainableControllerInterface)
