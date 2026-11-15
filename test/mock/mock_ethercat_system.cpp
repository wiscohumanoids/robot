#include "mock_ethercat_system.hpp"

#include <string>

namespace wisco_ethercat_ros2_hw
{
namespace mock
{

namespace
{
double param(
  const std::unordered_map<std::string, std::string> & params, const std::string & key,
  double fallback)
{
  auto it = params.find(key);
  return it == params.end() ? fallback : std::stod(it->second);
}
}  // namespace

MockEthercatSystem::CallbackReturn MockEthercatSystem::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }
  const size_t n = info.joints.size();
  const auto & hw = info.hardware_parameters;
  const int8_t mode = static_cast<int8_t>(param(hw, "mode_of_operation", cia402::kModeCsp));
  const auto afr = hw.find("auto_fault_reset");
  const bool auto_fault_reset = afr != hw.end() && afr->second == "true";

  slaves_.clear();
  masters_.assign(n, Cia402DriveMaster());
  commands_.assign(n, DriveCommands());
  states_.assign(n, DriveStates());
  inbox_.assign(n, TxPdo());
  direction_.assign(n, 1);
  zero_offset_.assign(n, 0.0);
  q_true_.assign(n, 0.0);
  load_.assign(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    const auto & p = info.joints[i].parameters;
    RmdxParams sp;
    sp.counts_per_rev = param(p, "counts_per_rev", 131072.0);
    const double nm_per_permille = param(p, "effort_nm_per_permille", 0.015);
    sp.rated_torque_nm = 1000.0 * nm_per_permille;
    sp.peak_torque_nm = param(p, "peak_effort_nm", sp.rated_torque_nm);
    sp.inertia = param(p, "inertia", sp.inertia);
    direction_[i] = static_cast<int>(param(p, "direction", 1.0));
    zero_offset_[i] = param(p, "zero_offset", 0.0);
    sp.initial_position = direction_[i] * (param(p, "initial_position", 0.0) + zero_offset_[i]);
    slaves_.emplace_back(sp);

    auto & m = masters_[i];
    m.mode_of_operation = mode;
    m.auto_fault_reset = auto_fault_reset;
    m.max_torque_permille = static_cast<int16_t>(param(p, "max_torque_permille", 1000.0));
    m.scaling = DriveScaling::from_joint(
      sp.counts_per_rev, nm_per_permille, direction_[i], zero_offset_[i]);
  }
  have_frame_ = false;
  drop_ = 0;
  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> MockEthercatSystem::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> out;
  for (size_t i = 0; i < info_.joints.size(); ++i) {
    const auto & j = info_.joints[i].name;
    auto & s = states_[i];
    out.emplace_back(j, "position", &s.position);
    out.emplace_back(j, "velocity", &s.velocity);
    out.emplace_back(j, "effort", &s.effort);
    out.emplace_back(j, "mode_of_operation", &s.mode_of_operation);
    out.emplace_back(j, "status_word", &s.status_word);
    out.emplace_back(j, "error_code", &s.error_code);
  }
  return out;
}

std::vector<hardware_interface::CommandInterface> MockEthercatSystem::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> out;
  for (size_t i = 0; i < info_.joints.size(); ++i) {
    const auto & j = info_.joints[i].name;
    auto & c = commands_[i];
    out.emplace_back(j, "position", &c.position);
    out.emplace_back(j, "velocity", &c.velocity);
    out.emplace_back(j, "effort", &c.effort);
    out.emplace_back(j, "torque_offset", &c.torque_offset);
    out.emplace_back(j, "mode_of_operation", &c.mode_of_operation);
    out.emplace_back(j, "reset_fault", &c.reset_fault);
    out.emplace_back(j, "max_torque", &c.max_torque);
  }
  return out;
}

hardware_interface::return_type MockEthercatSystem::read(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  for (size_t i = 0; i < slaves_.size(); ++i) {
    masters_[i].process(have_frame_ ? &inbox_[i] : nullptr, commands_[i], states_[i]);
  }
  have_frame_ = false;
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type MockEthercatSystem::write(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  if (load_model_) {
    for (size_t i = 0; i < slaves_.size(); ++i) {q_true_[i] = joint_position(i);}
    load_model_(q_true_, load_);
    for (size_t i = 0; i < slaves_.size(); ++i) {
      slaves_[i].set_load_torque(direction_[i] * load_[i]);
    }
  }
  const bool lost = drop_ > 0;
  if (lost) {--drop_;}
  for (size_t i = 0; i < slaves_.size(); ++i) {
    if (lost) {
      slaves_[i].cycle(nullptr);
    } else {
      inbox_[i] = slaves_[i].cycle(&masters_[i].rx());
    }
  }
  have_frame_ = !lost;
  return hardware_interface::return_type::OK;
}

double MockEthercatSystem::joint_position(size_t i) const
{
  return direction_[i] * slaves_[i].position() - zero_offset_[i];
}

double MockEthercatSystem::joint_velocity(size_t i) const
{
  return direction_[i] * slaves_[i].velocity();
}

double MockEthercatSystem::joint_torque(size_t i) const
{
  return direction_[i] * slaves_[i].applied_torque();
}

}  // namespace mock
}  // namespace wisco_ethercat_ros2_hw
