#include "ethercat_bridge/ethercat_hardware_interface.hpp"

#include <algorithm>
#include <exception>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "rclcpp/rclcpp.hpp"
#include "yaml-cpp/yaml.h"

namespace ethercat_bridge
{

namespace
{
rclcpp::Logger logger() { return rclcpp::get_logger("ethercat_bridge"); }

// IMU sensor interface suffixes, matching g1_description's <sensor
// name="imu_imu"> block and lowlevel_control's expectations exactly. See
// RESEARCH_NOTES.md -- a real pelvis IMU on this repo's target hardware is
// not necessarily an EtherCAT PDO device at all (often a separate serial/CAN
// link), so this stays a fixed placeholder here independent of the
// EtherCAT-specific TODOs elsewhere in this file.
constexpr const char * kImuSensorName = "imu_imu";
const std::vector<std::string> kImuInterfaceSuffixes = {
  "orientation.x", "orientation.y", "orientation.z", "orientation.w",
  "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
  "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z",
};
}  // namespace

hardware_interface::CallbackReturn EthercatHardwareInterface::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  for (const auto & joint : info_.joints)
  {
    joint_names_.push_back(joint.name);
  }
  const size_t n = joint_names_.size();
  position_state_.assign(n, 0.0);
  velocity_state_.assign(n, 0.0);
  effort_state_.assign(n, 0.0);
  effort_command_.assign(n, 0.0);

  // Seed position_state_ from each joint's <state_interface name="position">
  // initial_value param, same convention g1_description's <ros2_control>
  // block uses for mujoco_ros2_control/MujocoSystem -- so the STUBBED
  // read() below (which never updates these) at least reports the robot's
  // configured nominal pose instead of an arbitrary 0.0 for every joint.
  for (size_t i = 0; i < n; ++i)
  {
    for (const auto & state_if : info_.joints[i].state_interfaces)
    {
      if (state_if.name == hardware_interface::HW_IF_POSITION)
      {
        auto it = state_if.parameters.find("initial_value");
        if (it != state_if.parameters.end())
        {
          try
          {
            position_state_[i] = std::stod(it->second);
          }
          catch (const std::exception &)
          {
            RCLCPP_WARN(
              logger(), "joint '%s': could not parse initial_value '%s', using 0.0",
              joint_names_[i].c_str(), it->second.c_str());
          }
        }
      }
    }
  }

  auto get_param = [this](const std::string & key, const std::string & def) {
      auto it = info_.hardware_parameters.find(key);
      return it != info_.hardware_parameters.end() ? it->second : def;
    };
  network_interface_ = get_param("ethercat_network_interface", "eth0");
  const std::string config_rel = get_param("ethercat_slave_config", "config/ethercat_slaves.yaml");
  if (!config_rel.empty() && config_rel.front() == '/')
  {
    slave_config_path_ = config_rel;
  }
  else
  {
    try
    {
      slave_config_path_ =
        ament_index_cpp::get_package_share_directory("ethercat_bridge") + "/" + config_rel;
    }
    catch (const std::exception & e)
    {
      RCLCPP_ERROR(logger(), "could not resolve ethercat_bridge share directory: %s", e.what());
      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  RCLCPP_INFO(
    logger(), "on_init: %zu joints, network_interface='%s' (STUB -- not opened yet), "
    "slave_config='%s'", n, network_interface_.c_str(), slave_config_path_.c_str());
  return hardware_interface::CallbackReturn::SUCCESS;
}

bool EthercatHardwareInterface::load_and_validate_slave_config(const std::string & path)
{
  YAML::Node root;
  try
  {
    root = YAML::LoadFile(path);
  }
  catch (const std::exception & e)
  {
    RCLCPP_ERROR(logger(), "failed to load %s: %s", path.c_str(), e.what());
    return false;
  }

  if (!root["slaves"] || !root["slaves"].IsSequence())
  {
    RCLCPP_ERROR(logger(), "%s: missing or malformed top-level 'slaves' list", path.c_str());
    return false;
  }
  const auto slaves = root["slaves"];
  slave_count_ = slaves.size();

  if (slave_count_ != joint_names_.size())
  {
    RCLCPP_ERROR(
      logger(), "%s: %zu slave entries but URDF declares %zu joints -- these must match 1:1",
      path.c_str(), slave_count_, joint_names_.size());
    return false;
  }

  bool any_placeholder = false;
  for (size_t i = 0; i < slaves.size(); ++i)
  {
    const auto slave = slaves[i];
    if (!slave["joint"])
    {
      RCLCPP_ERROR(logger(), "%s: slaves[%zu] missing 'joint' key", path.c_str(), i);
      return false;
    }
    const std::string joint = slave["joint"].as<std::string>();
    if (joint != joint_names_[i])
    {
      RCLCPP_ERROR(
        logger(),
        "%s: slaves[%zu].joint = '%s' but CANONICAL_JOINT_ORDER index %zu is '%s' -- "
        "slave list order must match g1_description's <ros2_control> joint order exactly",
        path.c_str(), i, joint.c_str(), i, joint_names_[i].c_str());
      return false;
    }

    // Compared as strings (not parsed as hex integers) deliberately -- avoids
    // depending on yaml-cpp's hex-literal numeric conversion behavior, which
    // was not verified in this session (see RESEARCH_NOTES.md). "0x0" and
    // "0" are also treated as placeholders in case someone abbreviates.
    const std::string vendor_id = slave["vendor_id"] ? slave["vendor_id"].as<std::string>() : "";
    const std::string product_id = slave["product_id"] ? slave["product_id"].as<std::string>() : "";
    const std::vector<std::string> placeholder_values = {"", "0x00000000", "0x0", "0"};
    const bool is_placeholder =
      std::find(placeholder_values.begin(), placeholder_values.end(), vendor_id) !=
        placeholder_values.end() ||
      std::find(placeholder_values.begin(), placeholder_values.end(), product_id) !=
        placeholder_values.end();
    if (is_placeholder)
    {
      any_placeholder = true;
    }
  }

  if (any_placeholder)
  {
    RCLCPP_WARN(
      logger(),
      "%s: one or more slaves still have placeholder vendor_id/product_id == 0x00000000. "
      "Config structure and joint ordering are valid, so on_configure() will still succeed "
      "(this repo's sim path does not need real drive IDs) -- but on_activate() cannot do "
      "anything real against hardware until these are filled in from datasheets. "
      "See INTEGRATION_POINTS.md 'Low-level / EtherCAT'.",
      path.c_str());
  }

  RCLCPP_INFO(
    logger(), "%s: loaded and validated %zu slave configs, joint order matches canonical order",
    path.c_str(), slave_count_);
  return true;
}

hardware_interface::CallbackReturn EthercatHardwareInterface::on_configure(
  const rclcpp_lifecycle::State &)
{
  if (!load_and_validate_slave_config(slave_config_path_))
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  // TODO(hardware team): this is the real EtherCAT master bring-up point.
  // Per RESEARCH_NOTES.md section 1, this is where either (a) SOEM's
  // ec_init(network_interface_.c_str()) + ec_config_init() slave scan, or
  // (b) a delegated call into ethercat_driver_ros2's master abstraction,
  // belongs. Currently a no-op: no NIC is opened, no slaves are scanned, and
  // the slave count validated above is never checked against a REAL bus
  // topology (only against the YAML file's internal consistency).
  RCLCPP_WARN(
    logger(),
    "on_configure: STUB -- EtherCAT master on interface '%s' was NOT opened, no bus scan was "
    "performed. Config validated OK against %zu joints. See ethercat_bridge/README.md.",
    network_interface_.c_str(), joint_names_.size());

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn EthercatHardwareInterface::on_cleanup(
  const rclcpp_lifecycle::State &)
{
  // TODO(hardware team): real bus teardown (ec_close() or equivalent) goes here.
  activated_ = false;
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn EthercatHardwareInterface::on_activate(
  const rclcpp_lifecycle::State &)
{
  // TODO(hardware team): real CiA402 state-machine bring-up goes here, per
  // slave, per RESEARCH_NOTES.md section 3:
  //   write Controlword 0x0006 (Shutdown)        -> wait for Statusword "Ready to Switch On"
  //   write Controlword 0x0007 (Switch On)        -> wait for Statusword "Switched On"
  //   write Controlword 0x000F (Enable Operation) -> wait for Statusword "Operation Enabled"
  // The Statusword bitmask/decode table needed to actually detect those
  // "wait for" conditions is flagged UNCONFIRMED in RESEARCH_NOTES.md --
  // verify against the real drive datasheet before implementing this loop,
  // do not trust a copy-pasted generic mask blindly.
  RCLCPP_WARN(
    logger(),
    "on_activate: STUB -- no real CiA402 Controlword transitions were sent to any of the %zu "
    "slaves (would be Shutdown 0x0006 -> Switch On 0x0007 -> Enable Operation 0x000F per "
    "RESEARCH_NOTES.md section 3). Effort commands will be accepted and logged by write() but "
    "no torque reaches real hardware.",
    joint_names_.size());
  activated_ = true;
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn EthercatHardwareInterface::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  // TODO(hardware team): write Controlword 0x0000 (Disable Voltage) per slave.
  RCLCPP_WARN(logger(), "on_deactivate: STUB -- no real Disable Voltage transition was sent.");
  activated_ = false;
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> EthercatHardwareInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  for (size_t i = 0; i < joint_names_.size(); ++i)
  {
    state_interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_POSITION, &position_state_[i]);
    state_interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_VELOCITY, &velocity_state_[i]);
    state_interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_EFFORT, &effort_state_[i]);
  }

  // Placeholder IMU -- identity orientation, zero gyro/accel, see header
  // comment and README.md "What's not done".
  double * imu_ptrs[10] = {
    &imu_orientation_[0], &imu_orientation_[1], &imu_orientation_[2], &imu_orientation_[3],
    &imu_angular_velocity_[0], &imu_angular_velocity_[1], &imu_angular_velocity_[2],
    &imu_linear_acceleration_[0], &imu_linear_acceleration_[1], &imu_linear_acceleration_[2],
  };
  for (size_t i = 0; i < kImuInterfaceSuffixes.size(); ++i)
  {
    state_interfaces.emplace_back(kImuSensorName, kImuInterfaceSuffixes[i], imu_ptrs[i]);
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface>
EthercatHardwareInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  for (size_t i = 0; i < joint_names_.size(); ++i)
  {
    command_interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_EFFORT, &effort_command_[i]);
  }
  return command_interfaces;
}

hardware_interface::return_type EthercatHardwareInterface::read(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  // STUB: no real TxPDO consumption -- position_state_/velocity_state_/
  // effort_state_ are simply held at whatever they were last set to
  // (initialized from URDF initial_value in on_init(), never updated here).
  // Deliberately static rather than a fabricated physics model: per this
  // repo's "don't fake completeness" policy, this class must not pretend to
  // simulate a robot -- that job belongs to mujoco_ros2_control/MujocoSystem.
  //
  // TODO(hardware team): for each slave, decode its last-received TxPDO
  // frame per config/ethercat_slaves.yaml's tx_pdo mapping:
  //   position_state_[i] = decode(actual_position, 0x6064);
  //   velocity_state_[i] = decode(actual_velocity, 0x606C);
  //   effort_state_[i]   = decode(actual_torque,   0x6077);
  // Confirm each object's engineering-unit scaling (counts-per-radian,
  // per-mille-of-rated-torque, etc.) against the real drive datasheet --
  // CiA402 defines the objects, not the physical units a given vendor uses
  // for them.
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type EthercatHardwareInterface::write(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  if (!activated_)
  {
    return hardware_interface::return_type::OK;
  }

  // STUB: log what would be sent, send nothing. Throttled to avoid flooding
  // the log at 1kHz.
  RCLCPP_INFO_THROTTLE(
    logger(), steady_clock_, 5000,
    "write(): STUB -- would pack %zu joints' effort_command_ into RxPDO TargetTorque (0x6071) "
    "and run the EtherCAT master send/receive cycle on '%s'. No frame sent. "
    "(This message is throttled to once per 5s; it fires at whatever rate write() is called, "
    "1000 Hz per bringup/config/controllers.yaml.)",
    joint_names_.size(), network_interface_.c_str());

  // TODO(hardware team): for each slave, encode effort_command_[i] into its
  // RxPDO TargetTorque object per config/ethercat_slaves.yaml, honoring the
  // drive's actual torque-to-counts scaling from its datasheet, then submit
  // the frame via the real EtherCAT master's send/receive cycle
  // (ecx_send_processdata / ecx_receive_processdata in SOEM terms).

  return hardware_interface::return_type::OK;
}

}  // namespace ethercat_bridge

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  ethercat_bridge::EthercatHardwareInterface, hardware_interface::SystemInterface)
