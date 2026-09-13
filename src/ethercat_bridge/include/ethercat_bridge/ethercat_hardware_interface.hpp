#ifndef ETHERCAT_BRIDGE__ETHERCAT_HARDWARE_INTERFACE_HPP_
#define ETHERCAT_BRIDGE__ETHERCAT_HARDWARE_INTERFACE_HPP_

#include <string>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/clock.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace ethercat_bridge
{

// STATUS: REAL SCAFFOLD. This class is a genuine, loadable
// hardware_interface::SystemInterface plugin -- controller_manager can
// activate it today and lowlevel_control's JointImpedanceController will
// claim its interfaces exactly as it does mujoco_ros2_control/MujocoSystem's
// (see g1_description's <ros2_control> block and ARCHITECTURE.md "Sim/
// hardware symmetry"). What is STUBBED, explicitly, is everything that would
// require an actual EtherCAT master and physical bus: on_configure() parses
// and validates config/ethercat_slaves.yaml for real, but does not open a
// bus; on_activate()/on_deactivate() log the CiA402 Controlword transitions
// they *would* send, but send nothing; read() returns static/held state
// instead of consuming a real TxPDO frame; write() logs the torque command
// it would pack into a RxPDO frame instead of sending one.
//
// See RESEARCH_NOTES.md for the ethercat_driver_ros2 / CiA402 / ros2_control
// research this scaffold's structure follows, and INTEGRATION_POINTS.md
// "Low-level / EtherCAT" for the exact TODO checklist to turn this into a
// real hardware driver.
class EthercatHardwareInterface : public hardware_interface::SystemInterface
{
public:
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;
  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_cleanup(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  bool load_and_validate_slave_config(const std::string & path);

  std::vector<std::string> joint_names_;

  // Internal storage bound directly to exported state/command interfaces,
  // same pattern as mujoco_system.cpp (see RESEARCH_NOTES.md section 2).
  std::vector<double> position_state_;
  std::vector<double> velocity_state_;
  std::vector<double> effort_state_;
  std::vector<double> effort_command_;

  // Placeholder IMU state -- identity orientation, zero gyro/accel. See
  // README.md "What's not done" -- a real pelvis IMU is not necessarily on
  // the EtherCAT PDO bus at all (often a separate serial/CAN link), so this
  // is tracked as its own TODO, not assumed to arrive via read().
  double imu_orientation_[4] = {0.0, 0.0, 0.0, 1.0};  // x, y, z, w
  double imu_angular_velocity_[3] = {0.0, 0.0, 0.0};
  double imu_linear_acceleration_[3] = {0.0, 0.0, 0.0};

  std::string network_interface_;
  std::string slave_config_path_;
  size_t slave_count_{0};
  bool activated_{false};
  rclcpp::Clock steady_clock_{RCL_STEADY_TIME};  // for throttled logging in write()
};

}  // namespace ethercat_bridge

#endif  // ETHERCAT_BRIDGE__ETHERCAT_HARDWARE_INTERFACE_HPP_
