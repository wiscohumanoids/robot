// A ros2_control SystemInterface that stands in for ethercat_driver/EthercatDriver
// with one RmdxMockSlave per joint on a simulated ring. Test-only.
//
// The controllers under test (ArmController, SafeCommandController) claim
// the same interface names the real EcCiA402Drive joints export, and the
// per-drive master logic is Cia402DriveMaster (the upstream processData()
// transcription), so the controller -> command interface -> PDO -> drive
// path is the one the bench runs, minus the IgH master.
//
// Bus timing mirrors upstream EthercatDriver: read() receives the frame and
// runs processData() (TPDO -> state interfaces, command interfaces -> RPDO
// image), write() sends the image. A command written in update() at cycle k
// is therefore copied into the image at read() k+1 and reaches the drive at
// write() k+1.
//
// Hardware parameters (HardwareInfo::hardware_parameters):
//   mode_of_operation    0x6060 when the command interface is unset (default 8)
//   auto_fault_reset     "true" / "false" (default false)
// Joint parameters (ComponentInfo::parameters):
//   counts_per_rev, effort_nm_per_permille, direction, zero_offset,
//   max_torque_permille (0x6072), peak_effort_nm, inertia, initial_position
//
// Per joint it exports state interfaces position, velocity, effort,
// mode_of_operation (0x6061), status_word (0x6041), error_code (0x603F) and
// command interfaces position, velocity, effort, torque_offset,
// mode_of_operation, reset_fault, max_torque (0x6072, N m; unwritten = the
// slave-config default).
#ifndef WISCO_ETHERCAT_ROS2_HW__MOCK__MOCK_ETHERCAT_SYSTEM_HPP_
#define WISCO_ETHERCAT_ROS2_HW__MOCK__MOCK_ETHERCAT_SYSTEM_HPP_

#include <functional>
#include <string>
#include <vector>

#include "cia402_master.hpp"
#include "hardware_interface/system_interface.hpp"
#include "rmdx_mock_slave.hpp"

namespace wisco_ethercat_ros2_hw
{
namespace mock
{

class MockEthercatSystem : public hardware_interface::SystemInterface
{
public:
  CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  hardware_interface::return_type read(const rclcpp::Time &, const rclcpp::Duration &) override;
  hardware_interface::return_type write(const rclcpp::Time &, const rclcpp::Duration &) override;

  size_t size() const {return slaves_.size();}
  RmdxMockSlave & slave(size_t i) {return slaves_[i];}
  Cia402DriveMaster & master(size_t i) {return masters_[i];}
  DriveCommands & commands(size_t i) {return commands_[i];}
  const DriveStates & states(size_t i) const {return states_[i];}

  // Lose the next n frames (both directions).
  void drop_frames(int n) {drop_ += n;}
  // Load torque per joint from all joint positions (true positions, rad, in
  // joint frame), e.g. gravity. Called every cycle before the drives step.
  void set_load_model(std::function<void(const std::vector<double> &, std::vector<double> &)> f)
  {
    load_model_ = std::move(f);
  }
  // Joint-frame true position of joint i (direction and zero offset applied).
  double joint_position(size_t i) const;
  double joint_velocity(size_t i) const;
  // Joint-frame torque the drive produced last cycle, Nm.
  double joint_torque(size_t i) const;

private:
  std::vector<RmdxMockSlave> slaves_;
  std::vector<Cia402DriveMaster> masters_;
  std::vector<DriveCommands> commands_;
  std::vector<DriveStates> states_;
  std::vector<TxPdo> inbox_;
  std::vector<int> direction_;
  std::vector<double> zero_offset_;
  bool have_frame_ = false;
  int drop_ = 0;
  std::function<void(const std::vector<double> &, std::vector<double> &)> load_model_;
  std::vector<double> q_true_, load_;
};

}  // namespace mock
}  // namespace wisco_ethercat_ros2_hw

#endif  // WISCO_ETHERCAT_ROS2_HW__MOCK__MOCK_ETHERCAT_SYSTEM_HPP_
