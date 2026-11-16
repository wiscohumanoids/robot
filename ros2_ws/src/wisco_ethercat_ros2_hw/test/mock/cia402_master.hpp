// The master side of one CiA402 drive, mirroring what the upstream
// ethercat_generic_plugins/EcCiA402Drive plugin (ICube ethercat_driver_ros2)
// does every cycle in processData(). Test-only.
//
// ethercat_driver_ros2 is neither in the workspace nor installed, so its
// CiA402 logic cannot be linked into the tests. This is a transcription of
// that logic (from upstream main as of 2024; diff it against the version the
// bench installs):
//   * control word: auto state transitions from the last status word
//     (0x06 -> 0x07 -> 0x0F), fault reset 0x80 on a rising edge of the
//     `reset_fault` command interface, or always with auto_fault_reset,
//   * 0x607A: overridden with the actual position whenever 0x6061 is not
//     CSP, so a drive that is not in CSP is always handed a hold target,
//   * 0x6060: the mode_of_operation command interface, else the configured
//     mode,
//   * RPDO channels with a NaN command (never written) send their default,
//   * value = factor * raw + offset on read, raw = factor * value + offset on
//     write, cast to the channel's integer type (saturated here; upstream
//     casts, so out-of-range values are undefined behaviour there),
//   * RPDO channels are processed before TPDO channels (PDO order), so the
//     control word and the 0x607A override act on the previous cycle's
//     status word, mode display and position.
#ifndef WISCO_ETHERCAT_ROS2_HW__MOCK__CIA402_MASTER_HPP_
#define WISCO_ETHERCAT_ROS2_HW__MOCK__CIA402_MASTER_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "cia402.hpp"
#include "rmdx_mock_slave.hpp"

namespace wisco_ethercat_ros2_hw
{
namespace mock
{

// Scale factors as wisco_ethercat_ros2_hw/hardware_config.py writes them into
// the slave config (direction and zero offset folded in).
struct DriveScaling
{
  double position_state_factor = 1.0;   // rad per count (x direction)
  double position_state_offset = 0.0;   // rad (-zero_offset)
  double position_command_factor = 1.0; // counts per rad
  double position_command_offset = 0.0; // counts
  double velocity_state_factor = 1.0;
  double velocity_command_factor = 1.0;
  double effort_state_factor = 1.0;     // Nm per per-mille
  double effort_command_factor = 1.0;   // per-mille per Nm

  static DriveScaling from_joint(
    double counts_per_rev, double nm_per_permille, int direction, double zero_offset)
  {
    const double rpc = 2.0 * M_PI / counts_per_rev;
    DriveScaling s;
    s.position_state_factor = direction * rpc;
    s.position_state_offset = -zero_offset;
    s.position_command_factor = direction / rpc;
    s.position_command_offset = s.position_command_factor * zero_offset;
    s.velocity_state_factor = direction * rpc;
    s.velocity_command_factor = direction / rpc;
    s.effort_state_factor = direction * nm_per_permille;
    s.effort_command_factor = direction / nm_per_permille;
    return s;
  }
};

// Command interfaces of one drive; NaN = not written.
struct DriveCommands
{
  double position = std::numeric_limits<double>::quiet_NaN();
  double velocity = std::numeric_limits<double>::quiet_NaN();
  double effort = std::numeric_limits<double>::quiet_NaN();
  double torque_offset = std::numeric_limits<double>::quiet_NaN();
  double mode_of_operation = std::numeric_limits<double>::quiet_NaN();
  double reset_fault = std::numeric_limits<double>::quiet_NaN();
  double max_torque = std::numeric_limits<double>::quiet_NaN();   // 0x6072 in N m, when mapped
};

struct DriveStates
{
  double position = 0.0;
  double velocity = 0.0;
  double effort = 0.0;
  double mode_of_operation = 0.0;   // 0x6061
  double status_word = 0.0;         // 0x6041
  double error_code = 0.0;          // 0x603F
};

class Cia402DriveMaster
{
public:
  // slave-config / ec_module parameters
  bool auto_state_transitions = true;
  bool auto_fault_reset = false;
  int8_t mode_of_operation = cia402::kModeCsp;   // ec_module param, 0x6060 default
  int16_t max_torque_permille = 1000;            // 0x6072 default from the slave config
  DriveScaling scaling;

  // One processData() pass over every channel. received is null when the
  // frame was lost: the domain memory then still holds the last inputs.
  void process(const TxPdo * received, const DriveCommands & cmd, DriveStates & st)
  {
    if (received) {tx_ = *received;}

    // ---- RPDO channels, in template order ----
    // 0x6040 control word: default_value = transition(state_, current domain value)
    if (std::isfinite(cmd.reset_fault)) {
      if (cmd.reset_fault == 0.0) {last_fault_reset_command_ = false;}
      if (!last_fault_reset_command_ && cmd.reset_fault != 0.0) {
        last_fault_reset_command_ = true;
        fault_reset_ = true;
      }
    } else {
      last_fault_reset_command_ = false;
    }
    if (auto_state_transitions) {
      rx_.control_word = transition(state_, rx_.control_word);
    }
    // 0x607A: hold the actual position unless the drive is in CSP
    {
      const bool override_cmd = mode_display_ != cia402::kModeCsp;
      if (override_cmd || !std::isfinite(cmd.position)) {
        if (mode_display_ != cia402::kModeNone) {
          rx_.target_position = write_raw<int32_t>(
            scaling.position_command_factor * last_position_ + scaling.position_command_offset);
        }
      } else {
        rx_.target_position = write_raw<int32_t>(
          scaling.position_command_factor * cmd.position + scaling.position_command_offset);
      }
    }
    rx_.target_velocity = std::isfinite(cmd.velocity) ?
      write_raw<int32_t>(scaling.velocity_command_factor * cmd.velocity) : 0;
    rx_.target_torque = std::isfinite(cmd.effort) ?
      write_raw<int16_t>(scaling.effort_command_factor * cmd.effort) : 0;
    rx_.torque_offset = std::isfinite(cmd.torque_offset) ?
      write_raw<int16_t>(scaling.effort_command_factor * cmd.torque_offset) : 0;
    rx_.max_torque = std::isfinite(cmd.max_torque) ?
      write_raw<int16_t>(std::fabs(scaling.effort_command_factor * cmd.max_torque)) : max_torque_permille;
    rx_.mode_of_operation = std::isfinite(cmd.mode_of_operation) ?
      write_raw<int8_t>(cmd.mode_of_operation) : mode_of_operation;

    // ---- TPDO channels ----
    status_word_ = tx_.status_word;
    last_position_ = scaling.position_state_factor * tx_.position_actual +
      scaling.position_state_offset;
    st.status_word = tx_.status_word;
    st.position = last_position_;
    st.velocity = scaling.velocity_state_factor * tx_.velocity_actual;
    st.effort = scaling.effort_state_factor * tx_.torque_actual;
    st.error_code = tx_.error_code;
    st.mode_of_operation = tx_.mode_display;
    mode_display_ = tx_.mode_display;

    // state change check (last channel of the domain)
    if (status_word_ != last_status_word_) {
      state_ = cia402::decode_status(status_word_);
    }
    last_status_word_ = status_word_;
  }

  const RxPdo & rx() const {return rx_;}
  cia402::State state() const {return state_;}

  // EcCiA402Drive::transition()
  uint16_t transition(cia402::State state, uint16_t control_word)
  {
    using cia402::State;
    switch (state) {
      case State::kStart:
      case State::kNotReadyToSwitchOn:
        return control_word;
      case State::kSwitchOnDisabled:
        return (control_word & 0b01111110) | 0b00000110;
      case State::kReadyToSwitchOn:
        return (control_word & 0b01110111) | 0b00000111;
      case State::kSwitchedOn:
        return (control_word & 0b01111111) | 0b00001111;
      case State::kOperationEnabled:
        return control_word;
      case State::kQuickStopActive:
        return (control_word & 0b01111111) | 0b00001111;
      case State::kFaultReactionActive:
        return control_word;
      case State::kFault:
        if (fault_reset_ || auto_fault_reset) {
          fault_reset_ = false;
          return (control_word & 0b11111111) | 0b10000000;
        }
        return control_word;
    }
    return control_word;
  }

private:
  template<typename T>
  static T write_raw(double v)
  {
    const double lo = static_cast<double>(std::numeric_limits<T>::min());
    const double hi = static_cast<double>(std::numeric_limits<T>::max());
    return static_cast<T>(std::clamp(v, lo, hi));
  }

  RxPdo rx_;
  TxPdo tx_;
  cia402::State state_ = cia402::State::kStart;
  uint16_t status_word_ = 0;
  uint16_t last_status_word_ = 0xFFFF;
  int8_t mode_display_ = cia402::kModeNone;
  double last_position_ = 0.0;
  bool fault_reset_ = false;
  bool last_fault_reset_command_ = false;
};

}  // namespace mock
}  // namespace wisco_ethercat_ros2_hw

#endif  // WISCO_ETHERCAT_ROS2_HW__MOCK__CIA402_MASTER_HPP_
