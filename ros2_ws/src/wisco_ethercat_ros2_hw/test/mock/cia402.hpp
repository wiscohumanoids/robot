// CiA402 (IEC 61800-7-201) definitions shared by the mock drive and the
// mirrored master-side logic. Test-only.
#ifndef WISCO_ETHERCAT_ROS2_HW__MOCK__CIA402_HPP_
#define WISCO_ETHERCAT_ROS2_HW__MOCK__CIA402_HPP_

#include <cstdint>

namespace wisco_ethercat_ros2_hw
{
namespace mock
{
namespace cia402
{

// Object dictionary indices used by the RMD-X PDO map (config/drive_pdo_template.yaml).
constexpr uint16_t kControlWord = 0x6040;
constexpr uint16_t kStatusWord = 0x6041;
constexpr uint16_t kErrorCode = 0x603F;
constexpr uint16_t kModeOfOperation = 0x6060;
constexpr uint16_t kModeDisplay = 0x6061;
constexpr uint16_t kPositionActual = 0x6064;
constexpr uint16_t kVelocityActual = 0x606C;
constexpr uint16_t kTargetTorque = 0x6071;
constexpr uint16_t kMaxTorque = 0x6072;
constexpr uint16_t kTorqueActual = 0x6077;
constexpr uint16_t kTargetPosition = 0x607A;
constexpr uint16_t kTorqueOffset = 0x60B2;
constexpr uint16_t kTargetVelocity = 0x60FF;

// 0x6060 / 0x6061
constexpr int8_t kModeNone = 0;
constexpr int8_t kModeCsp = 8;
constexpr int8_t kModeCsv = 9;
constexpr int8_t kModeCst = 10;

// Control word commands (bits 0..3, 7). Bit 2 is "quick stop" active low.
constexpr uint16_t kCwShutdown = 0x0006;
constexpr uint16_t kCwSwitchOn = 0x0007;
constexpr uint16_t kCwDisableVoltage = 0x0000;
constexpr uint16_t kCwQuickStop = 0x0002;
constexpr uint16_t kCwDisableOperation = 0x0007;
constexpr uint16_t kCwEnableOperation = 0x000F;
constexpr uint16_t kCwFaultReset = 0x0080;

// Status word bits beyond the state pattern.
constexpr uint16_t kSwVoltageEnabled = 0x0010;
constexpr uint16_t kSwWarning = 0x0080;
constexpr uint16_t kSwRemote = 0x0200;
constexpr uint16_t kSwTargetReached = 0x0400;
constexpr uint16_t kSwFollowingCommand = 0x1000;   // CSP/CST: drive follows the command value
constexpr uint16_t kSwFollowingError = 0x2000;     // CSP

// Error codes (0x603F) the mock raises. CiA402 / CiA301 generic codes; the
// RMD-X's own codes are not documented (bench check).
constexpr uint16_t kErrFollowing = 0x8611;      // following error
constexpr uint16_t kErrHeartbeat = 0x8130;      // process data watchdog (SM watchdog) expired
constexpr uint16_t kErrOverCurrent = 0x2310;    // continuous over current (injection default)

enum class State
{
  kStart,
  kNotReadyToSwitchOn,
  kSwitchOnDisabled,
  kReadyToSwitchOn,
  kSwitchedOn,
  kOperationEnabled,
  kQuickStopActive,
  kFaultReactionActive,
  kFault,
};

inline const char * state_name(State s)
{
  switch (s) {
    case State::kStart: return "START";
    case State::kNotReadyToSwitchOn: return "NOT_READY_TO_SWITCH_ON";
    case State::kSwitchOnDisabled: return "SWITCH_ON_DISABLED";
    case State::kReadyToSwitchOn: return "READY_TO_SWITCH_ON";
    case State::kSwitchedOn: return "SWITCHED_ON";
    case State::kOperationEnabled: return "OPERATION_ENABLED";
    case State::kQuickStopActive: return "QUICK_STOP_ACTIVE";
    case State::kFaultReactionActive: return "FAULT_REACTION_ACTIVE";
    case State::kFault: return "FAULT";
  }
  return "?";
}

// Status word state pattern (bits 0,1,2,3,5,6), CiA402 table 30.
inline uint16_t state_bits(State s)
{
  switch (s) {
    case State::kStart:
    case State::kNotReadyToSwitchOn: return 0x0000;
    case State::kSwitchOnDisabled: return 0x0040;
    case State::kReadyToSwitchOn: return 0x0021;
    case State::kSwitchedOn: return 0x0023;
    case State::kOperationEnabled: return 0x0027;
    case State::kQuickStopActive: return 0x0007;
    case State::kFaultReactionActive: return 0x000F;
    case State::kFault: return 0x0008;
  }
  return 0;
}

// Status word -> state, with the masks ethercat_generic_plugins'
// EcCiA402Drive::deviceState() uses.
inline State decode_status(uint16_t sw)
{
  if ((sw & 0x4F) == 0x00) {return State::kNotReadyToSwitchOn;}
  if ((sw & 0x4F) == 0x40) {return State::kSwitchOnDisabled;}
  if ((sw & 0x6F) == 0x21) {return State::kReadyToSwitchOn;}
  if ((sw & 0x6F) == 0x23) {return State::kSwitchedOn;}
  if ((sw & 0x6F) == 0x27) {return State::kOperationEnabled;}
  if ((sw & 0x6F) == 0x07) {return State::kQuickStopActive;}
  if ((sw & 0x4F) == 0x0F) {return State::kFaultReactionActive;}
  if ((sw & 0x4F) == 0x08) {return State::kFault;}
  return State::kStart;
}

inline bool operation_enabled(uint16_t sw) {return (sw & 0x6F) == 0x27;}

}  // namespace cia402
}  // namespace mock
}  // namespace wisco_ethercat_ros2_hw

#endif  // WISCO_ETHERCAT_ROS2_HW__MOCK__CIA402_HPP_
