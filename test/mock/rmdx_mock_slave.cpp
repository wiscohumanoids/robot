#include "rmdx_mock_slave.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace wisco_ethercat_ros2_hw
{
namespace mock
{

using cia402::State;

namespace
{
template<typename T>
T saturate_round(double v)
{
  const double lo = static_cast<double>(std::numeric_limits<T>::min());
  const double hi = static_cast<double>(std::numeric_limits<T>::max());
  return static_cast<T>(std::clamp(std::round(v), lo, hi));
}
}  // namespace

RmdxMockSlave::RmdxMockSlave(const RmdxParams & params)
: p_(params)
{
  q_ = p_.initial_position;
  history_.push_back(state_);
  latched_ = latch();
}

double RmdxMockSlave::rad_per_count() const
{
  return 2.0 * M_PI / p_.counts_per_rev;
}

void RmdxMockSlave::set_state(State s)
{
  if (s == state_) {return;}
  state_ = s;
  history_.push_back(s);
  if (s == State::kOperationEnabled) {
    have_prev_target_ = false;
    following_error_time_ = 0.0;
  }
  hold_set_ = false;
  state_cycles_ = 0;
}

void RmdxMockSlave::inject_fault(uint16_t error_code, bool persistent)
{
  fault_cause_active_ = persistent;
  raise_fault(error_code);
}

void RmdxMockSlave::clear_fault_cause()
{
  fault_cause_active_ = false;
}

void RmdxMockSlave::raise_fault(uint16_t code)
{
  if (state_ == State::kFault || state_ == State::kFaultReactionActive) {return;}
  error_code_ = code;
  set_state(State::kFaultReactionActive);
}

void RmdxMockSlave::apply_control_word(uint16_t cw)
{
  const bool reset_edge = (cw & cia402::kCwFaultReset) && !(last_cw_ & cia402::kCwFaultReset);
  last_cw_ = cw;

  switch (state_) {
    case State::kFault:
      // transition 15, on the rising edge of bit 7 only, and only once the cause is gone
      if (reset_edge && !fault_cause_active_) {
        error_code_ = 0;
        following_error_flag_ = false;
        set_state(State::kSwitchOnDisabled);
      }
      return;
    case State::kFaultReactionActive:
    case State::kNotReadyToSwitchOn:
    case State::kStart:
      return;
    default:
      break;
  }

  enum class Cmd { kDisableVoltage, kQuickStop, kShutdown, kSwitchOn, kEnableOperation };
  Cmd c;
  if (!(cw & 0x2)) {
    c = Cmd::kDisableVoltage;
  } else if (!(cw & 0x4)) {
    c = Cmd::kQuickStop;
  } else if (!(cw & 0x1)) {
    c = Cmd::kShutdown;
  } else if (!(cw & 0x8)) {
    c = Cmd::kSwitchOn;
  } else {
    c = Cmd::kEnableOperation;
  }

  switch (state_) {
    case State::kSwitchOnDisabled:
      if (c == Cmd::kShutdown) {set_state(State::kReadyToSwitchOn);}                 // 2
      break;
    case State::kReadyToSwitchOn:
      // 3. Strict: 0x0F here only switches on; operation needs a second 0x0F.
      if (c == Cmd::kSwitchOn || c == Cmd::kEnableOperation) {set_state(State::kSwitchedOn);}
      if (c == Cmd::kDisableVoltage || c == Cmd::kQuickStop) {set_state(State::kSwitchOnDisabled);}  // 7
      break;
    case State::kSwitchedOn:
      if (c == Cmd::kEnableOperation) {set_state(State::kOperationEnabled);}        // 4
      if (c == Cmd::kShutdown) {set_state(State::kReadyToSwitchOn);}                // 6
      if (c == Cmd::kDisableVoltage || c == Cmd::kQuickStop) {set_state(State::kSwitchOnDisabled);}  // 10
      break;
    case State::kOperationEnabled:
      if (c == Cmd::kSwitchOn) {set_state(State::kSwitchedOn);}                     // 5
      if (c == Cmd::kShutdown) {set_state(State::kReadyToSwitchOn);}                // 8
      if (c == Cmd::kDisableVoltage) {set_state(State::kSwitchOnDisabled);}         // 9
      if (c == Cmd::kQuickStop) {set_state(State::kQuickStopActive);}               // 11
      break;
    case State::kQuickStopActive:
      if (c == Cmd::kDisableVoltage) {set_state(State::kSwitchOnDisabled);}         // 12
      if (c == Cmd::kEnableOperation && p_.quick_stop_option >= 5) {
        set_state(State::kOperationEnabled);                                        // 16
      }
      break;
    default:
      break;
  }
}

void RmdxMockSlave::update_mode(int8_t requested)
{
  // Guess: a mode the drive does not support is ignored (0x6061 keeps the old one).
  if (requested != cia402::kModeCsp && requested != cia402::kModeCsv &&
    requested != cia402::kModeCst)
  {
    return;
  }
  if (requested == mode_active_) {
    mode_pending_ = requested;
    mode_pending_cycles_ = 0;
    return;
  }
  if (requested != mode_pending_) {
    mode_pending_ = requested;
    mode_pending_cycles_ = 0;
  }
  if (mode_pending_cycles_ >= p_.mode_switch_delay_cycles) {
    mode_active_ = requested;
    mode_pending_cycles_ = 0;
    have_prev_target_ = false;
    following_error_time_ = 0.0;
  } else {
    ++mode_pending_cycles_;
  }
}

double RmdxMockSlave::braking_torque()
{
  // Quick-stop / fault-reaction ramp, modelled as the position loop holding
  // where the ramp started.
  const double qm = q_ + encoder_offset_;
  if (!hold_set_) {
    hold_ = qm;
    hold_set_ = true;
  }
  return p_.csp_kp * (hold_ - qm) - p_.csp_kd * qd_;
}

bool RmdxMockSlave::stopped() const
{
  return state_cycles_ >= 2 && std::fabs(qd_) < p_.stopped_velocity;
}

double RmdxMockSlave::drive_torque()
{
  const double nmpm = nm_per_permille();
  const double qm = q_ + encoder_offset_;
  double tau = 0.0;
  switch (state_) {
    case State::kOperationEnabled: {
        const double off = p_.supports_torque_offset ? rx_.torque_offset * nmpm : 0.0;
        if (mode_active_ == cia402::kModeCsp) {
          const double qt = rx_.target_position * rad_per_count();
          if (!have_prev_target_) {
            vt_filtered_ = 0.0;
          } else {
            const double raw = (qt - prev_target_) / p_.dt;
            vt_filtered_ += (raw - vt_filtered_) * p_.dt / (p_.velocity_feedforward_filter_s + p_.dt);
          }
          const double vt = p_.csp_velocity_feedforward * vt_filtered_;
          prev_target_ = qt;
          have_prev_target_ = true;
          tau = p_.csp_kp * (qt - qm) + p_.csp_kd * (vt - qd_) + off;
        } else if (mode_active_ == cia402::kModeCsv) {
          tau = p_.csv_kv * (rx_.target_velocity * rad_per_count() - qd_) + off;
        } else if (mode_active_ == cia402::kModeCst) {
          tau = rx_.target_torque * nmpm + (p_.torque_offset_in_cst ? off : 0.0);
        }
        break;
      }
    case State::kQuickStopActive:
      if (p_.quick_stop_option == 0) {
        set_state(State::kSwitchOnDisabled);
        break;
      }
      tau = braking_torque();
      if (p_.quick_stop_option <= 4 && stopped()) {
        set_state(State::kSwitchOnDisabled);
        tau = 0.0;
      }
      break;
    case State::kFaultReactionActive:
      if (p_.fault_reaction_option == 0) {
        set_state(State::kFault);
        break;
      }
      tau = braking_torque();
      if (stopped()) {
        set_state(State::kFault);
        tau = 0.0;
      }
      break;
    default:
      break;
  }
  // 0x6072 caps the torque in every mode (CiA402), and the motor its peak. As on
  // the RMD-X (EtherCAT manual), 0x6072 = 0 means the stall-current limit, not 0.
  const double lim = rx_.max_torque == 0 ? p_.peak_torque_nm :
    std::abs(static_cast<double>(rx_.max_torque)) * nmpm;
  const double cap = std::min(lim, p_.peak_torque_nm);
  return std::clamp(tau, -cap, cap);
}

void RmdxMockSlave::step_joint(double tau_motor)
{
  tau_ = tau_motor;
  const bool driving = state_ == State::kOperationEnabled || state_ == State::kQuickStopActive ||
    state_ == State::kFaultReactionActive;
  if (blocked_ || (p_.holding_brake && !driving)) {
    qd_ = 0.0;
    return;
  }
  const double qdd = (tau_motor - load_ - p_.viscous * qd_) / p_.inertia;
  qd_ += qdd * p_.dt;
  q_ += qd_ * p_.dt;
}

TxPdo RmdxMockSlave::latch() const
{
  TxPdo tx;
  uint16_t sw = cia402::state_bits(state_);
  if (state_ != State::kNotReadyToSwitchOn && state_ != State::kStart) {
    sw |= cia402::kSwVoltageEnabled | cia402::kSwRemote;
  }
  if (state_ == State::kOperationEnabled &&
    (mode_active_ == cia402::kModeCsp || mode_active_ == cia402::kModeCst))
  {
    sw |= cia402::kSwFollowingCommand;
  }
  if (following_error_flag_) {sw |= cia402::kSwFollowingError;}
  tx.status_word = sw;
  tx.position_actual = saturate_round<int32_t>((q_ + encoder_offset_) / rad_per_count());
  tx.velocity_actual = saturate_round<int32_t>(qd_ / rad_per_count());
  tx.torque_actual = saturate_round<int16_t>(tau_ / nm_per_permille());
  tx.error_code = error_code_;
  tx.mode_display = mode_active_;
  return tx;
}

TxPdo RmdxMockSlave::cycle(const RxPdo * rx)
{
  const TxPdo out = latched_;
  ++cycle_count_;
  ++state_cycles_;

  if (rx) {
    rx_ = *rx;
    missed_time_ = 0.0;
  } else {
    missed_time_ += p_.dt;
    const bool powered = state_ == State::kReadyToSwitchOn || state_ == State::kSwitchedOn ||
      state_ == State::kOperationEnabled || state_ == State::kQuickStopActive;
    if (missed_time_ > p_.sm_watchdog && powered) {
      raise_fault(cia402::kErrHeartbeat);
    }
  }

  if (state_ == State::kNotReadyToSwitchOn) {
    if (cycle_count_ >= p_.boot_cycles) {set_state(State::kSwitchOnDisabled);}
  } else if (rx) {
    apply_control_word(rx_.control_word);
  }
  if (rx) {update_mode(rx_.mode_of_operation);}

  // following error supervision (CSP only)
  if (state_ == State::kOperationEnabled && mode_active_ == cia402::kModeCsp) {
    const double err = rx_.target_position * rad_per_count() - (q_ + encoder_offset_);
    if (std::fabs(err) > p_.following_error_window) {
      following_error_time_ += p_.dt;
      if (following_error_time_ > p_.following_error_timeout) {
        following_error_flag_ = true;
        raise_fault(cia402::kErrFollowing);
      }
    } else {
      following_error_time_ = 0.0;
    }
  }

  step_joint(drive_torque());
  latched_ = latch();
  return out;
}

}  // namespace mock
}  // namespace wisco_ethercat_ros2_hw
