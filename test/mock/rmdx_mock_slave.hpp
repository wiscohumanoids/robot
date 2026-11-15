// A MyActuator RMD-X EtherCAT drive as a CiA402 slave, for tests. Test-only.
//
// It owns the drive side of the bus: the 0x6040/0x6041 state machine, the
// 0x6060/0x6061 mode, a CSP position loop and CST torque path, the 0x6072
// torque cap, fault detection (following error, process-data watchdog,
// injected faults) and a one-joint model that produces 0x6064/0x606C/0x6077.
//
// Timing follows a DC-synchronous slave: cycle() returns the inputs latched
// at the previous cycle (what the frame picks up), then applies the new
// outputs and steps the joint by dt. So the master sees every reaction one
// cycle late, as on the bus.
//
// Everything here is the CiA402 profile plus guesses where the RMD-X manual
// is silent; the guesses are the fields of RmdxParams marked "guess".
#ifndef WISCO_ETHERCAT_ROS2_HW__MOCK__RMDX_MOCK_SLAVE_HPP_
#define WISCO_ETHERCAT_ROS2_HW__MOCK__RMDX_MOCK_SLAVE_HPP_

#include <cstdint>
#include <vector>

#include "cia402.hpp"

namespace wisco_ethercat_ros2_hw
{
namespace mock
{

// RxPDO 0x1600 (master -> drive), raw object values.
struct RxPdo
{
  uint16_t control_word = 0;     // 0x6040
  int32_t target_position = 0;   // 0x607A, counts
  int32_t target_velocity = 0;   // 0x60FF, counts/s
  int16_t target_torque = 0;     // 0x6071, per mille of rated torque
  int16_t torque_offset = 0;     // 0x60B2, per mille
  int16_t max_torque = 1000;     // 0x6072, per mille
  int8_t mode_of_operation = 0;  // 0x6060
};

// TxPDO 0x1A00 (drive -> master).
struct TxPdo
{
  uint16_t status_word = 0;      // 0x6041
  int32_t position_actual = 0;   // 0x6064, counts
  int32_t velocity_actual = 0;   // 0x606C, counts/s
  int16_t torque_actual = 0;     // 0x6077, per mille
  uint16_t error_code = 0;       // 0x603F
  int8_t mode_display = 0;       // 0x6061
};

struct RmdxParams
{
  double dt = 0.001;                  // bus cycle, s
  double counts_per_rev = 131072.0;   // 0x6064 counts per output revolution
  double rated_torque_nm = 15.0;      // 1000 per mille (x4_10: 0.015 Nm/per-mille)
  double peak_torque_nm = 10.0;       // the motor cannot exceed this whatever 0x6072 says

  // joint model
  double inertia = 0.05;              // kg m^2 at the output
  double viscous = 0.02;              // Nm s/rad joint friction
  double initial_position = 0.0;      // rad

  // drive loops (guess: the RMD-X CSP gains are set over SDO, values unknown)
  double csp_kp = 300.0;              // Nm/rad
  double csp_kd = 6.0;                // Nm s/rad
  double csp_velocity_feedforward = 1.0;  // gain on d(0x607A)/dt in the CSP loop (guess)
  double velocity_feedforward_filter_s = 0.003;  // its low-pass; 0x607A moves in whole counts (guess)
  double csv_kv = 5.0;                // Nm s/rad
  // quick-stop / fault-reaction ramp: the CSP loop holding where it started
  double stopped_velocity = 0.02;     // rad/s, "stopped" for the ramps

  // supervision
  double following_error_window = 0.2;  // rad (0x6065, guess)
  double following_error_timeout = 0.01;  // s (0x6066, guess)
  double sm_watchdog = 0.1;           // s without process data -> fault (guess: ESC default 100 ms)

  // option codes
  int quick_stop_option = 2;          // 0x605A: 1/2 stop then SWITCH_ON_DISABLED, 5/6 stay (guess)
  int fault_reaction_option = 0;      // 0x605E: 0 torque off at once, 1/2 brake first (guess)

  int boot_cycles = 2;                // NOT_READY_TO_SWITCH_ON -> SWITCH_ON_DISABLED after this
  int mode_switch_delay_cycles = 0;   // cycles between a new 0x6060 and the drive acting on it (guess)
  bool supports_torque_offset = true; // whether 0x60B2 is mapped and applied (unconfirmed on RMD-X)
  bool torque_offset_in_cst = true;   // CiA402 adds 0x60B2 to 0x6071 in CST too (unconfirmed on RMD-X)
  bool holding_brake = false;         // power-off brake holds the joint when not enabled (guess:
                                      // depends on the RMD-X variant and its brake output config)
};

class RmdxMockSlave
{
public:
  explicit RmdxMockSlave(const RmdxParams & params = RmdxParams());

  // One bus cycle. rx is null when the frame was lost (the drive keeps its
  // last outputs and the process-data watchdog runs). Returns the inputs
  // latched before this cycle's outputs were applied.
  TxPdo cycle(const RxPdo * rx);

  // ---- fault injection ----
  // Raise a fault now. persistent: the cause stays until clear_fault_cause(),
  // so fault resets are refused until then.
  void inject_fault(uint16_t error_code = cia402::kErrOverCurrent, bool persistent = false);
  void clear_fault_cause();
  // The joint stops moving (a collision or a seized gearbox), which a moving
  // CSP target turns into a following error.
  void set_blocked(bool blocked) {blocked_ = blocked;}
  // A position step in the encoder reading (slip, bad zero), which a CSP hold
  // turns into a following error at once.
  void inject_position_jump(double rad) {encoder_offset_ += rad;}

  // ---- environment ----
  // External torque on the joint (gravity of the links it carries), Nm, as
  // the holding torque: the motor must produce +load to stay still.
  void set_load_torque(double nm) {load_ = nm;}

  // ---- observation (ground truth, not over the bus) ----
  cia402::State state() const {return state_;}
  int8_t active_mode() const {return mode_active_;}
  double position() const {return q_;}            // rad, true joint position
  double velocity() const {return qd_;}
  double applied_torque() const {return tau_;}     // Nm the motor produced last cycle
  uint16_t error_code() const {return error_code_;}
  const RxPdo & last_rx() const {return rx_;}
  const RmdxParams & params() const {return p_;}
  RmdxParams & params() {return p_;}
  int64_t cycles() const {return cycle_count_;}
  bool following_error() const {return following_error_flag_;}
  // every state the drive passed through, in order, de-duplicated
  const std::vector<cia402::State> & state_history() const {return history_;}

private:
  void set_state(cia402::State s);
  void raise_fault(uint16_t code);
  void apply_control_word(uint16_t cw);
  void update_mode(int8_t requested);
  double drive_torque();
  double braking_torque();
  bool stopped() const;
  void step_joint(double tau_motor);
  TxPdo latch() const;

  double rad_per_count() const;
  double nm_per_permille() const {return p_.rated_torque_nm / 1000.0;}

  RmdxParams p_;
  RxPdo rx_;
  cia402::State state_ = cia402::State::kNotReadyToSwitchOn;
  std::vector<cia402::State> history_;
  int64_t cycle_count_ = 0;
  uint16_t last_cw_ = 0;
  uint16_t error_code_ = 0;
  bool fault_cause_active_ = false;
  bool following_error_flag_ = false;
  double following_error_time_ = 0.0;
  double missed_time_ = 0.0;

  int8_t mode_active_ = cia402::kModeNone;
  int8_t mode_pending_ = cia402::kModeNone;
  int mode_pending_cycles_ = 0;

  double q_ = 0.0, qd_ = 0.0, tau_ = 0.0, load_ = 0.0;
  double encoder_offset_ = 0.0;
  bool blocked_ = false;
  double prev_target_ = 0.0;
  bool have_prev_target_ = false;
  double vt_filtered_ = 0.0;
  double hold_ = 0.0;
  bool hold_set_ = false;
  int state_cycles_ = 0;
  TxPdo latched_;
};

}  // namespace mock
}  // namespace wisco_ethercat_ros2_hw

#endif  // WISCO_ETHERCAT_ROS2_HW__MOCK__RMDX_MOCK_SLAVE_HPP_
