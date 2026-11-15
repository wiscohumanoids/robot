// The mock RMD-X drive (test/mock/rmdx_mock_slave.*) on its own and under the
// mirrored master-side CiA402 logic (test/mock/cia402_master.hpp): enable
// sequencing, the strict CiA402 state machine, quick stop, fault reaction
// and reset, CSP <-> CST, following error, 0x6072 and lost frames.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "cia402_master.hpp"
#include "rmdx_mock_slave.hpp"

using wisco_ethercat_ros2_hw::mock::Cia402DriveMaster;
using wisco_ethercat_ros2_hw::mock::DriveCommands;
using wisco_ethercat_ros2_hw::mock::DriveScaling;
using wisco_ethercat_ros2_hw::mock::DriveStates;
using wisco_ethercat_ros2_hw::mock::RmdxMockSlave;
using wisco_ethercat_ros2_hw::mock::RmdxParams;
using wisco_ethercat_ros2_hw::mock::RxPdo;
using wisco_ethercat_ros2_hw::mock::TxPdo;
namespace cia402 = wisco_ethercat_ros2_hw::mock::cia402;
using cia402::State;

namespace
{

// One drive on the bus with the master logic, in EthercatDriver's order:
// read (processData) then write (exchange).
struct Bus
{
  explicit Bus(const RmdxParams & p = RmdxParams())
  : slave(p)
  {
    master.scaling = DriveScaling::from_joint(p.counts_per_rev, p.rated_torque_nm / 1000.0, 1, 0.0);
  }

  void cycle(bool drop = false)
  {
    master.process(have ? &tx : nullptr, cmd, st);
    if (drop) {
      slave.cycle(nullptr);
      have = false;
    } else {
      tx = slave.cycle(&master.rx());
      have = true;
      if (cw_seen.empty() || cw_seen.back() != master.rx().control_word) {
        cw_seen.push_back(master.rx().control_word);
      }
    }
  }

  // cycles until the drive reports operation enabled to the master, or -1
  int enable(int max_cycles = 50)
  {
    for (int k = 1; k <= max_cycles; ++k) {
      cycle();
      if (cia402::operation_enabled(static_cast<uint16_t>(st.status_word))) {return k;}
    }
    return -1;
  }

  RmdxMockSlave slave;
  Cia402DriveMaster master;
  DriveCommands cmd;
  DriveStates st;
  TxPdo tx;
  bool have = false;
  std::vector<uint16_t> cw_seen;
};

RmdxParams params()
{
  RmdxParams p;
  p.rated_torque_nm = 15.0;   // 0.015 Nm per per-mille, like the x4_10
  p.peak_torque_nm = 10.0;
  p.inertia = 0.05;
  return p;
}

// Drive the slave directly with a control word (no master logic).
TxPdo send(RmdxMockSlave & s, uint16_t cw, int8_t mode = cia402::kModeCst, int16_t torque = 0)
{
  RxPdo rx;
  rx.control_word = cw;
  rx.mode_of_operation = mode;
  rx.target_torque = torque;
  rx.max_torque = 1000;
  rx.target_position = static_cast<int32_t>(std::lround(s.position() / (2 * M_PI / s.params().counts_per_rev)));
  return s.cycle(&rx);
}

void boot(RmdxMockSlave & s)
{
  for (int k = 0; k < 5; ++k) {send(s, 0x0000);}
  ASSERT_EQ(s.state(), State::kSwitchOnDisabled);
}

}  // namespace

TEST(MockSlave, MasterEnablesDriveThroughEveryState)
{
  Bus bus(params());
  bus.slave.set_load_torque(-1.0);   // gravity on the joint while it enables
  const int cycles = bus.enable();
  ASSERT_GT(cycles, 0) << "drive never reached OPERATION_ENABLED";
  EXPECT_LE(cycles, 20);

  const std::vector<State> expected = {
    State::kNotReadyToSwitchOn, State::kSwitchOnDisabled, State::kReadyToSwitchOn,
    State::kSwitchedOn, State::kOperationEnabled};
  EXPECT_EQ(bus.slave.state_history(), expected);
  // shutdown, switch on, enable operation, nothing else
  const std::vector<uint16_t> cws = {0x0000, 0x0006, 0x0007, 0x000F};
  EXPECT_EQ(bus.cw_seen, cws);
  // 0x1237: operation enabled, voltage, remote, following the CSP command
  EXPECT_EQ(static_cast<uint16_t>(bus.st.status_word), 0x1237);
  EXPECT_EQ(bus.st.mode_of_operation, cia402::kModeCsp);

  // Enabled in CSP with nothing written to the position interface (NaN), the
  // master re-sends the last actual position as 0x607A every cycle. That
  // target lags the joint by the bus delay, so the CSP loop has no stiffness
  // and a loaded joint creeps (terminal speed ~ load / (kp * delay)). On the
  // arm this is the gap between the drives enabling and the ArmController
  // activating; hold the brake or keep it short.
  for (int k = 0; k < 300; ++k) {bus.cycle();}
  EXPECT_EQ(bus.slave.state(), State::kOperationEnabled);
  EXPECT_GT(std::fabs(bus.slave.position()), 0.05);

  // a held target holds, with only the loop's static sag (load / kp)
  bus.cmd.position = bus.st.position;
  for (int k = 0; k < 500; ++k) {bus.cycle();}
  // holding torque -1 Nm: kp (target - q) = -1 -> q = target + 1 / kp
  EXPECT_NEAR(bus.slave.position(), bus.cmd.position + 1.0 / bus.slave.params().csp_kp, 2e-3);
  EXPECT_LT(std::fabs(bus.slave.velocity()), 1e-3);
}

TEST(MockSlave, StateMachineRejectsSkippedTransitions)
{
  RmdxMockSlave s(params());
  boot(s);
  send(s, cia402::kCwEnableOperation);           // 0x0F straight from SWITCH_ON_DISABLED
  EXPECT_EQ(s.state(), State::kSwitchOnDisabled);
  send(s, cia402::kCwShutdown);
  EXPECT_EQ(s.state(), State::kReadyToSwitchOn);
  send(s, cia402::kCwEnableOperation);           // only switches on (transition 3)
  EXPECT_EQ(s.state(), State::kSwitchedOn);
  send(s, cia402::kCwEnableOperation);
  EXPECT_EQ(s.state(), State::kOperationEnabled);
  send(s, cia402::kCwDisableOperation);          // 5
  EXPECT_EQ(s.state(), State::kSwitchedOn);
  send(s, cia402::kCwEnableOperation);
  send(s, cia402::kCwShutdown);                  // 8
  EXPECT_EQ(s.state(), State::kReadyToSwitchOn);
  send(s, cia402::kCwDisableVoltage);            // 7
  EXPECT_EQ(s.state(), State::kSwitchOnDisabled);
  // a fault reset edge outside FAULT does nothing
  send(s, cia402::kCwFaultReset);
  EXPECT_EQ(s.state(), State::kSwitchOnDisabled);
  // no torque in any of these unless operation is enabled
  send(s, cia402::kCwShutdown, cia402::kModeCst, 500);
  EXPECT_EQ(s.applied_torque(), 0.0);
}

TEST(MockSlave, QuickStopBrakesThenDisablesOrHolds)
{
  for (int option : {2, 6}) {
    RmdxParams p = params();
    p.quick_stop_option = option;
    RmdxMockSlave s(p);
    boot(s);
    send(s, cia402::kCwShutdown);
    send(s, cia402::kCwSwitchOn);
    send(s, cia402::kCwEnableOperation);
    ASSERT_EQ(s.state(), State::kOperationEnabled);
    for (int k = 0; k < 100; ++k) {send(s, cia402::kCwEnableOperation, cia402::kModeCst, 100);}
    const double v0 = s.velocity();
    ASSERT_GT(v0, 1.0);

    send(s, cia402::kCwQuickStop);
    EXPECT_EQ(s.state(), State::kQuickStopActive);
    EXPECT_LT(s.applied_torque(), 0.0) << "quick stop must brake";
    for (int k = 0; k < 500 && s.state() == State::kQuickStopActive; ++k) {
      send(s, cia402::kCwQuickStop);
    }
    EXPECT_LT(std::fabs(s.velocity()), 0.05);
    if (option == 2) {
      EXPECT_EQ(s.state(), State::kSwitchOnDisabled);
    } else {
      EXPECT_EQ(s.state(), State::kQuickStopActive);
      send(s, cia402::kCwEnableOperation);       // 16
      EXPECT_EQ(s.state(), State::kOperationEnabled);
    }
  }
}

TEST(MockSlave, FaultReactionThenResetThroughMaster)
{
  Bus bus(params());
  ASSERT_GT(bus.enable(), 0);
  bus.cmd.mode_of_operation = cia402::kModeCst;
  bus.cmd.effort = 1.0;
  for (int k = 0; k < 20; ++k) {bus.cycle();}
  ASSERT_NEAR(bus.slave.applied_torque(), 1.0, 0.02);

  bus.slave.inject_fault(cia402::kErrOverCurrent);
  bus.cycle();
  // fault reaction 0 (torque off at once), then FAULT
  EXPECT_EQ(bus.slave.state(), State::kFault);
  EXPECT_EQ(bus.slave.applied_torque(), 0.0);
  for (int k = 0; k < 5; ++k) {bus.cycle();}
  EXPECT_EQ(bus.master.state(), State::kFault);
  EXPECT_EQ(static_cast<uint16_t>(bus.st.error_code), cia402::kErrOverCurrent);

  // no auto_fault_reset: it stays faulted while effort is still commanded
  for (int k = 0; k < 50; ++k) {bus.cycle();}
  EXPECT_EQ(bus.slave.state(), State::kFault);

  // rising edge on reset_fault -> 0x80 -> SWITCH_ON_DISABLED -> re-enabled
  bus.cmd.reset_fault = 0.0;
  bus.cycle();
  bus.cmd.reset_fault = 1.0;
  int k = 0;
  for (; k < 50 && !cia402::operation_enabled(static_cast<uint16_t>(bus.st.status_word)); ++k) {
    bus.cycle();
  }
  EXPECT_LT(k, 50);
  EXPECT_EQ(bus.slave.error_code(), 0);
  bool saw_reset = false;
  for (auto cw : bus.cw_seen) {saw_reset = saw_reset || (cw & 0x80);}
  EXPECT_TRUE(saw_reset);
  const auto & h = bus.slave.state_history();
  const std::vector<State> tail(h.end() - 6, h.end());
  const std::vector<State> expected = {
    State::kFaultReactionActive, State::kFault, State::kSwitchOnDisabled,
    State::kReadyToSwitchOn, State::kSwitchedOn, State::kOperationEnabled};
  EXPECT_EQ(tail, expected);
}

TEST(MockSlave, PersistentFaultRefusesResetAndBit7Latches)
{
  Bus bus(params());
  ASSERT_GT(bus.enable(), 0);
  bus.slave.inject_fault(cia402::kErrOverCurrent, true);
  for (int k = 0; k < 5; ++k) {bus.cycle();}
  bus.cmd.reset_fault = 0.0;
  bus.cycle();
  bus.cmd.reset_fault = 1.0;
  for (int k = 0; k < 20; ++k) {bus.cycle();}
  EXPECT_EQ(bus.slave.state(), State::kFault) << "reset must be refused while the cause is active";

  // Cause gone, a second reset edge on the interface. The transcribed upstream
  // transition() ORs 0x80 into the control word it reads back and never
  // clears it while in FAULT, so the drive sees no new edge and stays
  // faulted. If upstream really does this, a failed reset needs a driver
  // restart; check against the installed ethercat_generic_plugins.
  bus.slave.clear_fault_cause();
  bus.cmd.reset_fault = 0.0;
  for (int k = 0; k < 3; ++k) {bus.cycle();}
  bus.cmd.reset_fault = 1.0;
  for (int k = 0; k < 20; ++k) {bus.cycle();}
  EXPECT_TRUE(bus.master.rx().control_word & 0x80);
  EXPECT_EQ(bus.slave.state(), State::kFault);
}

TEST(MockSlave, ModeSwitchFollowsAfterDelayAndIgnoresUnsupported)
{
  RmdxParams p = params();
  p.mode_switch_delay_cycles = 2;
  RmdxMockSlave s(p);
  boot(s);
  send(s, cia402::kCwShutdown, cia402::kModeCsp);
  send(s, cia402::kCwSwitchOn, cia402::kModeCsp);
  send(s, cia402::kCwEnableOperation, cia402::kModeCsp);
  for (int k = 0; k < 5; ++k) {send(s, cia402::kCwEnableOperation, cia402::kModeCsp);}
  ASSERT_EQ(s.active_mode(), cia402::kModeCsp);

  send(s, cia402::kCwEnableOperation, cia402::kModeCst);
  EXPECT_EQ(s.active_mode(), cia402::kModeCsp);
  send(s, cia402::kCwEnableOperation, cia402::kModeCst);
  EXPECT_EQ(s.active_mode(), cia402::kModeCsp);
  send(s, cia402::kCwEnableOperation, cia402::kModeCst);
  EXPECT_EQ(s.active_mode(), cia402::kModeCst);
  const TxPdo tx = send(s, cia402::kCwEnableOperation, cia402::kModeCst);
  EXPECT_EQ(tx.mode_display, cia402::kModeCst);
  EXPECT_TRUE(tx.status_word & cia402::kSwFollowingCommand);

  for (int k = 0; k < 5; ++k) {send(s, cia402::kCwEnableOperation, 1);}   // profile position
  EXPECT_EQ(s.active_mode(), cia402::kModeCst);
}

TEST(MockSlave, FollowingErrorFaults)
{
  Bus bus(params());
  ASSERT_GT(bus.enable(), 0);
  // a seized joint and a CSP target moving away
  bus.slave.set_blocked(true);
  bus.cmd.mode_of_operation = cia402::kModeCsp;
  double target = 0.0;
  int k = 0;
  for (; k < 1000 && bus.slave.state() == State::kOperationEnabled; ++k) {
    target += 0.002;   // 2 rad/s
    bus.cmd.position = target;
    bus.cycle();
  }
  EXPECT_EQ(bus.slave.error_code(), cia402::kErrFollowing);
  EXPECT_TRUE(bus.slave.following_error());
  for (int j = 0; j < 4; ++j) {bus.cycle();}
  EXPECT_TRUE(static_cast<uint16_t>(bus.st.status_word) & cia402::kSwFollowingError);
  EXPECT_EQ(bus.master.state(), State::kFault);
  // the window is 0.2 rad, the timeout 10 ms: tripped between 0.2 and 0.25 rad
  EXPECT_GT(target, 0.2);
  EXPECT_LT(target, 0.26);

  // an encoder jump under a held CSP target trips it too
  Bus bus2(params());
  ASSERT_GT(bus2.enable(), 0);
  bus2.cmd.position = 0.0;
  for (int j = 0; j < 5; ++j) {bus2.cycle();}
  bus2.slave.inject_position_jump(0.5);
  for (int j = 0; j < 9; ++j) {bus2.cycle();}
  EXPECT_EQ(bus2.slave.state(), State::kOperationEnabled) << "tripped before the 10 ms timeout";
  for (int j = 0; j < 5; ++j) {bus2.cycle();}
  EXPECT_EQ(bus2.slave.error_code(), cia402::kErrFollowing);
}

TEST(MockSlave, MaxTorqueCapsEveryMode)
{
  Bus bus(params());
  bus.master.max_torque_permille = 100;   // 1.5 Nm
  ASSERT_GT(bus.enable(), 0);
  bus.cmd.mode_of_operation = cia402::kModeCst;
  bus.cmd.effort = 5.0;
  for (int k = 0; k < 10; ++k) {bus.cycle();}
  EXPECT_NEAR(bus.slave.applied_torque(), 1.5, 1e-9);
  EXPECT_NEAR(bus.st.effort, 1.5, 0.015);   // 0x6077 reports what was applied

  bus.cmd.effort = -5.0;
  bus.cmd.torque_offset = -5.0;   // 0x60B2 is added, and capped with it
  for (int k = 0; k < 10; ++k) {bus.cycle();}
  EXPECT_NEAR(bus.slave.applied_torque(), -1.5, 1e-9);

  // CSP: a 1 rad step would ask the drive's loop for 300 Nm
  bus.cmd.mode_of_operation = cia402::kModeCsp;
  bus.cmd.torque_offset = 0.0;
  bus.cmd.position = 1.0;
  for (int k = 0; k < 5; ++k) {
    bus.cycle();
    EXPECT_LE(std::fabs(bus.slave.applied_torque()), 1.5 + 1e-9);
  }
  // the motor's peak wins over a 0x6072 above it
  bus.master.max_torque_permille = 1000;   // 15 Nm > 10 Nm peak
  for (int k = 0; k < 3; ++k) {bus.cycle();}
  EXPECT_LE(std::fabs(bus.slave.applied_torque()), 10.0 + 1e-9);
}

TEST(MockSlave, LostFramesKeepLastOutputsThenTripWatchdog)
{
  Bus bus(params());
  ASSERT_GT(bus.enable(), 0);
  bus.cmd.mode_of_operation = cia402::kModeCst;
  bus.cmd.effort = 0.5;
  for (int k = 0; k < 10; ++k) {bus.cycle();}

  // 50 ms of lost frames: the drive keeps applying the last torque, the
  // master's view is frozen, no fault (SM watchdog 100 ms)
  bus.cycle(true);   // processes the last frame that did arrive
  const double seen = bus.st.position;
  for (int k = 0; k < 49; ++k) {bus.cycle(true);}
  EXPECT_EQ(bus.slave.state(), State::kOperationEnabled);
  EXPECT_NEAR(bus.slave.applied_torque(), 0.5, 0.02);
  EXPECT_GT(std::fabs(bus.slave.position() - seen), 0.01);   // the joint moved on
  EXPECT_DOUBLE_EQ(bus.st.position, seen);                    // the master did not see it

  // frames back: no fault, the drive follows again
  bus.cmd.effort = 0.0;
  for (int k = 0; k < 5; ++k) {bus.cycle();}
  EXPECT_EQ(bus.slave.state(), State::kOperationEnabled);
  EXPECT_NEAR(bus.slave.applied_torque(), 0.0, 0.02);

  // 150 ms lost: the drive faults (0x8130) and drops torque
  bus.cmd.effort = 0.5;
  for (int k = 0; k < 5; ++k) {bus.cycle();}
  for (int k = 0; k < 150; ++k) {bus.cycle(true);}
  EXPECT_EQ(bus.slave.state(), State::kFault);
  EXPECT_EQ(bus.slave.error_code(), cia402::kErrHeartbeat);
  EXPECT_EQ(bus.slave.applied_torque(), 0.0);
  for (int k = 0; k < 5; ++k) {bus.cycle();}
  EXPECT_EQ(bus.master.state(), State::kFault);
}
