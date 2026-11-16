// The RT core (ArmController and SafeCommandController, the real plugin
// code) against mock RMD-X drives on a mock EtherCAT ring
// (test/mock/: MockEthercatSystem, Cia402DriveMaster, RmdxMockSlave).
//
// The arm is a 2-link planar arm in a vertical plane, so gravity torques are
// known in closed form and the mock joints carry them as load.
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "controller_harness.hpp"
#include "rclcpp/rclcpp.hpp"
#include "wisco_ethercat_ros2_hw/arm_controller.hpp"
#include "wisco_ethercat_ros2_hw/safe_command_controller.hpp"
#include "wisco_ethercat_ros2_hw/safety_logic.hpp"

using wisco_ethercat_ros2_hw::ArmController;
using wisco_ethercat_ros2_hw::SafeCommandController;
using wisco_ethercat_ros2_hw::mock::ControllerHarness;
using wisco_ethercat_ros2_hw::mock::JointSpec;
namespace cia402 = wisco_ethercat_ros2_hw::mock::cia402;
namespace safety = wisco_ethercat_ros2_hw::safety;
using cia402::State;

namespace
{
const double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---- the test arm ----
constexpr double kM1 = 2.0, kL1c = 0.15, kL1 = 0.3, kM2 = 1.0, kL2c = 0.1, kG = 9.81;

const char * kUrdf = R"(<?xml version="1.0"?>
<robot name="mock_arm">
  <link name="base"/>
  <link name="link1">
    <inertial><origin xyz="0.15 0 0"/><mass value="2.0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial>
  </link>
  <link name="link2">
    <inertial><origin xyz="0.1 0 0"/><mass value="1.0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/><origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="60" velocity="5"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="link2"/><origin xyz="0.3 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="10" velocity="5"/>
  </joint>
</robot>)";

// Holding torque (Pinocchio's g(q)) for the arm above: z up, joints about +y,
// links along +x at q = 0.
void gravity(const std::vector<double> & q, std::vector<double> & g)
{
  const double c1 = std::cos(q[0]), c12 = std::cos(q[0] + q[1]);
  g[0] = -kG * ((kM1 * kL1c + kM2 * kL1) * c1 + kM2 * kL2c * c12);
  g[1] = -kG * kM2 * kL2c * c12;
}

std::vector<double> gravity_at(const std::vector<double> & q)
{
  std::vector<double> g(2);
  gravity(q, g);
  return g;
}

const std::vector<double> kKdFallback = {2.0, 0.6};
const std::vector<double> kMaxTorque = {15.0, 3.0};

std::vector<JointSpec> arm_joints()
{
  JointSpec j1;
  j1.name = "j1";
  j1.effort_nm_per_permille = 0.039375;   // x6_60
  j1.peak_effort_nm = 60.0;
  j1.max_torque_permille = 450;           // 17.7 Nm
  j1.inertia = 0.2;
  j1.initial_position = 0.2;
  JointSpec j2;
  j2.name = "j2";
  j2.effort_nm_per_permille = 0.015;      // x4_10
  j2.peak_effort_nm = 10.0;
  j2.max_torque_permille = 250;           // 3.75 Nm
  j2.inertia = 0.02;
  j2.initial_position = -0.3;
  return {j1, j2};
}

std::vector<rclcpp::Parameter> arm_params(bool drive_feedback, bool write_max_torque = false)
{
  return {
    rclcpp::Parameter("write_max_torque", write_max_torque),
    rclcpp::Parameter("joints", std::vector<std::string>{"j1", "j2"}),
    rclcpp::Parameter("max_torque", kMaxTorque),
    rclcpp::Parameter("max_velocity", std::vector<double>{2.0, 2.0}),
    rclcpp::Parameter("soft_lower", std::vector<double>{-1.5, -1.0}),
    rclcpp::Parameter("soft_upper", std::vector<double>{1.5, 0.5}),
    rclcpp::Parameter("kp", std::vector<double>{60.0, 20.0}),
    rclcpp::Parameter("kd", std::vector<double>{2.0, 0.6}),
    rclcpp::Parameter("kd_fallback", kKdFallback),
    rclcpp::Parameter("watchdog_s", 0.05),
    rclcpp::Parameter("soft_margin", 0.05),
    rclcpp::Parameter("blend_s", 0.02),
    rclcpp::Parameter("robot_description", std::string(kUrdf)),
    rclcpp::Parameter("drive_feedback", drive_feedback),
    rclcpp::Parameter("payload_frame", std::string("link2")),   // no tool0 in this URDF
  };
}

// ~/command: [mode, flags, friction_ff_gain, q[n], qd[n], kp[n], kd[n], tau_ff[n], clamp[n]]
std::vector<double> command(
  int mode, const std::vector<double> & q, const std::vector<double> & tau_ff = {0.0, 0.0},
  const std::vector<double> & clamp = {kNaN, kNaN},
  const std::vector<double> & kp = {60.0, 20.0}, const std::vector<double> & kd = {2.0, 0.6})
{
  std::vector<double> d = {static_cast<double>(mode), 1.0, 1.0};
  d.insert(d.end(), q.begin(), q.end());
  d.insert(d.end(), {0.0, 0.0});
  d.insert(d.end(), kp.begin(), kp.end());
  d.insert(d.end(), kd.begin(), kd.end());
  d.insert(d.end(), tau_ff.begin(), tau_ff.end());
  d.insert(d.end(), clamp.begin(), clamp.end());
  return d;
}

constexpr int kModeP = 0;
constexpr int kModeT = 1;

bool all_enabled(ControllerHarness & h)
{
  for (size_t i = 0; i < h.hw().size(); ++i) {
    if (!cia402::operation_enabled(static_cast<uint16_t>(h.hw().states(i).status_word))) {
      return false;
    }
  }
  return true;
}

struct Arm
{
  std::unique_ptr<ControllerHarness> h;
  std::shared_ptr<ArmController> c;
  double q(size_t i) {return h->hw().joint_position(i);}
  double tau(size_t i) {return h->hw().joint_torque(i);}
  const wisco_ethercat_ros2_hw::mock::DriveCommands & cmd(size_t i) {return h->hw().commands(i);}
  const wisco_ethercat_ros2_hw::mock::DriveStates & st(size_t i) {return h->hw().states(i);}
  std::vector<double> q_meas() {return {st(0).position, st(1).position};}
  std::vector<double> q_true() {return {q(0), q(1)};}
};

// Drives enabled under the master logic, then the controller loaded (the
// order the launch file produces). start_bus = false loads the controller
// before the first frame instead.
Arm make_arm(
  bool drive_feedback, int mode_switch_delay = 0, bool start_bus = true,
  bool holding_brake = false, bool write_max_torque = false)
{
  Arm a;
  a.h = std::make_unique<ControllerHarness>(
    arm_joints(), std::unordered_map<std::string, std::string>{{"mode_of_operation", "8"}});
  for (size_t i = 0; i < a.h->hw().size(); ++i) {
    a.h->hw().slave(i).params().mode_switch_delay_cycles = mode_switch_delay;
    a.h->hw().slave(i).params().holding_brake = holding_brake;
  }
  a.h->hw().set_load_model(gravity);
  if (start_bus) {
    for (int k = 0; k < 50 && !all_enabled(*a.h); ++k) {a.h->run_bus(1);}
    EXPECT_TRUE(all_enabled(*a.h));
  }
  a.c = std::make_shared<ArmController>();
  EXPECT_TRUE(a.h->load(a.c, "arm_controller", arm_params(drive_feedback, write_max_torque), "/command"));
  return a;
}

double max_step(const std::vector<double> & v, size_t * at = nullptr)
{
  double m = 0.0;
  for (size_t k = 1; k < v.size(); ++k) {
    if (std::fabs(v[k] - v[k - 1]) > m) {
      m = std::fabs(v[k] - v[k - 1]);
      if (at) {*at = k;}
    }
  }
  return m;
}

std::string window(const std::vector<double> & v, size_t at, size_t half = 4)
{
  std::string s;
  for (size_t k = at > half ? at - half : 0; k < std::min(v.size(), at + half); ++k) {
    s += std::to_string(k) + ":" + std::to_string(v[k]) + " ";
  }
  return s;
}

}  // namespace

class ArmMock : public ::testing::Test {};

TEST_F(ArmMock, CommissioningStartupAndTimeoutHoldWithoutGravity)
{
  ControllerHarness h(arm_joints());
  h.set_realtime(false);
  h.run_bus(50);
  auto params = arm_params(true, true);
  params.emplace_back("hold_on_startup", true);
  params.emplace_back("hold_on_timeout", true);
  params.emplace_back("gravity_enabled", false);
  ASSERT_TRUE(h.load(std::make_shared<ArmController>(), "arm_controller", params, "/command"));
  h.cycle(100);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(h.hw().commands(i).mode_of_operation, cia402::kModeCsp);
    EXPECT_DOUBLE_EQ(h.hw().commands(i).torque_offset, 0.0);
    EXPECT_DOUBLE_EQ(h.hw().commands(i).effort, 0.0);
  }
  h.stream(command(kModeP, {0.25, -0.25}));
  h.cycle(100);
  h.stop_stream();
  std::this_thread::sleep_for(std::chrono::milliseconds(60));
  h.cycle(50);
  EXPECT_EQ(h.hw().commands(0).mode_of_operation, cia402::kModeCsp);
  EXPECT_DOUBLE_EQ(h.hw().commands(0).torque_offset, 0.0);
  h.deactivate();
}

TEST_F(ArmMock, MixedFirmwareWithoutTorqueOffset)
{
  auto joints = arm_joints();
  joints[0].supports_torque_offset = false;
  for (const bool feedback : {false, true}) {
    ControllerHarness h(joints);
    h.set_realtime(false);
    h.hw().set_load_model(gravity);
    for (size_t i = 0; i < h.hw().size(); ++i) {
      h.hw().slave(i).params().mode_switch_delay_cycles = 8;
      h.hw().slave(i).params().holding_brake = true;
    }
    h.run_bus(50);
    ASSERT_TRUE(all_enabled(h));
    auto controller = std::make_shared<ArmController>();
    auto params = arm_params(feedback, true);
    params.emplace_back("torque_offset_enabled", std::vector<bool>{false, true});
    ASSERT_TRUE(h.load(controller, "arm_controller", params, "/command"));
    const auto claimed = controller->command_interface_configuration().names;
    EXPECT_EQ(std::count(claimed.begin(), claimed.end(), "j1/torque_offset"), 0);
    EXPECT_EQ(std::count(claimed.begin(), claimed.end(), "j2/torque_offset"), 1);

    const std::vector<double> target = {0.2, -0.3};
    // The initial watchdog fallback and active torque mode use 0x6071 on
    // the legacy drive, including while its mode switch is delayed.
    h.cycle(250);
    EXPECT_EQ(h.hw().states(0).mode_of_operation, cia402::kModeCst);
    EXPECT_LT(h.hw().commands(0).effort, -5.0);
    for (const int mode : {kModeP, kModeT, kModeP}) {
      h.stream(command(mode, target));
      double previous_target = h.hw().commands(0).position;
      for (int k = 0; k < 500; ++k) {
        h.cycle();
        EXPECT_TRUE(all_enabled(h));
        EXPECT_LE(std::fabs(h.hw().commands(0).effort), kMaxTorque[0]);
        EXPECT_TRUE(std::isnan(h.hw().commands(0).torque_offset));
        if (mode == kModeP && k > 1) {
          EXPECT_LE(std::fabs(h.hw().commands(0).position - previous_target), 0.0021);
        }
        previous_target = h.hw().commands(0).position;
      }
      EXPECT_NEAR(h.hw().joint_position(0), target[0], 0.06);
      EXPECT_NEAR(h.hw().joint_position(1), target[1], 0.03);
      EXPECT_EQ(h.hw().states(0).mode_of_operation,
        mode == kModeP ? cia402::kModeCsp : cia402::kModeCst);
      if (mode == kModeP) {
        EXPECT_LT(h.hw().commands(1).torque_offset, -0.5);
      }
    }
    // A smaller command clamp must also constrain the switching torque
    // retained in 0x6071 while the legacy drive is still in CST.
    h.stream(command(kModeP, target, {0.0, 0.0}, {1.0, 1.0}));
    h.cycle(1);
    EXPECT_LE(std::fabs(h.hw().commands(0).effort), 1.0);
    h.deactivate();
  }
}

// --------------------------------------------------------------------------
// Enable sequencing: the controller is active before the first frame; the
// drives walk SWITCH_ON_DISABLED -> READY -> SWITCHED_ON -> OPERATION_ENABLED
// under the master, no torque before that, then the fallback holds the arm.
// (Brakes on: an unbraked arm falls freely until the drives enable.)
TEST_F(ArmMock, EnableSequencingWithControllerActive)
{
  Arm a = make_arm(true, 0, false, true);
  std::vector<std::vector<double>> torque(2);
  int enabled_at = -1;
  for (int k = 0; k < 600; ++k) {
    a.h->cycle();
    const bool en = all_enabled(*a.h);
    if (en && enabled_at < 0) {enabled_at = k;}
    for (size_t i = 0; i < 2; ++i) {
      if (a.h->hw().slave(i).state() != State::kOperationEnabled) {
        EXPECT_EQ(a.tau(i), 0.0) << "torque before OPERATION_ENABLED, joint " << i;
      }
      torque[i].push_back(a.tau(i));
    }
  }
  ASSERT_GE(enabled_at, 0);
  EXPECT_LE(enabled_at, 25);
  const std::vector<State> expected = {
    State::kNotReadyToSwitchOn, State::kSwitchOnDisabled, State::kReadyToSwitchOn,
    State::kSwitchedOn, State::kOperationEnabled};
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(a.h->hw().slave(i).state_history(), expected) << "joint " << i;
    EXPECT_EQ(a.st(i).mode_of_operation, cia402::kModeCst) << "fallback runs in CST";
  }
  // on enable the drive goes straight to gravity comp (the step is g, no
  // more), so the brake hand-over does not drop the arm
  const auto g = gravity_at(a.q_true());
  EXPECT_LT(max_step(torque[0]), std::fabs(g[0]) + 0.2);
  EXPECT_LT(max_step(torque[1]), std::fabs(g[1]) + 0.1);
  for (size_t i = 0; i < 2; ++i) {EXPECT_NEAR(a.tau(i), g[i], 0.05);}
  // The fallback has no position hold: what is left is a slow creep from the
  // torque resolution (0x6071 counts are 0.039 / 0.015 Nm and are truncated)
  // against kd_fallback.
  EXPECT_NEAR(a.q(0), 0.2, 0.02);
  EXPECT_NEAR(a.q(1), -0.3, 0.02);
  EXPECT_LT(std::fabs(a.h->hw().joint_velocity(0)), 0.03);
}

// --------------------------------------------------------------------------
// 50 ms watchdog: commands stop -> still Mode P at 30 ms, gravity comp +
// damping in CST by 80 ms, exactly g(q) - kd_fallback * qd, and the arm holds.
TEST_F(ArmMock, WatchdogFallsBackToGravityCompAndDamping)
{
  Arm a = make_arm(false);
  const std::vector<double> q0 = {0.2, -0.3};
  a.h->stream(command(kModeP, q0));
  a.h->cycle(300);
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_EQ(a.cmd(i).mode_of_operation, cia402::kModeCsp);
    // Mode P: 0x60B2 carries the gravity feedforward
    EXPECT_NEAR(a.cmd(i).torque_offset, gravity_at(a.q_meas())[i], 1e-6);
    EXPECT_NEAR(a.q(i), q0[i], 0.01);
  }

  // The watchdog runs on the steady clock (command age), so measure wall time.
  using Clock = std::chrono::steady_clock;
  a.h->stop_stream();
  a.h->send(command(kModeP, q0));   // the last command
  const auto sent = Clock::now();
  double fell_back_ms = -1.0;
  for (int k = 0; k < 300 && fell_back_ms < 0; ++k) {
    a.h->cycle();
    const double age_ms = std::chrono::duration<double, std::milli>(Clock::now() - sent).count();
    if (a.cmd(0).mode_of_operation == cia402::kModeCst) {fell_back_ms = age_ms;}
  }
  EXPECT_GE(fell_back_ms, 50.0) << "fell back before the 50 ms watchdog";
  EXPECT_LE(fell_back_ms, 100.0);
  ASSERT_EQ(a.cmd(0).mode_of_operation, cia402::kModeCst);
  EXPECT_EQ(a.cmd(1).mode_of_operation, cia402::kModeCst);

  a.h->cycle(40);   // past the 20 ms blend
  const auto g = gravity_at(a.q_meas());
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NEAR(a.cmd(i).effort, g[i] - kKdFallback[i] * a.st(i).velocity, 1e-6);
    EXPECT_EQ(a.cmd(i).torque_offset, 0.0);
  }
  a.h->cycle(400);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NEAR(a.q(i), q0[i], 0.02) << "fallback must hold the arm, joint " << i;
  }
}

// --------------------------------------------------------------------------
// Payload: the optional trailing [kg, x, y, z] block (a point mass in
// payload_frame, link2 here) goes into g(q): the Mode P feedforward, and the
// watchdog fallback after the commands stop. Old-length commands keep it;
// kg = 0 clears it.
TEST_F(ArmMock, PayloadInGravityAndWatchdogFallback)
{
  constexpr double kMp = 0.6, kPx = 0.2;   // kg, m along link2's x
  auto with_payload = [](const std::vector<double> & q) {
      auto g = gravity_at(q);
      const double c1 = std::cos(q[0]), c12 = std::cos(q[0] + q[1]);
      g[0] += -kG * kMp * (kL1 * c1 + kPx * c12);
      g[1] += -kG * kMp * kPx * c12;
      return g;
    };
  Arm a = make_arm(false);
  a.h->hw().set_load_model([&](const std::vector<double> & q, std::vector<double> & g) {
      const auto w = with_payload(q);
      g[0] = w[0];
      g[1] = w[1];
    });
  const std::vector<double> q0 = {0.2, -0.3};
  auto loaded = command(kModeP, q0);
  loaded.insert(loaded.end(), {kMp, kPx, 0.0, 0.0});
  a.h->stream(loaded);
  a.h->cycle(300);
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_EQ(a.cmd(i).mode_of_operation, cia402::kModeCsp) << "payload command accepted";
    EXPECT_NEAR(a.cmd(i).torque_offset, with_payload(a.q_meas())[i], 1e-6);
    EXPECT_NEAR(a.q(i), q0[i], 0.01);
  }
  ASSERT_GT(std::fabs(with_payload(q0)[0] - gravity_at(q0)[0]), 1.0);

  // an old-length command leaves the payload in
  a.h->stream(command(kModeP, q0));
  a.h->cycle(50);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NEAR(a.cmd(i).torque_offset, with_payload(a.q_meas())[i], 1e-6);
  }

  // watchdog: commands stop -> gravity comp WITH the payload + damping, and it holds
  a.h->stop_stream();
  for (int k = 0; k < 2000 && a.cmd(0).mode_of_operation != cia402::kModeCst; ++k) {a.h->cycle();}
  ASSERT_EQ(a.cmd(0).mode_of_operation, cia402::kModeCst);
  a.h->cycle(40);   // past the 20 ms blend
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NEAR(a.cmd(i).effort, with_payload(a.q_meas())[i] - kKdFallback[i] * a.st(i).velocity, 1e-6);
  }
  a.h->cycle(400);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NEAR(a.q(i), q0[i], 0.02) << "fallback must hold the loaded arm, joint " << i;
  }

  // kg 0 clears it (the load is still there; check what is commanded)
  auto cleared = command(kModeP, q0);
  cleared.insert(cleared.end(), {0.0, 0.0, 0.0, 0.0});
  a.h->stream(cleared);
  a.h->cycle(100);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NEAR(a.cmd(i).torque_offset, gravity_at(a.q_meas())[i], 1e-6);
  }
}

// --------------------------------------------------------------------------
// CSP <-> CST: Mode P hold -> Mode T hold -> Mode P hold under gravity, no
// torque step at either switch. With drive feedback this holds even when the
// drive takes 3 cycles to act on 0x6060.
TEST_F(ArmMock, CspCstSwitchingWithoutTorqueJump)
{
  struct Case {bool feedback; int delay;};
  for (const Case & c : {Case{false, 0}, Case{true, 0}, Case{true, 3}}) {
    SCOPED_TRACE("drive_feedback " + std::to_string(c.feedback) + ", mode delay " +
      std::to_string(c.delay));
    Arm a = make_arm(c.feedback, c.delay);
    const std::vector<double> q0 = {0.2, -0.3};
    a.h->stream(command(kModeP, q0));
    a.h->cycle(400);
    std::vector<std::vector<double>> torque(2);
    auto run = [&](int n) {
        for (int k = 0; k < n; ++k) {
          a.h->cycle();
          for (size_t i = 0; i < 2; ++i) {torque[i].push_back(a.tau(i));}
        }
      };
    run(50);
    a.h->stream(command(kModeT, q0));
    run(200);
    for (size_t i = 0; i < 2; ++i) {EXPECT_EQ(a.st(i).mode_of_operation, cia402::kModeCst);}
    a.h->stream(command(kModeP, q0));
    run(200);
    for (size_t i = 0; i < 2; ++i) {EXPECT_EQ(a.st(i).mode_of_operation, cia402::kModeCsp);}

    // gravity is 6.9 / 1.0 Nm here; switches at samples 50 and 250
    size_t at0 = 0, at1 = 0;
    EXPECT_LT(max_step(torque[0], &at0), 0.5) << window(torque[0], at0);
    EXPECT_LT(max_step(torque[1], &at1), 0.15) << window(torque[1], at1);
    for (size_t i = 0; i < 2; ++i) {
      EXPECT_EQ(a.h->hw().slave(i).state(), State::kOperationEnabled);
      EXPECT_NEAR(a.q(i), q0[i], 0.01);
    }
  }
}

// What drive feedback is for: a drive that acts on 0x6060 three cycles late,
// without feedback, gets P -> T with the torque in 0x6071 while it still runs
// CSP (0x60B2 = 0): gravity comp drops out for those cycles.
TEST_F(ArmMock, DocumentsTorqueDipOnLateModeSwitchWithoutFeedback)
{
  Arm a = make_arm(false, 3);
  const std::vector<double> q0 = {0.2, -0.3};
  a.h->stream(command(kModeP, q0));
  a.h->cycle(400);
  std::vector<double> torque;
  a.h->stream(command(kModeT, q0));
  for (int k = 0; k < 50; ++k) {
    a.h->cycle();
    torque.push_back(a.tau(0));
  }
  EXPECT_GT(max_step(torque), 3.0);
}

// --------------------------------------------------------------------------
// Torque clamps: the controller's max_torque, the per-command torque_clamp,
// and the drive's 0x6072 under both, in Mode T and Mode P.
TEST_F(ArmMock, TorqueClampHonoured)
{
  Arm a = make_arm(false);
  a.h->hw().master(1).max_torque_permille = 100;   // j2 drive cap 1.5 Nm < controller's 3 Nm
  a.h->cycle(5);   // the new 0x6072 reaches the drive, and its 0x6077 the master
  const std::vector<double> q0 = {0.2, -0.3};

  // Mode T, j2 pulled 0.5 rad away with kp 20: asks for ~11 Nm
  a.h->stream(command(kModeT, {q0[0], q0[1] - 0.5}));
  for (int k = 0; k < 100; ++k) {
    a.h->cycle();
    EXPECT_LE(std::fabs(a.cmd(0).effort), kMaxTorque[0] + 1e-9);
    EXPECT_LE(std::fabs(a.cmd(1).effort), kMaxTorque[1] + 1e-9);
    EXPECT_LE(std::fabs(a.tau(1)), 1.5 + 1e-9) << "0x6072 not honoured";
    EXPECT_LE(std::fabs(a.st(1).effort), 1.5 + 0.015);
  }
  EXPECT_NEAR(a.cmd(1).effort, -kMaxTorque[1], 1e-9);   // saturated, not wrapped
  EXPECT_NEAR(a.tau(1), -1.5, 1e-9);

  // per-command clamp below both
  a.h->stream(command(kModeT, {q0[0], q0[1] - 0.5}, {0.0, 0.0}, {kNaN, 0.8}));
  a.h->cycle(30);
  for (int k = 0; k < 50; ++k) {
    a.h->cycle();
    EXPECT_LE(std::fabs(a.cmd(1).effort), 0.8 + 1e-9);
    EXPECT_LE(std::fabs(a.tau(1)), 0.8 + 0.015);
  }

  // Mode P: a 10 Nm feedforward is clamped in 0x60B2 too
  a.h->stream(command(kModeP, q0, {0.0, 10.0}));
  a.h->cycle(30);
  for (int k = 0; k < 50; ++k) {
    a.h->cycle();
    EXPECT_EQ(a.cmd(1).mode_of_operation, cia402::kModeCsp);
    EXPECT_LE(std::fabs(a.cmd(1).torque_offset), kMaxTorque[1] + 1e-9);
    EXPECT_LE(std::fabs(a.tau(1)), 1.5 + 1e-9);
  }
  a.h->stream(command(kModeP, q0, {0.0, 10.0}, {kNaN, 0.8}));
  a.h->cycle(5);
  EXPECT_NEAR(a.cmd(1).torque_offset, 0.8, 1e-9);
}

// --------------------------------------------------------------------------
// write_max_torque: the command's clamp goes to 0x6072, so in CSP the drive's
// own position loop (not just the 0x60B2 offset) is held to it. j2 holds
// ~0.98 Nm of gravity at q0; with a 0.3 Nm clamp the drive saturates at 0.3.
TEST_F(ArmMock, MaxTorqueFollowsCommandClampInCsp)
{
  Arm a = make_arm(false, 0, true, false, true);
  const std::vector<double> q0 = {0.2, -0.3};
  a.h->stream(command(kModeP, q0));
  a.h->cycle(100);
  EXPECT_NEAR(a.cmd(1).max_torque, kMaxTorque[1], 1e-9);        // no clamp: configured max
  EXPECT_NEAR(std::fabs(a.tau(1)), std::fabs(gravity_at(q0)[1]), 0.1);

  a.h->stream(command(kModeP, q0, {0.0, 0.0}, {kNaN, 0.3}));
  a.h->cycle(6);   // the new command reaches the controller, then the drive over the bus
  double peak = 0.0;
  for (int k = 0; k < 150; ++k) {
    a.h->cycle();
    peak = std::max(peak, std::fabs(a.tau(1)));
    EXPECT_LE(std::fabs(a.tau(1)), 0.3 + 0.015) << "cycle " << k;
  }
  EXPECT_NEAR(a.cmd(1).max_torque, 0.3, 1e-9);
  EXPECT_EQ(a.h->hw().master(1).rx().max_torque, 20);   // 0.3 Nm / 0.015 Nm per mille
  EXPECT_GT(peak, 0.28) << "the cap never bound";
  EXPECT_NEAR(a.cmd(0).max_torque, kMaxTorque[0], 1e-9);        // j1 unclamped

  // watchdog fallback: back to the configured max
  a.h->stop_stream();
  a.h->cycle(100);
  EXPECT_NEAR(a.cmd(1).max_torque, kMaxTorque[1], 1e-9);
}

// A zero torque clamp must not reach 0x6072 as 0, which the drive reads as
// "stall-current limit": the controller floors it at 1% of the configured max.
TEST_F(ArmMock, ZeroTorqueClampNeverWritesZeroMaxTorque)
{
  Arm a = make_arm(false, 0, true, false, true);
  const std::vector<double> q0 = {0.2, -0.3};
  a.h->stream(command(kModeP, q0));
  a.h->cycle(100);
  a.h->stream(command(kModeP, q0, {0.0, 0.0}, {kNaN, 0.0}));
  a.h->cycle(6);
  for (int k = 0; k < 100; ++k) {
    a.h->cycle();
    EXPECT_GE(a.h->hw().master(1).rx().max_torque, 1) << "cycle " << k;
    EXPECT_LE(std::fabs(a.tau(1)), 0.01 * kMaxTorque[1] + 0.015) << "cycle " << k;
  }
}

// --------------------------------------------------------------------------
// Safety-layer rejections in the ArmController: malformed commands, a
// missing Mode P target, targets past the soft limits, NaN gains.
TEST_F(ArmMock, ArmControllerRejectsUnsafeCommands)
{
  Arm a = make_arm(false);
  const std::vector<double> q0 = {0.2, -0.3};
  a.h->cycle(10);
  ASSERT_EQ(a.cmd(0).mode_of_operation, cia402::kModeCst);   // no command: fallback

  // wrong length: ignored, stays in the fallback
  auto bad = command(kModeP, q0);
  bad.pop_back();
  a.h->stream(bad);
  a.h->cycle(20);
  EXPECT_EQ(a.cmd(0).mode_of_operation, cia402::kModeCst);
  // a short payload block is the wrong length too
  bad = command(kModeP, q0);
  bad.insert(bad.end(), {0.5, 0.1});
  a.h->stream(bad);
  a.h->cycle(20);
  EXPECT_EQ(a.cmd(0).mode_of_operation, cia402::kModeCst);

  // Mode P without a target for j2: fallback, not a jump to 0
  a.h->stream(command(kModeP, {q0[0], kNaN}));
  a.h->cycle(20);
  EXPECT_EQ(a.cmd(0).mode_of_operation, cia402::kModeCst);
  EXPECT_EQ(a.cmd(1).mode_of_operation, cia402::kModeCst);

  // target past the soft upper limit (0.5): the CSP target stops at it
  a.h->stream(command(kModeP, {q0[0], 3.0}));
  double max_target = -1e9;
  for (int k = 0; k < 700; ++k) {
    a.h->cycle();
    ASSERT_EQ(a.cmd(1).mode_of_operation, cia402::kModeCsp);
    max_target = std::max(max_target, a.cmd(1).position);
  }
  EXPECT_LE(max_target, 0.5 + 1e-9);
  EXPECT_NEAR(a.cmd(1).position, 0.5, 1e-9);
  EXPECT_NEAR(a.q(1), 0.5, 0.02);
  // and it got there at no more than max_velocity (2 rad/s): >= 0.4 s for 0.8 rad

  // NaN gains / feedforward / clamp: treated as unset, nothing non-finite written
  a.h->stream(command(kModeT, q0, {kNaN, kNaN}, {kNaN, kNaN}, {kNaN, kNaN}, {kNaN, kNaN}));
  for (int k = 0; k < 50; ++k) {
    a.h->cycle();
    for (size_t i = 0; i < 2; ++i) {
      EXPECT_TRUE(std::isfinite(a.cmd(i).effort));
      EXPECT_TRUE(std::isfinite(a.cmd(i).position));
      EXPECT_TRUE(std::isfinite(a.cmd(i).torque_offset));
    }
  }
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(a.h->hw().slave(i).state(), State::kOperationEnabled);
  }
}

// --------------------------------------------------------------------------
// Fault -> reaction -> reset with drive feedback. j1 faults mid-move (torque
// off, brake holds); the controller drops the arm to the fallback and tracks
// j1's actual position; after reset_fault the move resumes from where j1 is,
// with no stale-target step and no following error.
TEST_F(ArmMock, DriveFaultReactionAndResetWithFeedback)
{
  Arm a = make_arm(true, 0, true, true);
  const std::vector<double> q0 = {0.2, -0.3};
  a.h->stream(command(kModeP, q0));
  a.h->cycle(300);
  const std::vector<double> goal = {0.7, -0.3};
  a.h->stream(command(kModeP, goal));
  a.h->cycle(60);   // j1 moving at 2 rad/s

  a.h->hw().slave(0).inject_fault(cia402::kErrOverCurrent);
  a.h->cycle();
  EXPECT_EQ(a.h->hw().slave(0).state(), State::kFault);
  EXPECT_EQ(a.tau(0), 0.0);
  a.h->cycle(5);
  // reaction: the arm drops to the fallback; j2 is held by gravity comp
  EXPECT_EQ(a.cmd(0).mode_of_operation, cia402::kModeCst);
  EXPECT_EQ(a.cmd(1).mode_of_operation, cia402::kModeCst);
  EXPECT_EQ(static_cast<uint16_t>(a.st(0).error_code), cia402::kErrOverCurrent);
  const double q_fault = a.q(0);
  a.h->cycle(100);
  EXPECT_NEAR(a.q(0), q_fault, 1e-9) << "brake";
  EXPECT_NEAR(a.cmd(0).position, a.st(0).position, 1e-9) << "target must track the stopped joint";
  EXPECT_NEAR(a.q(1), q0[1], 0.02);

  // reset
  a.h->hw().commands(0).reset_fault = 0.0;
  a.h->cycle();
  a.h->hw().commands(0).reset_fault = 1.0;
  int k = 0;
  for (; k < 30 && !all_enabled(*a.h); ++k) {a.h->cycle();}
  ASSERT_LT(k, 30);
  a.h->hw().commands(0).reset_fault = 0.0;

  // resumes Mode P from the actual position, not from where the target
  // would have got to
  a.h->cycle();
  EXPECT_EQ(a.cmd(0).mode_of_operation, cia402::kModeCsp);
  EXPECT_NEAR(a.cmd(0).position, a.st(0).position, 0.01);
  for (int j = 0; j < 600; ++j) {
    a.h->cycle();
    ASSERT_EQ(a.h->hw().slave(0).state(), State::kOperationEnabled);
    ASSERT_LT(std::fabs(a.h->hw().slave(0).last_rx().target_position * 2 * M_PI / 131072.0 -
      a.h->hw().slave(0).position()), 0.05) << "CSP error at cycle " << j;
  }
  EXPECT_FALSE(a.h->hw().slave(0).following_error());
  EXPECT_NEAR(a.q(0), goal[0], 0.01);
}

// The same fault without drive feedback (the current bench config: no
// 0x6041 / 0x6061 state interfaces): the controller never sees the fault,
// keeps advancing the CSP target, and re-enabling steps the drive to it.
TEST_F(ArmMock, DocumentsStaleTargetStepAfterResetWithoutFeedback)
{
  Arm a = make_arm(false, 0, true, true);
  a.h->stream(command(kModeP, {0.2, -0.3}));
  a.h->cycle(300);
  a.h->stream(command(kModeP, {0.7, -0.3}));
  a.h->cycle(60);
  a.h->hw().slave(0).inject_fault(cia402::kErrOverCurrent);
  a.h->cycle(150);
  EXPECT_EQ(a.cmd(0).mode_of_operation, cia402::kModeCsp) << "controller did not notice";
  EXPECT_GT(a.cmd(0).position - a.q(0), 0.25) << "target ran ahead of the braked joint";

  a.h->hw().commands(0).reset_fault = 0.0;
  a.h->cycle();
  a.h->hw().commands(0).reset_fault = 1.0;
  double max_tau = 0.0;
  for (int j = 0; j < 100; ++j) {
    a.h->cycle();
    max_tau = std::max(max_tau, std::fabs(a.tau(0)));
  }
  // re-enabled straight into a >0.25 rad CSP step: the 17.7 Nm 0x6072 cap,
  // then the drive's following-error fault
  EXPECT_GT(max_tau, 0.9 * 450 * 0.039375);
  EXPECT_TRUE(a.h->hw().slave(0).following_error());
}

// --------------------------------------------------------------------------
// Lost frames: a short burst is ridden through (the drive keeps its last
// targets); a long one trips the drive's process-data watchdog, the
// controller falls back, and a reset brings both drives back.
TEST_F(ArmMock, DroppedFramesShortRideThroughLongFaults)
{
  Arm a = make_arm(true, 0, true, true);
  const std::vector<double> q0 = {0.2, -0.3};
  a.h->stream(command(kModeP, q0));
  a.h->cycle(300);

  a.h->hw().drop_frames(20);
  a.h->cycle(20);
  a.h->cycle(10);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(a.h->hw().slave(i).state(), State::kOperationEnabled);
    EXPECT_NEAR(a.q(i), q0[i], 0.01);
    EXPECT_EQ(a.cmd(i).mode_of_operation, cia402::kModeCsp);
  }

  a.h->hw().drop_frames(150);
  a.h->cycle(150);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(a.h->hw().slave(i).state(), State::kFault);
    EXPECT_EQ(a.h->hw().slave(i).error_code(), cia402::kErrHeartbeat);
  }
  a.h->cycle(5);
  EXPECT_EQ(a.cmd(0).mode_of_operation, cia402::kModeCst) << "fallback once the fault is seen";

  for (size_t i = 0; i < 2; ++i) {a.h->hw().commands(i).reset_fault = 0.0;}
  a.h->cycle();
  for (size_t i = 0; i < 2; ++i) {a.h->hw().commands(i).reset_fault = 1.0;}
  int k = 0;
  for (; k < 30 && !all_enabled(*a.h); ++k) {a.h->cycle();}
  ASSERT_LT(k, 30);
  a.h->cycle(200);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(a.cmd(i).mode_of_operation, cia402::kModeCsp);
    EXPECT_NEAR(a.q(i), q0[i], 0.01);
  }
}

// --------------------------------------------------------------------------
// SafeCommandController (effort mode, CST) against the same drives: magnitude
// clamp, NaN rejection, soft-limit taper and outward block, and the timeout.
TEST_F(ArmMock, SafeCommandControllerRejectionsReachTheDrive)
{
  JointSpec j1;
  j1.name = "j1";
  j1.initial_position = 0.0;
  JointSpec j2;
  j2.name = "j2";
  j2.initial_position = 0.95;
  ControllerHarness h({j1, j2}, {{"mode_of_operation", "10"}});
  for (int k = 0; k < 50 && !all_enabled(h); ++k) {h.run_bus(1);}
  ASSERT_TRUE(all_enabled(h));
  auto c = std::make_shared<SafeCommandController>();
  ASSERT_TRUE(h.load(c, "effort_controller", {
      rclcpp::Parameter("joints", std::vector<std::string>{"j1", "j2"}),
      rclcpp::Parameter("interface_name", std::string("effort")),
      rclcpp::Parameter("command_timeout", 0.1),
      rclcpp::Parameter("soft_margin", 0.1),
      rclcpp::Parameter("max_command", std::vector<double>{2.0, 1.0}),
      rclcpp::Parameter("position_min", std::vector<double>{-1.0, -1.0}),
      rclcpp::Parameter("position_max", std::vector<double>{1.0, 1.0}),
      rclcpp::Parameter("damping", std::vector<double>{0.5, 0.2}),
    }, "/commands"));
  const std::vector<safety::JointLimits> lim = {{2.0, -1.0, 1.0, 0.5}, {1.0, -1.0, 1.0, 0.2}};
  auto expected = [&](size_t i, double cmd, bool fresh) {
      return safety::safe_command(
        safety::Mode::kEffort, cmd, fresh, h.hw().states(i).position, h.hw().states(i).velocity,
        lim[i], 0.1);
    };

  // over the limit: clamped; j2 near its upper limit: tapered
  h.send({10.0, 1.0});
  h.cycle();
  EXPECT_DOUBLE_EQ(h.hw().commands(0).effort, 2.0);
  EXPECT_DOUBLE_EQ(h.hw().commands(1).effort, expected(1, 1.0, true));
  EXPECT_LT(h.hw().commands(1).effort, 0.6);
  h.cycle(3);
  EXPECT_NEAR(h.hw().joint_torque(0), 2.0, 0.015) << "the drive applies the clamped torque";

  // NaN: hold (damping), never NaN on the bus
  h.send({kNaN, 0.0});
  for (int k = 0; k < 5; ++k) {
    h.cycle();
    EXPECT_DOUBLE_EQ(h.hw().commands(0).effort, expected(0, kNaN, true));
    EXPECT_TRUE(std::isfinite(h.hw().commands(0).effort));
  }

  // wrong length: ignored
  const double before = h.hw().commands(1).effort;
  h.send({1.0, 1.0, 1.0});
  h.cycle();
  EXPECT_DOUBLE_EQ(h.hw().commands(1).effort, expected(1, 0.0, true));
  (void)before;

  // something pushes j2 out past its limit while the command still asks for
  // more: past the limit, no outward torque, only damping against the motion
  h.stream({0.0, 1.0}, 5);
  h.hw().slave(1).set_load_torque(-0.3);   // 0.3 Nm outward
  int past = 0;
  for (int k = 0; k < 400; ++k) {
    h.cycle();
    if (h.hw().states(1).position > 1.0 && h.hw().states(1).velocity > 0.0) {
      ++past;
      EXPECT_LE(h.hw().commands(1).effort, 0.0);
      EXPECT_DOUBLE_EQ(h.hw().commands(1).effort, expected(1, 1.0, true));
    }
  }
  EXPECT_GT(past, 50);
  h.hw().slave(1).set_load_torque(0.0);

  // timeout: stops streaming -> damping only after command_timeout (100 ms)
  h.stream({1.0, 0.0}, 5);
  h.cycle(50);
  h.stop_stream();
  h.cycle(60);
  EXPECT_DOUBLE_EQ(h.hw().commands(0).effort, expected(0, 1.0, true)) << "stale before 100 ms";
  h.cycle(80);
  EXPECT_DOUBLE_EQ(h.hw().commands(0).effort, expected(0, 1.0, false));
  EXPECT_LE(std::fabs(h.hw().commands(0).effort), 0.5 * std::fabs(h.hw().states(0).velocity) + 1e-12);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
