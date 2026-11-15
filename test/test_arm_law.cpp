// Unit tests for the arm's RT law (arm_law.hpp). The same cases are checked
// against the Python mirror in test/arm_stack/test_law.py.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "wisco_ethercat_ros2_hw/arm_law.hpp"

using wisco_ethercat_ros2_hw::arm::blend;
using wisco_ethercat_ros2_hw::arm::compute;
using wisco_ethercat_ros2_hw::arm::kModeP;
using wisco_ethercat_ros2_hw::arm::kModeT;
using wisco_ethercat_ros2_hw::arm::LawCommand;
using wisco_ethercat_ros2_hw::arm::LawConfig;
using wisco_ethercat_ros2_hw::arm::LawOutput;
using wisco_ethercat_ros2_hw::arm::rate_limit;

namespace
{
const double kNaN = std::numeric_limits<double>::quiet_NaN();

LawConfig config()
{
  LawConfig c;
  c.max_torque = {10.0, 2.0};
  c.max_velocity = {2.0, 4.0};
  c.lower = {-1.0, -0.5};
  c.upper = {1.0, 0.5};
  c.kd_fallback = {0.5, 0.1};
  c.coulomb = {0.0, 0.0};
  c.viscous = {0.0, 0.0};
  return c;
}

LawCommand command(int mode)
{
  LawCommand cmd;
  cmd.present = true;
  cmd.age_s = 0.0;
  cmd.mode = mode;
  cmd.q = {0.2, 0.1};
  cmd.qd = {0.0, 0.0};
  cmd.kp = {50.0, 10.0};
  cmd.kd = {1.0, 0.2};
  cmd.tau_ff = {0.0, 0.0};
  cmd.torque_clamp = {kNaN, kNaN};
  return cmd;
}

const std::vector<double> kQ = {0.0, 0.0};
const std::vector<double> kQd = {0.0, 0.0};
const std::vector<double> kG = {3.0, 0.5};
}  // namespace

TEST(ArmLaw, NoCommandIsGravityCompAndDampingInCst)
{
  LawOutput out;
  const std::vector<double> qd = {1.0, -2.0};
  compute(config(), LawCommand{}, kQ, qd, kG, out);
  EXPECT_TRUE(out.fallback);
  EXPECT_EQ(out.mode, kModeT);
  EXPECT_NEAR(out.tau[0], 3.0 - 0.5, 1e-12);
  EXPECT_NEAR(out.tau[1], 0.5 + 0.2, 1e-12);
}

TEST(ArmLaw, StaleCommandTripsWatchdog)
{
  auto cmd = command(kModeP);
  cmd.age_s = 0.051;
  LawOutput out;
  compute(config(), cmd, kQ, kQd, kG, out);
  EXPECT_TRUE(out.fallback);
  cmd.age_s = 0.049;
  compute(config(), cmd, kQ, kQd, kG, out);
  EXPECT_FALSE(out.fallback);
}

TEST(ArmLaw, ModePMissingTargetFallsBack)
{
  auto cmd = command(kModeP);
  cmd.q[1] = kNaN;
  LawOutput out;
  compute(config(), cmd, kQ, kQd, kG, out);
  EXPECT_TRUE(out.fallback);
}

TEST(ArmLaw, ModePSendsTargetAndGravityOffset)
{
  auto cmd = command(kModeP);
  cmd.tau_ff = {0.5, kNaN};
  LawOutput out;
  compute(config(), cmd, kQ, kQd, kG, out);
  EXPECT_EQ(out.mode, kModeP);
  EXPECT_DOUBLE_EQ(out.q_target[0], 0.2);
  EXPECT_DOUBLE_EQ(out.tau[0], 3.5);
  EXPECT_DOUBLE_EQ(out.tau[1], 0.5);
}

TEST(ArmLaw, ModePClampsTargetToSoftLimits)
{
  auto cmd = command(kModeP);
  cmd.q = {5.0, -5.0};
  LawOutput out;
  compute(config(), cmd, kQ, kQd, kG, out);
  EXPECT_TRUE(out.limit_hit);
  EXPECT_DOUBLE_EQ(out.q_target[0], 1.0);
  EXPECT_DOUBLE_EQ(out.q_target[1], -0.5);
  cmd.ignore_soft_limits = true;
  compute(config(), cmd, kQ, kQd, kG, out);
  EXPECT_DOUBLE_EQ(out.q_target[0], 5.0);
}

TEST(ArmLaw, ModeTPdPlusFeedforward)
{
  auto cmd = command(kModeT);
  LawOutput out;
  const std::vector<double> qd = {0.1, 0.0};
  compute(config(), cmd, kQ, qd, kG, out);
  EXPECT_EQ(out.mode, kModeT);
  // joint 1: 50*0.2 - 1*0.1 + 3 = 12.9, clamped to 10 below
  // joint 2: 10*0.1 + 0.5 = 1.5, under the 2 Nm limit
  EXPECT_NEAR(out.tau[1], 1.5, 1e-12);
  EXPECT_TRUE(out.clamped);  // joint 1 hit 10 Nm
  EXPECT_DOUBLE_EQ(out.tau[0], 10.0);
}

TEST(ArmLaw, CommandTorqueClampOnlyTightens)
{
  auto cmd = command(kModeT);
  cmd.torque_clamp = {1.0, 50.0};
  cmd.kp = {0.0, 0.0};
  cmd.kd = {0.0, 0.0};
  cmd.tau_ff = {0.0, 5.0};
  LawOutput out;
  compute(config(), cmd, kQ, kQd, kG, out);
  EXPECT_DOUBLE_EQ(out.tau[0], 1.0);
  EXPECT_DOUBLE_EQ(out.tau[1], 2.0);
}

TEST(ArmLaw, ModeTOutsideSoftLimitNeverPushesFurtherOut)
{
  auto cmd = command(kModeT);
  cmd.q = {2.0, 0.0};          // asks to go further past upper
  cmd.ignore_soft_limits = true;
  const std::vector<double> q = {1.02, 0.0};
  LawOutput out;
  compute(config(), cmd, q, kQd, kG, out);
  EXPECT_TRUE(out.limit_hit);
  EXPECT_LT(out.tau[0], 0.0);
}

TEST(ArmLaw, OverspeedIsDamped)
{
  auto cmd = command(kModeT);
  cmd.kp = {0.0, 0.0};
  cmd.kd = {0.0, 0.0};
  cmd.gravity_ff = false;
  const std::vector<double> qd = {3.0, 0.0};
  LawOutput out;
  compute(config(), cmd, kQ, qd, kG, out);
  EXPECT_NEAR(out.tau[0], -5.0 * 0.5 * 1.0, 1e-12);
}

TEST(ArmLaw, BumplessHelpers)
{
  EXPECT_DOUBLE_EQ(rate_limit(0.0, 1.0, 0.002), 0.002);
  EXPECT_DOUBLE_EQ(rate_limit(0.0, -1.0, 0.002), -0.002);
  EXPECT_DOUBLE_EQ(rate_limit(0.5, 0.501, 0.002), 0.501);
  EXPECT_DOUBLE_EQ(blend(2.0, 4.0, 0.0, 0.02), 2.0);
  EXPECT_DOUBLE_EQ(blend(2.0, 4.0, 0.01, 0.02), 3.0);
  EXPECT_DOUBLE_EQ(blend(2.0, 4.0, 0.05, 0.02), 4.0);
}

// C++ only (no Python mirror yet): CSP re-entry ease, zero slope at both ends.
TEST(ArmLaw, EaseInStartsAndEndsAtRest)
{
  using wisco_ethercat_ros2_hw::arm::ease_in;
  EXPECT_DOUBLE_EQ(ease_in(1.0, 2.0, 0.0, 0.02), 1.0);
  EXPECT_DOUBLE_EQ(ease_in(1.0, 2.0, 0.01, 0.02), 1.5);
  EXPECT_DOUBLE_EQ(ease_in(1.0, 2.0, 0.03, 0.02), 2.0);
  EXPECT_DOUBLE_EQ(ease_in(1.0, 2.0, 0.0, 0.0), 2.0);
  const double h = 1e-7;
  EXPECT_NEAR((ease_in(0.0, 1.0, h, 0.02) - ease_in(0.0, 1.0, 0.0, 0.02)) / h, 0.0, 1e-2);
  EXPECT_NEAR((ease_in(0.0, 1.0, 0.02, 0.02) - ease_in(0.0, 1.0, 0.02 - h, 0.02)) / h, 0.0, 1e-2);
}
