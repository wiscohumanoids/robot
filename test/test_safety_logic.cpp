#include <gtest/gtest.h>

#include <limits>

#include "wisco_ethercat_ros2_hw/safety_logic.hpp"

using wisco_ethercat_ros2_hw::safety::JointLimits;
using wisco_ethercat_ros2_hw::safety::Mode;
using wisco_ethercat_ros2_hw::safety::safe_command;

namespace
{
const JointLimits kLimits{2.0, -1.0, 1.0, 0.5};  // max 2, range [-1, 1], damping 0.5
const double kMargin = 0.1;
const double kNaN = std::numeric_limits<double>::quiet_NaN();
}  // namespace

TEST(SafetyLogic, PassesCommandInsideLimits)
{
  EXPECT_DOUBLE_EQ(safe_command(Mode::kVelocity, 1.5, true, 0.0, 0.0, kLimits, kMargin), 1.5);
  EXPECT_DOUBLE_EQ(safe_command(Mode::kEffort, -1.5, true, 0.0, 0.0, kLimits, kMargin), -1.5);
}

TEST(SafetyLogic, ClampsMagnitude)
{
  EXPECT_DOUBLE_EQ(safe_command(Mode::kVelocity, 10.0, true, 0.0, 0.0, kLimits, kMargin), 2.0);
  EXPECT_DOUBLE_EQ(safe_command(Mode::kEffort, -10.0, true, 0.0, 0.0, kLimits, kMargin), -2.0);
}

TEST(SafetyLogic, StaleCommandStopsVelocityAndDampsEffort)
{
  EXPECT_DOUBLE_EQ(safe_command(Mode::kVelocity, 1.0, false, 0.0, 3.0, kLimits, kMargin), 0.0);
  EXPECT_DOUBLE_EQ(safe_command(Mode::kEffort, 1.0, false, 0.0, 3.0, kLimits, kMargin), -1.5);
  // Damping torque is clamped too.
  EXPECT_DOUBLE_EQ(safe_command(Mode::kEffort, 1.0, false, 0.0, 100.0, kLimits, kMargin), -2.0);
}

TEST(SafetyLogic, RejectsNaN)
{
  EXPECT_DOUBLE_EQ(safe_command(Mode::kVelocity, kNaN, true, 0.0, 0.0, kLimits, kMargin), 0.0);
  EXPECT_DOUBLE_EQ(safe_command(Mode::kVelocity, 1.0, true, kNaN, 0.0, kLimits, kMargin), 0.0);
  EXPECT_DOUBLE_EQ(safe_command(Mode::kEffort, kNaN, true, 0.0, kNaN, kLimits, kMargin), 0.0);
}

TEST(SafetyLogic, TapersTowardLimit)
{
  // Halfway into the margin below the upper limit: half the command.
  EXPECT_NEAR(safe_command(Mode::kVelocity, 1.0, true, 0.95, 0.0, kLimits, kMargin), 0.5, 1e-9);
  // Motion away from the limit is not reduced.
  EXPECT_DOUBLE_EQ(safe_command(Mode::kVelocity, -1.0, true, 0.95, 0.0, kLimits, kMargin), -1.0);
  // Same at the lower limit.
  EXPECT_NEAR(safe_command(Mode::kVelocity, -1.0, true, -0.95, 0.0, kLimits, kMargin), -0.5, 1e-9);
}

TEST(SafetyLogic, BlocksOutwardMotionPastLimit)
{
  EXPECT_DOUBLE_EQ(safe_command(Mode::kVelocity, 1.0, true, 1.2, 0.0, kLimits, kMargin), 0.0);
  EXPECT_DOUBLE_EQ(safe_command(Mode::kVelocity, -1.0, true, 1.2, 0.0, kLimits, kMargin), -1.0);
  EXPECT_DOUBLE_EQ(safe_command(Mode::kVelocity, -1.0, true, -1.2, 0.0, kLimits, kMargin), 0.0);
}

TEST(SafetyLogic, EffortPastLimitAddsDampingOnlyOutward)
{
  // Past the upper limit, still moving out at 2 rad/s: no outward torque, damping -1.
  EXPECT_DOUBLE_EQ(safe_command(Mode::kEffort, 1.0, true, 1.2, 2.0, kLimits, kMargin), -1.0);
  // Moving back in: the inward command passes, no damping.
  EXPECT_DOUBLE_EQ(safe_command(Mode::kEffort, -1.0, true, 1.2, -2.0, kLimits, kMargin), -1.0);
  // Past the lower limit, moving further out.
  EXPECT_DOUBLE_EQ(safe_command(Mode::kEffort, -1.0, true, -1.2, -2.0, kLimits, kMargin), 1.0);
}

TEST(SafetyLogic, ZeroMarginIsHardLimit)
{
  EXPECT_DOUBLE_EQ(safe_command(Mode::kVelocity, 1.0, true, 0.999, 0.0, kLimits, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(safe_command(Mode::kVelocity, 1.0, true, 1.0, 0.0, kLimits, 0.0), 0.0);
}
