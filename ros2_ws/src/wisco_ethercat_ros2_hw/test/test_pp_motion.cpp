// Tests of the profile-position step sequencing used by pp_test (no hardware).

#include <gtest/gtest.h>
#include "pp_motion.hpp"

namespace
{
using namespace pp_test;
const auto begin = Clock::time_point{} + 1s;
constexpr int32_t start = -32000;

// A motion staged from the start position.
Motion stage()
{
  Motion m(start, begin);
  EXPECT_EQ(m.control, 6);
  EXPECT_EQ(m.target, start);
  m.update({0x231, start, 1, 0}, begin + 1ms);
  EXPECT_EQ(m.control, 7);
  m.update({0x233, start, 1, 0}, begin + 2ms);
  EXPECT_EQ(m.control, 15);
  m.update({0x237, start, 1, 0}, begin + 3ms);
  EXPECT_EQ(m.target, start + step_counts);
  EXPECT_EQ(m.control, 15);
  m.update({0x237, start, 1, 0}, begin + 10ms);
  EXPECT_EQ(m.control, 15);
  return m;
}

TEST(PPMotion, MeasuresTravelAfterRisingEdgeAndRequiresSettledFeedback)
{
  auto m = stage();
  m.update({0x237, start, 1, 0}, begin + 30ms);
  EXPECT_EQ(m.control, 0x1f);
  m.update({0x1237, start - 57, 1, 0}, begin + 40ms);
  EXPECT_TRUE(m.ack_edge);
  EXPECT_EQ(m.control, 0xf);
  m.update({0x1637, m.target + tolerance + 1, 1, 0}, begin + 900ms);
  m.update({0x1637, m.target, 1, 0}, begin + 1s);
  EXPECT_EQ(m.phase, Motion::travel);
  m.update({0x1637, m.target, 1, 0}, begin + 1110ms);
  EXPECT_EQ(m.phase, Motion::done);
  EXPECT_EQ(m.control, 0);
}

TEST(PPMotion, StaleReachedAndStaleAckWithFrozenEncoderCannotPass)
{
  auto m = stage();
  m.update({0x1637, start, 1, 0}, begin + 30ms);
  m.update({0x1637, start, 1, 0}, begin + 1s);
  EXPECT_FALSE(m.ack_edge);
  EXPECT_THROW(m.update({0x1637, start, 1, 0}, begin + 6s), std::runtime_error);
}

TEST(PPMotion, RejectsFaultWrongModeAndLostEnable)
{
  const Feedback bad[] = {
    {0x237, start, 8, 0}, {0x237, start, 1, 7}, {0x2237, start, 1, 0},
    {0x250, start, 1, 0}};
  for (const auto & sample : bad) {
    auto m = stage();
    m.update({0x237, start, 1, 0}, begin + 30ms);
    EXPECT_THROW(m.update(sample, begin + 40ms), std::runtime_error);
  }
}
TEST(PPMotion, RejectsOverflowBeforeEnable)
{
  EXPECT_THROW(Motion(std::numeric_limits<int32_t>::max(), begin), std::runtime_error);
}
}
