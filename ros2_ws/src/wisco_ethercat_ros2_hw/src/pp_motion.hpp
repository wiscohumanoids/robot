// One absolute PP setpoint relative to the live output encoder (131072 counts/turn).
#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pp_test
{
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
constexpr int step_degrees = 5;
constexpr int step_counts = (131072 * step_degrees + 180) / 360;
constexpr int tolerance = 55;     // 0.15 degrees
constexpr int speed = 728;        // 2 degrees/s
constexpr int acceleration = 2913;  // 8 degrees/s^2

struct Feedback
{
  uint16_t status;
  int32_t actual;
  int mode;
  uint16_t error;
  bool enabled() const { return (status & 0x6f) == 0x27; }
  bool disabled() const { return (status & 0x4f) == 0x40; }
};

struct Motion
{
  enum Phase { ready, switched_on, enabled, stage_target, travel, done };
  Phase phase = ready;
  uint16_t control = 6;
  int32_t target;
  int32_t start;
  Clock::time_point deadline;
  Clock::time_point staged_at{};
  Clock::time_point settled_at{};
  bool saw_ack_low = false;
  bool ack_edge = false;

  Motion(int32_t position, Clock::time_point now)
  : target(position), start(position), deadline(now + 3s)
  {
    if (position > std::numeric_limits<int32_t>::max() - step_counts) {
      throw std::runtime_error("PP target would overflow the signed encoder range");
    }
  }

  void update(const Feedback & f, Clock::time_point now)
  {
    if (f.error || (f.status & 0x2008) || f.mode != 1) {
      throw std::runtime_error("Drive fault, following error, or loss of PP mode");
    }
    if (now >= deadline) {
      throw std::runtime_error(phase == travel ? "PP target timed out" : "PP enable/staging timed out");
    }
    if (phase >= stage_target && !f.enabled()) {
      throw std::runtime_error("Drive lost operation enabled during PP test");
    }

    switch (phase) {
      case ready:
        if ((f.status & 0x6f) == 0x21) {
          phase = switched_on;
          control = 7;
        }
        break;
      case switched_on:
        if ((f.status & 0x6f) == 0x23) {
          phase = enabled;
          control = 0xf;
        }
        break;
      case enabled:
        if (f.enabled()) {
          target = start + step_counts;
          staged_at = now;
          phase = stage_target;
        }
        break;
      case stage_target:
        saw_ack_low = !(f.status & 0x1000);
        if (now - staged_at >= 20ms) {
          control = 0x1f;
          phase = travel;
          deadline = now + 5s;
        }
        break;
      case travel:
        if (!(f.status & 0x1000)) { saw_ack_low = true; }
        if (saw_ack_low && (f.status & 0x1000)) {
          ack_edge = true;
          control = 0xf;  // release new-setpoint after the drive acknowledges it
        }
        if ((f.status & 0x0400) && std::abs(static_cast<int64_t>(f.actual) - target) <= tolerance) {
          if (settled_at == Clock::time_point{}) { settled_at = now; }
          if (now - settled_at >= 100ms) {
            phase = done;
            control = 0;
          }
        } else {
          settled_at = {};
        }
        break;
      case done: break;
    }
  }
};
}  // namespace pp_test
