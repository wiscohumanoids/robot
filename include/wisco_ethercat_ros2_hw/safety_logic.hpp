// Pure functions behind SafeCommandController, kept free of ROS so they can
// be unit tested on their own.
#ifndef WISCO_ETHERCAT_ROS2_HW__SAFETY_LOGIC_HPP_
#define WISCO_ETHERCAT_ROS2_HW__SAFETY_LOGIC_HPP_

#include <algorithm>
#include <cmath>

namespace wisco_ethercat_ros2_hw
{
namespace safety
{

enum class Mode { kVelocity, kEffort };

struct JointLimits
{
  double max_command;   // rad/s in velocity mode, Nm in effort mode
  double position_min;  // rad
  double position_max;  // rad
  double damping;       // Nm s/rad, effort mode only
};

inline double clamp_abs(double value, double limit)
{
  return std::clamp(value, -limit, limit);
}

// Fraction of a command toward a position limit that is let through:
// 1 farther than `margin` from the limit, falling linearly to 0 at the limit,
// 0 at or past it. `toward_max` picks which limit the command points at.
inline double toward_limit_scale(
  double position, bool toward_max, const JointLimits & limits, double margin)
{
  const double distance =
    toward_max ? limits.position_max - position : position - limits.position_min;
  if (distance <= 0.0) {
    return 0.0;
  }
  if (margin <= 0.0 || distance >= margin) {
    return 1.0;
  }
  return distance / margin;
}

// What to send when there is no fresh command: stop in velocity mode, damp
// the joint (never push it) in effort mode.
inline double hold_command(Mode mode, double velocity, const JointLimits & limits)
{
  if (mode == Mode::kVelocity || !std::isfinite(velocity)) {
    return 0.0;
  }
  return clamp_abs(-limits.damping * velocity, limits.max_command);
}

// The command actually written to the hardware for one joint.
inline double safe_command(
  Mode mode, double command, bool fresh, double position, double velocity,
  const JointLimits & limits, double margin)
{
  if (!fresh || !std::isfinite(command) || !std::isfinite(position)) {
    return hold_command(mode, velocity, limits);
  }
  double out = clamp_abs(command, limits.max_command);
  if (out != 0.0) {
    out *= toward_limit_scale(position, out > 0.0, limits, margin);
  }

  // Past a limit in effort mode, also damp motion that is still heading out.
  if (mode == Mode::kEffort && std::isfinite(velocity)) {
    const bool outward = (position > limits.position_max && velocity > 0.0) ||
      (position < limits.position_min && velocity < 0.0);
    if (outward) {
      out -= limits.damping * velocity;
    }
    out = clamp_abs(out, limits.max_command);
  }
  return out;
}

}  // namespace safety
}  // namespace wisco_ethercat_ros2_hw

#endif  // WISCO_ETHERCAT_ROS2_HW__SAFETY_LOGIC_HPP_
