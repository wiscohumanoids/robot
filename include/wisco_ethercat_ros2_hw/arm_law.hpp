// The arm's RT control law (spec: Hardware assumptions, Software architecture).
//
// Header-only and free of ROS so it can be unit tested. It mirrors
// arm_stack/control/law.py, which the sim backend runs, so procedures see the
// same behaviour with and without --sim.
//
//   Mode P (CSP): q_target -> 0x607A, tau_offset = g(q) + f(qd_d) + tau_ff -> 0x60B2
//   Mode T (CST): tau = Kp(q_d - q) + Kd(qd_d - qd) + g(q) + f(qd) + tau_ff -> 0x6071
//
// then torque and velocity clamps, soft limits, and a fallback to gravity comp
// plus joint damping (in CST) when no command arrived within the watchdog.
#ifndef WISCO_ETHERCAT_ROS2_HW__ARM_LAW_HPP_
#define WISCO_ETHERCAT_ROS2_HW__ARM_LAW_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace wisco_ethercat_ros2_hw
{
namespace arm
{

enum Mode : int { kModeP = 0, kModeT = 1 };

// CiA402 modes of operation (0x6060)
constexpr int kCsp = 8;
constexpr int kCst = 10;

struct LawConfig
{
  std::vector<double> max_torque;    // Nm, per joint
  std::vector<double> max_velocity;  // rad/s
  std::vector<double> lower;         // soft limits, rad
  std::vector<double> upper;
  std::vector<double> kd_fallback;   // Nm s/rad, watchdog damping
  std::vector<double> coulomb;       // friction model, Nm
  std::vector<double> viscous;       // Nm s/rad
  double soft_margin = 0.05;         // rad, sets the soft-limit spring
  double watchdog_s = 0.05;
};

struct LawCommand
{
  bool present = false;      // a command has been received since activation
  double age_s = 1e9;
  int mode = kModeP;
  bool gravity_ff = true;
  bool ignore_soft_limits = false;
  double friction_ff_gain = 1.0;
  // NaN entries mean "not set": q -> no target (fallback in Mode P),
  // qd/kp/kd/tau_ff -> 0, torque_clamp -> max_torque.
  std::vector<double> q, qd, kp, kd, tau_ff, torque_clamp;
};

struct LawOutput
{
  int mode = kModeT;
  std::vector<double> q_target;
  std::vector<double> tau;   // Mode P: torque offset; Mode T: target torque
  bool clamped = false;
  bool limit_hit = false;
  bool fallback = true;
};

inline double friction(const LawConfig & c, size_t i, double v)
{
  return c.coulomb[i] * std::tanh(v / 0.02) + c.viscous[i] * v;
}

inline double or_zero(const std::vector<double> & v, size_t i)
{
  return (i < v.size() && std::isfinite(v[i])) ? v[i] : 0.0;
}

// q, qd: measured. g: gravity torque at q. out is resized once and reused.
inline void compute(
  const LawConfig & c, const LawCommand & cmd, const std::vector<double> & q,
  const std::vector<double> & qd, const std::vector<double> & g, LawOutput & out)
{
  const size_t n = q.size();
  out.q_target.resize(n);
  out.tau.resize(n);
  out.clamped = false;
  out.limit_hit = false;

  bool q_missing = false;
  for (size_t i = 0; i < n; ++i) {
    if (i >= cmd.q.size() || !std::isfinite(cmd.q[i])) {q_missing = true;}
  }
  const bool fallback = !cmd.present || cmd.age_s > c.watchdog_s ||
    (cmd.mode == kModeP && q_missing);
  out.fallback = fallback;

  auto clamp_tau = [&](size_t i, double tau, bool use_cmd_clamp) {
      double lim = c.max_torque[i];
      if (use_cmd_clamp && i < cmd.torque_clamp.size() && std::isfinite(cmd.torque_clamp[i])) {
        lim = std::min(lim, std::fabs(cmd.torque_clamp[i]));
      }
      const double o = std::clamp(tau, -lim, lim);
      if (o != tau) {out.clamped = true;}
      return o;
    };

  if (fallback) {
    out.mode = kModeT;
    for (size_t i = 0; i < n; ++i) {
      out.q_target[i] = q[i];
      out.tau[i] = clamp_tau(i, g[i] - c.kd_fallback[i] * qd[i], false);
    }
    return;
  }

  const double limit_k_scale = 4.0 / std::max(c.soft_margin, 1e-3);
  for (size_t i = 0; i < n; ++i) {
    double q_d = (i < cmd.q.size() && std::isfinite(cmd.q[i])) ? cmd.q[i] : q[i];
    double qd_d = or_zero(cmd.qd, i);
    const double ff_v = (cmd.mode == kModeP) ? qd_d : qd[i];
    const double ff = cmd.gravity_ff ? g[i] + cmd.friction_ff_gain * friction(c, i, ff_v) : 0.0;
    const double tau_ff = or_zero(cmd.tau_ff, i);

    if (q_d < c.lower[i] || q_d > c.upper[i]) {out.limit_hit = true;}
    if (!cmd.ignore_soft_limits) {q_d = std::clamp(q_d, c.lower[i], c.upper[i]);}
    qd_d = std::clamp(qd_d, -c.max_velocity[i], c.max_velocity[i]);

    if (cmd.mode == kModeP) {
      out.mode = kModeP;
      out.q_target[i] = q_d;
      out.tau[i] = clamp_tau(i, ff + tau_ff, true);
      continue;
    }

    out.mode = kModeT;
    out.q_target[i] = q[i];
    double tau = or_zero(cmd.kp, i) * (q_d - q[i]) + or_zero(cmd.kd, i) * (qd_d - qd[i]) + ff + tau_ff;
    // outside a soft limit: spring back in, never push further out
    const double over = q[i] - c.upper[i];
    const double under = c.lower[i] - q[i];
    const double k_lim = limit_k_scale * c.max_torque[i];
    if (over > 0) {
      out.limit_hit = true;
      tau = std::min(tau, -k_lim * over - c.kd_fallback[i] * qd[i]);
    } else if (under > 0) {
      out.limit_hit = true;
      tau = std::max(tau, k_lim * under - c.kd_fallback[i] * qd[i]);
    }
    // velocity clamp: damp hard above the limit
    if (std::fabs(qd[i]) > c.max_velocity[i]) {
      const double excess = qd[i] - std::copysign(c.max_velocity[i], qd[i]);
      tau -= 5.0 * c.kd_fallback[i] * excess;
    }
    out.tau[i] = clamp_tau(i, tau, true);
  }
}

// Bumpless transfer helpers, applied by the controller on top of compute().
//
// CSP target: never jumps; moves toward the law's target at max_velocity,
// starting from the actual position when CSP is (re)entered.
inline double rate_limit(double previous, double target, double max_step)
{
  return previous + std::clamp(target - previous, -max_step, max_step);
}

// CSP re-entry: eases the target from the actual position at the switch to
// the law's target over blend_s (smoothstep: zero velocity at both ends), so
// a drive that feeds forward the target velocity (d 0x607A / dt) sees no step.
// Applied before rate_limit(). C++ only so far; arm_stack/sim/arm_sim.py
// rate-limits without it (its drive model has no velocity feedforward).
inline double ease_in(double at_switch, double target, double since_switch_s, double blend_s)
{
  if (since_switch_s >= blend_s || blend_s <= 0.0) {return target;}
  const double a = std::max(since_switch_s, 0.0) / blend_s;
  return at_switch + (target - at_switch) * a * a * (3.0 - 2.0 * a);
}

// CST torque: blends from the measured torque at the switch over blend_s.
inline double blend(double measured_at_switch, double target, double since_switch_s, double blend_s)
{
  if (since_switch_s >= blend_s || blend_s <= 0.0) {return target;}
  const double a = since_switch_s / blend_s;
  return (1.0 - a) * measured_at_switch + a * target;
}

}  // namespace arm
}  // namespace wisco_ethercat_ros2_hw

#endif  // WISCO_ETHERCAT_ROS2_HW__ARM_LAW_HPP_
