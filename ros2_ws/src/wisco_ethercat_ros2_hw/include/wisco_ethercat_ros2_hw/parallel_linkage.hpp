// Parallel two-motor joints (Wisco V1 ankles and waist): conversion between the two JOINT angles the walking policy
// commands (pitch / roll, a serial two-axis joint as in the URDF and the simulator) and the two MOTOR (crank) angles
// that drive it through push rods. This is what Unitree's "PR mode" does inside its robots ("Series Control for
// Pitch/Roll Joints", versus "AB mode", "Parallel Control for A/B Joints", in unitree_sdk2's G1 ankle example): the
// policy only ever sees pitch and roll.
//
// Geometry comes from the CAD (robots/wisco_v1/linkage.json, read by parallel_linkage_yaml.hpp): the joint's two axes,
// and for each motor its crank axis, the crank ball, the ball at the far end of the rod and the rod length, all in the
// base frame at the zero pose. A motor angle is the crank's rotation from its CAD position, about its crank axis.
//
//   motor_angles(q)   joints -> motors, closed form: each rod's length is fixed, so the crank ball lies on a circle
//                     and one rod length from the far ball: P cos(th) + Q sin(th) = K. Of the two roots (the crank on
//                     either side: two ways to assemble the rod) the one the CAD shows is kept, always; it cannot
//                     switch without passing through a locked (singular) pose. Angles are taken within half a turn of
//                     the reference.
//   jacobian(q)       J = d(motor angles)/d(joint angles), 2x2 (rows motors, columns joints)
//   joint_angles(th)  motors -> joints, Newton's method on motor_angles (warm-started from the previous joint angles)
//   velocities        motor = J * joint;  torques: joint = J^T * motor  (virtual work)
//   stiffness/damping a joint-space PD (K_q) seen at the motors is K_th = J^-T K_q J^-1 (2x2); per-motor PD (MyActuator
//                     mode 5) can only apply its diagonal, so the coupling term is reported (motor_gains)
//
// Real-time safe: no allocation, no exceptions, bounded iterations. The Python reference is robots/wisco_v1/linkage.py;
// test/test_parallel_linkage.cpp checks this file against it (test/data/linkage_parity.json).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace wisco_ethercat_ros2_hw::linkage
{

using Vec3 = std::array<double, 3>;
using Vec2 = std::array<double, 2>;
using Mat2 = std::array<std::array<double, 2>, 2>;   // m[row][col]

// ---- small vector helpers ----------------------------------------------------------------------------------------
inline Vec3 add(const Vec3 & a, const Vec3 & b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline Vec3 sub(const Vec3 & a, const Vec3 & b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline Vec3 scale(const Vec3 & a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline double dot(const Vec3 & a, const Vec3 & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec3 cross(const Vec3 & a, const Vec3 & b)
{
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline Vec3 unit(const Vec3 & a)
{
  const double n = std::sqrt(dot(a, a));
  return n > 0.0 ? scale(a, 1.0 / n) : a;
}

// Rotate point p by angle ang about the axis through pt with direction axis (unit), Rodrigues' formula.
inline Vec3 rotate_about(const Vec3 & p, const Vec3 & pt, const Vec3 & axis, double ang)
{
  const Vec3 v = sub(p, pt);
  const double c = std::cos(ang), s = std::sin(ang);
  const Vec3 r = add(add(scale(v, c), scale(cross(axis, v), s)), scale(axis, dot(axis, v) * (1.0 - c)));
  return add(r, pt);
}

inline double det(const Mat2 & m) { return m[0][0] * m[1][1] - m[0][1] * m[1][0]; }
inline bool inverse(const Mat2 & m, Mat2 & out)
{
  const double d = det(m);
  if (!(std::fabs(d) > 1e-12)) {return false;}
  out = {{{m[1][1] / d, -m[0][1] / d}, {-m[1][0] / d, m[0][0] / d}}};
  return true;
}
inline Vec2 mul(const Mat2 & m, const Vec2 & v)
{
  return {m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]};
}
inline Mat2 mul(const Mat2 & a, const Mat2 & b)
{
  Mat2 r{};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];}
  }
  return r;
}
inline Mat2 transpose(const Mat2 & m) { return {{{m[0][0], m[1][0]}, {m[0][1], m[1][1]}}}; }

// ---- geometry ----------------------------------------------------------------------------------------------------
struct Crank
{
  Vec3 axis_pt{}, axis_dir{};   // crank (motor output) axis
  Vec3 crank_ball{};            // ball on the crank, at the CAD pose
  Vec3 far_ball{};              // ball at the other end of the rod, at the joint's zero pose
  double rod_len{0.0};
};

struct Geometry
{
  Vec3 axis1_pt{}, axis1_dir{};   // first joint axis (ankle pitch / waist roll), fixed in the base
  Vec3 axis2_pt{}, axis2_dir{};   // second joint axis (ankle roll / waist pitch), carried by the first
  bool crank_on_fixed{true};      // true: cranks on the fixed side, far balls move with the joint (ankles)
                                  // false: cranks ride on the moving side, far balls fixed (waist)
  std::array<Crank, 2> cranks{};
};

class Mechanism
{
public:
  Mechanism() = default;
  explicit Mechanism(const Geometry & g) { set_geometry(g); }

  void set_geometry(const Geometry & g)
  {
    g_ = g;
    g_.axis1_dir = unit(g_.axis1_dir);
    g_.axis2_dir = unit(g_.axis2_dir);
    for (auto & c : g_.cranks) {c.axis_dir = unit(c.axis_dir);}
    // the assembly the CAD shows: at the zero pose, the root nearest the crank's drawn position
    for (int i = 0; i < 2; ++i) {
      branch_[i] = 0;
      double plus = 0.0, minus = 0.0;
      branch_[i] = +1;
      const bool ok_p = motor_angle(i, {0.0, 0.0}, 0.0, plus);
      branch_[i] = -1;
      const bool ok_m = motor_angle(i, {0.0, 0.0}, 0.0, minus);
      branch_[i] = (ok_p && (!ok_m || std::fabs(plus) <= std::fabs(minus))) ? +1 : -1;
      zero_[i] = branch_[i] > 0 ? plus : minus;
    }
  }
  // crank angles at the zero pose (the offset between CAD crank positions and the joint zero pose)
  const Vec2 & zero_pose_motor_angles() const { return zero_; }
  const Geometry & geometry() const { return g_; }

  // Point carried by the moving side, at joint angles q (second axis first, then the first axis: a serial chain).
  Vec3 move(const Vec3 & p, const Vec2 & q) const
  {
    return rotate_about(rotate_about(p, g_.axis2_pt, g_.axis2_dir, q[1]), g_.axis1_pt, g_.axis1_dir, q[0]);
  }
  // Inverse of move().
  Vec3 unmove(const Vec3 & p, const Vec2 & q) const
  {
    return rotate_about(rotate_about(p, g_.axis1_pt, g_.axis1_dir, -q[0]), g_.axis2_pt, g_.axis2_dir, -q[1]);
  }

  // Crank angle of motor i at joint angles q, the root nearest `ref`. False when the rod cannot close (out of reach).
  bool motor_angle(int i, const Vec2 & q, double ref, double & out) const
  {
    const Crank & c = g_.cranks[i];
    // the far ball, in the frame the crank turns in
    const Vec3 f = g_.crank_on_fixed ? move(c.far_ball, q) : unmove(c.far_ball, q);
    const Vec3 r = sub(c.crank_ball, c.axis_pt);
    const Vec3 r_par = scale(c.axis_dir, dot(r, c.axis_dir));
    const Vec3 r_perp = sub(r, r_par);
    const Vec3 r_cross = cross(c.axis_dir, r_perp);
    const Vec3 d = sub(add(c.axis_pt, r_par), f);
    // |d + r_perp cos(th) + r_cross sin(th)|^2 = L^2  ->  P cos(th) + Q sin(th) = K
    const double P = 2.0 * dot(d, r_perp), Q = 2.0 * dot(d, r_cross);
    const double K = c.rod_len * c.rod_len - dot(d, d) - dot(r_perp, r_perp);
    const double R = std::hypot(P, Q);
    if (!(R > 0.0) || std::fabs(K) > R) {return false;}
    const double base = std::atan2(Q, P), half = std::acos(K / R);
    if (branch_[i] == 0) {                      // before set_geometry picked the branch: nearest root
      const double a = wrap_near(base + half, ref), b = wrap_near(base - half, ref);
      out = std::fabs(a - ref) <= std::fabs(b - ref) ? a : b;
    } else {
      out = wrap_near(base + branch_[i] * half, ref);
    }
    return true;
  }

  // Both motor angles at q; `ref` picks the branch (zeros: the CAD branch).
  bool motor_angles(const Vec2 & q, Vec2 & out, const Vec2 & ref = {0.0, 0.0}) const
  {
    return motor_angle(0, q, ref[0], out[0]) && motor_angle(1, q, ref[1], out[1]);
  }

  // J = d(motor angles)/d(joint angles) at q (central differences of the closed form, on the branch near `ref`).
  bool jacobian(const Vec2 & q, Mat2 & J, const Vec2 & ref = {0.0, 0.0}, double h = 1e-6) const
  {
    Vec2 th0;
    if (!motor_angles(q, th0, ref)) {return false;}
    for (int k = 0; k < 2; ++k) {
      Vec2 qp = q, qm = q;
      qp[k] += h;
      qm[k] -= h;
      Vec2 tp, tm;
      if (!motor_angles(qp, tp, th0) || !motor_angles(qm, tm, th0)) {return false;}
      J[0][k] = (tp[0] - tm[0]) / (2.0 * h);
      J[1][k] = (tp[1] - tm[1]) / (2.0 * h);
    }
    return true;
  }

  // Joint angles for measured motor angles `th`, Newton's method from `guess`, each step limited to `max_step` rad so
  // a far guess cannot overshoot. False when it does not converge (out of reach, or near a singular pose); `q` then
  // keeps the last iterate. Warm-started from the previous cycle it takes one or two iterations.
  bool joint_angles(
    const Vec2 & th, const Vec2 & guess, Vec2 & q, int max_iter = 40, double tol = 1e-12, double max_step = 0.2) const
  {
    q = guess;
    for (int it = 0; it < max_iter; ++it) {
      Vec2 t;
      Mat2 J, Ji;
      if (!motor_angles(q, t, th) || !jacobian(q, J, th) || !inverse(J, Ji)) {return false;}
      const Vec2 e{t[0] - th[0], t[1] - th[1]};
      if (std::fabs(e[0]) < tol && std::fabs(e[1]) < tol) {return reproduces(q, th);}
      const Vec2 dq = mul(Ji, e);
      const double big = std::max(std::fabs(dq[0]), std::fabs(dq[1]));
      const double s = big > max_step ? max_step / big : 1.0;
      q = {q[0] - s * dq[0], q[1] - s * dq[1]};
    }
    return reproduces(q, th);
  }

  // q gives exactly the motor angles th, without adding whole turns (angles within half a turn of the zero pose)
  bool reproduces(const Vec2 & q, const Vec2 & th) const
  {
    Vec2 t;
    return motor_angles(q, t, zero_) && std::fabs(t[0] - th[0]) < 1e-9 && std::fabs(t[1] - th[1]) < 1e-9;
  }

private:
  // angle equivalent to a (mod 2 pi) nearest to ref
  static double wrap_near(double a, double ref)
  {
    const double two_pi = 2.0 * M_PI;
    return a + two_pi * std::round((ref - a) / two_pi);
  }

  Geometry g_;
  std::array<int, 2> branch_{0, 0};   // which root each crank is assembled on (+1 / -1; 0 = not chosen yet)
  Vec2 zero_{};
};

// ---- joint space <-> motor space ---------------------------------------------------------------------------------
struct JointState2 { Vec2 position{}, velocity{}, effort{}; };
struct MotorState2 { Vec2 position{}, velocity{}, effort{}; };

// A joint-space command as a PD-in-the-drive (MyActuator mode 5 / Unitree PR) controller takes it.
struct JointCommand2 { Vec2 position{}, velocity{}, effort{}, kp{}, kd{}; };
struct MotorCommand2
{
  Vec2 position{}, velocity{}, effort{}, kp{}, kd{};
  double stiffness_coupling{0.0};   // |off-diagonal| / diagonal of the motor stiffness: what per-motor PD drops
};

// Motor stiffness for a joint stiffness diag(k) at Jacobian J: K_th = J^-T diag(k) J^-1.
inline bool motor_stiffness(const Mat2 & J, const Vec2 & k, Mat2 & out)
{
  Mat2 Ji;
  if (!inverse(J, Ji)) {return false;}
  const Mat2 Kq{{{k[0], 0.0}, {0.0, k[1]}}};
  out = mul(mul(transpose(Ji), Kq), Ji);
  return true;
}

// Measured motor state -> joint state. `prev_q` warm-starts the solve (use the last result each cycle).
inline bool to_joint(const Mechanism & m, const MotorState2 & s, const Vec2 & prev_q, JointState2 & out)
{
  Mat2 J, Ji;
  if (!m.joint_angles(s.position, prev_q, out.position) || !m.jacobian(out.position, J, s.position) ||
    !inverse(J, Ji))
  {
    return false;
  }
  out.velocity = mul(Ji, s.velocity);                    // qdot = J^-1 thdot
  out.effort = mul(transpose(J), s.effort);              // tau_joint = J^T tau_motor
  return true;
}

// Joint-space command -> per-motor command (targets through the linkage, feedforward torque through J^-T, gains
// from the motor stiffness diagonal). `q_now` is the measured joint position (for the torque and gain mapping);
// `motor_ref` the measured motor angles (branch reference).
inline bool to_motor(
  const Mechanism & m, const JointCommand2 & c, const Vec2 & q_now, const Vec2 & motor_ref, MotorCommand2 & out)
{
  Mat2 Jt, Jn, Jn_inv_t, Kth, Dth;
  if (!m.motor_angles(c.position, out.position, motor_ref) || !m.jacobian(c.position, Jt, out.position) ||
    !m.jacobian(q_now, Jn, motor_ref) || !inverse(transpose(Jn), Jn_inv_t) ||
    !motor_stiffness(Jn, c.kp, Kth) || !motor_stiffness(Jn, c.kd, Dth))
  {
    return false;
  }
  out.velocity = mul(Jt, c.velocity);                    // thdot* = J(q*) qdot*
  out.effort = mul(Jn_inv_t, c.effort);                  // tau_motor = J^-T tau_joint
  out.kp = {Kth[0][0], Kth[1][1]};
  out.kd = {Dth[0][0], Dth[1][1]};
  const double diag = std::sqrt(std::fabs(Kth[0][0] * Kth[1][1]));
  out.stiffness_coupling = diag > 0.0 ? std::fabs(Kth[0][1]) / diag : 0.0;
  return true;
}

// Exact joint-space PD computed on the host, as motor torques (for a torque-mode drive): tau_m = J^-T tau_q.
inline bool pd_motor_torque(
  const Mechanism & m, const JointCommand2 & c, const JointState2 & s, const Vec2 & motor_ref, Vec2 & tau_m)
{
  Mat2 J, Jit;
  if (!m.jacobian(s.position, J, motor_ref) || !inverse(transpose(J), Jit)) {return false;}
  Vec2 tau_q;
  for (int k = 0; k < 2; ++k) {
    tau_q[k] = c.kp[k] * (c.position[k] - s.position[k]) + c.kd[k] * (c.velocity[k] - s.velocity[k]) + c.effort[k];
  }
  tau_m = mul(Jit, tau_q);
  return true;
}

}  // namespace wisco_ethercat_ros2_hw::linkage
