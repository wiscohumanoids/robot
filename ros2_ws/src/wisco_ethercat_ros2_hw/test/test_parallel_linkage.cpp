// Parallel-joint conversion (include/wisco_ethercat_ros2_hw/parallel_linkage.hpp) against the Python reference
// (robots/wisco_v1/linkage.py, via test/data/linkage_parity.json from tools/make_linkage_golden.py), plus the
// physics the conversion must respect: round trips, power balance, stiffness mapping, unreachable poses refused.
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>

#include <yaml-cpp/yaml.h>

#include "wisco_ethercat_ros2_hw/parallel_linkage.hpp"
#include "wisco_ethercat_ros2_hw/parallel_linkage_yaml.hpp"

using namespace wisco_ethercat_ros2_hw::linkage;

namespace
{
const char * kMechanisms[] = {"left_ankle", "right_ankle", "waist"};

Mechanism load(const std::string & name) { return Mechanism(load_mechanism(LINKAGE_JSON, name).geometry); }
YAML::Node golden() { return YAML::LoadFile(LINKAGE_GOLDEN)["mechanisms"]; }
Vec2 vec2(const YAML::Node & n) { return {n[0].as<double>(), n[1].as<double>()}; }
}  // namespace

TEST(ParallelLinkage, LoadsTheCadGeometry)
{
  const auto m = load_mechanism(LINKAGE_JSON, "left_ankle");
  EXPECT_EQ(m.joints[0], "left_ankle_pitch");
  EXPECT_EQ(m.joints[1], "left_ankle_roll");
  EXPECT_TRUE(m.geometry.crank_on_fixed);
  EXPECT_FALSE(load_mechanism(LINKAGE_JSON, "waist").geometry.crank_on_fixed);
  EXPECT_THROW(load_mechanism(LINKAGE_JSON, "no_such_joint"), std::runtime_error);
}

TEST(ParallelLinkage, ClosedFormMatchesThePythonReference)
{
  const YAML::Node g = golden();
  for (const char * name : kMechanisms) {
    const Mechanism m = load(name);
    for (const auto & c : g[name]["cases"]) {
      const Vec2 q = vec2(c["q"]), th_ref = vec2(c["motor"]);
      Vec2 th;
      ASSERT_TRUE(m.motor_angles(q, th)) << name;
      EXPECT_NEAR(th[0], th_ref[0], 1e-9) << name;
      EXPECT_NEAR(th[1], th_ref[1], 1e-9) << name;
      Mat2 J;
      ASSERT_TRUE(m.jacobian(q, J, th));
      for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 2; ++k) {EXPECT_NEAR(J[i][k], c["jacobian"][i][k].as<double>(), 1e-5) << name;}
      }
    }
  }
}

TEST(ParallelLinkage, MotorsToJointsInvertsJointsToMotors)
{
  const YAML::Node g = golden();
  for (const char * name : kMechanisms) {
    const Mechanism m = load(name);
    Vec2 prev{0.0, 0.0};
    for (const auto & c : g[name]["cases"]) {
      const Vec2 q_ref = vec2(c["q"]), th = vec2(c["motor"]);
      Vec2 q;
      ASSERT_TRUE(m.joint_angles(th, {0.0, 0.0}, q)) << name << ": from a cold start";
      EXPECT_NEAR(q[0], q_ref[0], 1e-10);
      EXPECT_NEAR(q[1], q_ref[1], 1e-10);
      ASSERT_TRUE(m.joint_angles(th, prev, q)) << name << ": from the previous pose";
      prev = q;
    }
  }
}

TEST(ParallelLinkage, StateConversionConservesPower)
{
  // tau_motor . thdot == tau_joint . qdot: J^T for torque and J^-1 for velocity are consistent
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (const char * name : kMechanisms) {
    const Mechanism m = load(name);
    for (int k = 0; k < 20; ++k) {
      const Vec2 q{0.3 * u(rng), 0.2 * u(rng)};
      MotorState2 s;
      ASSERT_TRUE(m.motor_angles(q, s.position));
      s.velocity = {u(rng), u(rng)};
      s.effort = {10 * u(rng), 10 * u(rng)};
      JointState2 j;
      ASSERT_TRUE(to_joint(m, s, {0.0, 0.0}, j));
      EXPECT_NEAR(j.position[0], q[0], 1e-10);
      EXPECT_NEAR(j.position[1], q[1], 1e-10);
      const double p_motor = s.effort[0] * s.velocity[0] + s.effort[1] * s.velocity[1];
      const double p_joint = j.effort[0] * j.velocity[0] + j.effort[1] * j.velocity[1];
      EXPECT_NEAR(p_motor, p_joint, 1e-6 * (1.0 + std::fabs(p_motor))) << name;
    }
  }
}

TEST(ParallelLinkage, JointVelocityMatchesFiniteMotion)
{
  // a small joint motion dq moves the motors by J dq
  const Mechanism m = load("left_ankle");
  const Vec2 q{0.2, -0.1}, dq{1e-6, -2e-6};
  Vec2 t0, t1;
  Mat2 J;
  ASSERT_TRUE(m.motor_angles(q, t0) && m.motor_angles({q[0] + dq[0], q[1] + dq[1]}, t1) && m.jacobian(q, J, t0));
  const Vec2 pred = mul(J, dq);
  EXPECT_NEAR(t1[0] - t0[0], pred[0], 1e-10);
  EXPECT_NEAR(t1[1] - t0[1], pred[1], 1e-10);
}

TEST(ParallelLinkage, MotorStiffnessReproducesTheJointStiffness)
{
  // full 2x2 motor stiffness K_th = J^-T K_q J^-1: a joint deflection dq gives joint torque -K_q dq exactly
  const Mechanism m = load("right_ankle");
  const Vec2 q{0.1, 0.05}, kq{35.0, 35.0}, dq{1e-4, -2e-4};
  Vec2 th;
  Mat2 J, Kth;
  ASSERT_TRUE(m.motor_angles(q, th) && m.jacobian(q, J, th) && motor_stiffness(J, kq, Kth));
  const Vec2 dth = mul(J, dq);
  const Vec2 tau_m{-(Kth[0][0] * dth[0] + Kth[0][1] * dth[1]), -(Kth[1][0] * dth[0] + Kth[1][1] * dth[1])};
  const Vec2 tau_q = mul(transpose(J), tau_m);
  EXPECT_NEAR(tau_q[0], -kq[0] * dq[0], 1e-9);
  EXPECT_NEAR(tau_q[1], -kq[1] * dq[1], 1e-9);
}

TEST(ParallelLinkage, CommandToMotorsHitsTheTargetAndReportsCoupling)
{
  const Mechanism m = load("left_ankle");
  JointCommand2 c;
  c.position = {0.3, -0.15};
  c.velocity = {0.5, 0.0};
  c.effort = {5.0, -2.0};
  c.kp = {35.0, 35.0};
  c.kd = {2.1, 0.9};
  Vec2 th_now;
  ASSERT_TRUE(m.motor_angles({0.25, -0.1}, th_now));
  MotorCommand2 out;
  ASSERT_TRUE(to_motor(m, c, {0.25, -0.1}, th_now, out));
  Vec2 q;
  ASSERT_TRUE(m.joint_angles(out.position, {0.25, -0.1}, q));      // the motor targets are the joint target
  EXPECT_NEAR(q[0], 0.3, 1e-10);
  EXPECT_NEAR(q[1], -0.15, 1e-10);
  EXPECT_GT(out.kp[0], 0.0);
  EXPECT_GT(out.kp[1], 0.0);
  EXPECT_GT(out.kd[0], 0.0);
  EXPECT_GE(out.stiffness_coupling, 0.0);
  EXPECT_LT(out.stiffness_coupling, 1.0);
}

TEST(ParallelLinkage, HostPdIsTheJointPdThroughTheLinkage)
{
  const Mechanism m = load("waist");
  JointCommand2 c;
  c.position = {0.1, 0.0};
  c.kp = {100.0, 100.0};
  c.kd = {5.0, 5.0};
  JointState2 s;
  s.position = {0.0, 0.05};
  s.velocity = {0.2, 0.0};
  Vec2 th;
  ASSERT_TRUE(m.motor_angles(s.position, th));
  Vec2 tau_m;
  ASSERT_TRUE(pd_motor_torque(m, c, s, th, tau_m));
  Mat2 J;
  ASSERT_TRUE(m.jacobian(s.position, J, th));
  const Vec2 tau_q = mul(transpose(J), tau_m);                     // back to the joints
  EXPECT_NEAR(tau_q[0], 100.0 * 0.1 - 5.0 * 0.2, 1e-9);
  EXPECT_NEAR(tau_q[1], 100.0 * -0.05, 1e-9);
}

TEST(ParallelLinkage, UnreachablePosesAreRefused)
{
  // a rod far too short to reach its far ball: no crank angle closes it, so nothing is commanded or reported
  Geometry g = load_mechanism(LINKAGE_JSON, "left_ankle").geometry;
  g.cranks[0].rod_len = 0.01;
  const Mechanism m(g);
  Vec2 th, q;
  EXPECT_FALSE(m.motor_angles({0.0, 0.0}, th));
  EXPECT_FALSE(m.joint_angles({0.0, 0.0}, {0.0, 0.0}, q));
  MotorCommand2 out;
  EXPECT_FALSE(to_motor(m, JointCommand2{}, {0.0, 0.0}, {0.0, 0.0}, out));
}
