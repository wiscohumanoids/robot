// ParallelLinkageController against fake motor interfaces whose readings come from the CAD linkage: it reports the
// joint state the motors imply, turns joint references into motor commands (both modes), holds on startup, refuses
// unreachable targets, follows ~/commands when not chained, and honours motor directions.
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/loaned_command_interface.hpp"
#include "hardware_interface/loaned_state_interface.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "wisco_ethercat_ros2_hw/parallel_linkage_controller.hpp"
#include "wisco_ethercat_ros2_hw/parallel_linkage_yaml.hpp"

using wisco_ethercat_ros2_hw::ParallelLinkageController;
namespace lk = wisco_ethercat_ros2_hw::linkage;

namespace
{
// Two ankle motors on a fake bus. Positions are hardware-frame: 0 at the joint zero pose, times `dir`.
struct FakeAnkle
{
  std::vector<std::string> cmd_names;
  std::vector<double> cmd, state;                 // state per motor: position, velocity, effort
  std::vector<hardware_interface::CommandInterface> cmd_ifaces;
  std::vector<hardware_interface::StateInterface> state_ifaces;
  lk::Mechanism mech;
  lk::Vec2 offset{};
  std::array<double, 2> dir{1.0, 1.0};

  explicit FakeAnkle(const std::vector<std::string> & ifaces, std::array<double, 2> d = {1.0, 1.0})
  : cmd_names(ifaces), dir(d)
  {
    mech.set_geometry(lk::load_mechanism(LINKAGE_JSON, "left_ankle").geometry);
    mech.motor_angles({0.0, 0.0}, offset);
    cmd.assign(2 * ifaces.size(), 0.0);
    state.assign(6, 0.0);
    cmd_ifaces.reserve(cmd.size());
    for (int m = 0; m < 2; ++m) {
      for (std::size_t i = 0; i < ifaces.size(); ++i) {
        cmd_ifaces.emplace_back(motor(m), ifaces[i], &cmd[m * ifaces.size() + i]);
      }
    }
    state_ifaces.reserve(6);
    for (int m = 0; m < 2; ++m) {
      state_ifaces.emplace_back(motor(m), "position", &state[3 * m]);
      state_ifaces.emplace_back(motor(m), "velocity", &state[3 * m + 1]);
      state_ifaces.emplace_back(motor(m), "effort", &state[3 * m + 2]);
    }
  }
  static std::string motor(int m) { return m == 0 ? "left_ankle_motor_a" : "left_ankle_motor_b"; }

  // put the ankle at joint pose q (moving at qd): motor readings as the drives would report them
  void set_joint(const lk::Vec2 & q, const lk::Vec2 & qd = {0.0, 0.0})
  {
    lk::Vec2 th;
    lk::Mat2 J;
    ASSERT_TRUE(mech.motor_angles(q, th) && mech.jacobian(q, J, th));
    const lk::Vec2 thd = lk::mul(J, qd);
    for (int m = 0; m < 2; ++m) {
      state[3 * m] = dir[m] * (th[m] - offset[m]);
      state[3 * m + 1] = dir[m] * thd[m];
      state[3 * m + 2] = 0.0;
    }
  }
  double command(int m, const std::string & iface) const
  {
    for (std::size_t i = 0; i < cmd_names.size(); ++i) {
      if (cmd_names[i] == iface) {return cmd[m * cmd_names.size() + i];}
    }
    return std::nan("");
  }
};

std::vector<rclcpp::Parameter> params(const std::string & mode, const std::vector<std::string> & ifaces,
  std::vector<double> dirs = {1.0, 1.0})
{
  return {
    rclcpp::Parameter("linkage_file", std::string(LINKAGE_JSON)),
    rclcpp::Parameter("mechanisms", std::vector<std::string>{"left_ankle"}),
    rclcpp::Parameter("command_mode", mode),
    rclcpp::Parameter("motor_command_interfaces", ifaces),
    rclcpp::Parameter("left_ankle.motors",
      std::vector<std::string>{"left_ankle_motor_a", "left_ankle_motor_b"}),
    rclcpp::Parameter("left_ankle.motor_directions", dirs),
    rclcpp::Parameter("left_ankle.kp", std::vector<double>{35.0, 35.0}),
    rclcpp::Parameter("left_ankle.kd", std::vector<double>{2.14, 0.88}),
    rclcpp::Parameter("left_ankle.position_min", std::vector<double>{-0.546, -0.2618}),   // robot.xml
    rclcpp::Parameter("left_ankle.position_max", std::vector<double>{0.7854, 0.2618}),
  };
}

// init, configure, wire, export, activate
std::shared_ptr<ParallelLinkageController> load(FakeAnkle & hw, const std::vector<rclcpp::Parameter> & p,
  std::vector<hardware_interface::CommandInterface::SharedPtr> & refs,
  std::vector<hardware_interface::StateInterface::ConstSharedPtr> & states, bool chained = true)
{
  auto c = std::make_shared<ParallelLinkageController>();
  controller_interface::ControllerInterfaceParams ip;
  ip.controller_name = "linkage";
  ip.update_rate = 500;
  ip.controller_manager_update_rate = 500;
  ip.node_options = rclcpp::NodeOptions().parameter_overrides(p).allow_undeclared_parameters(true)
    .automatically_declare_parameters_from_overrides(true);
  EXPECT_EQ(c->init(ip), controller_interface::return_type::OK);
  EXPECT_EQ(c->configure().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  std::vector<hardware_interface::LoanedCommandInterface> cmds;
  for (const auto & want : c->command_interface_configuration().names) {
    for (auto & ci : hw.cmd_ifaces) {if (ci.get_name() == want) {cmds.emplace_back(ci);}}
  }
  std::vector<hardware_interface::LoanedStateInterface> sts;
  for (const auto & want : c->state_interface_configuration().names) {
    for (auto & si : hw.state_ifaces) {if (si.get_name() == want) {sts.emplace_back(si);}}
  }
  EXPECT_EQ(cmds.size(), c->command_interface_configuration().names.size());
  EXPECT_EQ(sts.size(), c->state_interface_configuration().names.size());
  c->assign_interfaces(std::move(cmds), std::move(sts));
  refs = c->export_reference_interfaces();
  states = c->export_state_interfaces();
  c->set_chained_mode(chained);
  EXPECT_EQ(c->get_node()->activate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  return c;
}

double get(const std::vector<hardware_interface::StateInterface::ConstSharedPtr> & s, const std::string & name)
{
  for (const auto & i : s) {if (i->get_name() == name) {return i->get_value();}}
  return std::nan("");
}
void set(const std::vector<hardware_interface::CommandInterface::SharedPtr> & r, const std::string & name, double v)
{
  for (const auto & i : r) {if (i->get_name() == name) {(void)i->set_value(v);}}
}
void tick(ParallelLinkageController & c)
{
  ASSERT_EQ(c.update(rclcpp::Time(0, 0, RCL_ROS_TIME), rclcpp::Duration::from_seconds(0.002)),
    controller_interface::return_type::OK);
}
const std::vector<std::string> kPdIfaces{"position", "velocity", "effort", "kp", "kd"};
}  // namespace

TEST(ParallelLinkageController, ExportsTheSerialJointsAndTheirState)
{
  FakeAnkle hw(kPdIfaces);
  hw.set_joint({0.2, -0.1}, {0.5, 0.2});
  std::vector<hardware_interface::CommandInterface::SharedPtr> refs;
  std::vector<hardware_interface::StateInterface::ConstSharedPtr> st;
  auto c = load(hw, params("pd_in_drive", kPdIfaces), refs, st);
  ASSERT_EQ(refs.size(), 6u);                                     // 2 joints x position, velocity, effort
  EXPECT_EQ(refs[0]->get_name(), "linkage/left_ankle_pitch/position");
  tick(*c);
  EXPECT_NEAR(get(st, "linkage/left_ankle_pitch/position"), 0.2, 1e-9);
  EXPECT_NEAR(get(st, "linkage/left_ankle_roll/position"), -0.1, 1e-9);
  EXPECT_NEAR(get(st, "linkage/left_ankle_pitch/velocity"), 0.5, 1e-6);
  EXPECT_NEAR(get(st, "linkage/left_ankle_roll/velocity"), 0.2, 1e-6);
}

TEST(ParallelLinkageController, HoldsWhereItStartsThenFollowsReferences)
{
  FakeAnkle hw(kPdIfaces);
  hw.set_joint({0.1, 0.05});
  std::vector<hardware_interface::CommandInterface::SharedPtr> refs;
  std::vector<hardware_interface::StateInterface::ConstSharedPtr> st;
  auto c = load(hw, params("pd_in_drive", kPdIfaces), refs, st);
  tick(*c);
  for (int m = 0; m < 2; ++m) {EXPECT_NEAR(hw.command(m, "position"), hw.state[3 * m], 1e-9);}   // hold
  set(refs, "linkage/left_ankle_pitch/position", 0.3);
  set(refs, "linkage/left_ankle_roll/position", -0.12);
  tick(*c);
  lk::Vec2 th, q;
  for (int m = 0; m < 2; ++m) {th[m] = hw.command(m, "position") + hw.offset[m];}
  ASSERT_TRUE(hw.mech.joint_angles(th, {0.1, 0.05}, q));        // the motor targets ARE the joint target
  EXPECT_NEAR(q[0], 0.3, 1e-9);
  EXPECT_NEAR(q[1], -0.12, 1e-9);
  EXPECT_GT(hw.command(0, "kp"), 0.0);
  EXPECT_GT(hw.command(1, "kd"), 0.0);
  EXPECT_GT(c->max_stiffness_coupling(), 0.0);
  EXPECT_EQ(c->reach_failures(), 0u);
}

TEST(ParallelLinkageController, TorqueModeIsTheJointPdThroughTheLinkage)
{
  FakeAnkle hw({"effort"});
  hw.set_joint({0.0, 0.0});
  std::vector<hardware_interface::CommandInterface::SharedPtr> refs;
  std::vector<hardware_interface::StateInterface::ConstSharedPtr> st;
  auto c = load(hw, params("torque", {"effort"}), refs, st);
  set(refs, "linkage/left_ankle_pitch/position", 0.1);            // 0.1 rad pitch error, no roll error
  tick(*c);
  lk::Vec2 th;
  lk::Mat2 J;
  ASSERT_TRUE(hw.mech.motor_angles({0.0, 0.0}, th) && hw.mech.jacobian({0.0, 0.0}, J, th));
  const lk::Vec2 tau_m{hw.command(0, "effort"), hw.command(1, "effort")};
  const lk::Vec2 tau_q = lk::mul(lk::transpose(J), tau_m);
  EXPECT_NEAR(tau_q[0], 35.0 * 0.1, 1e-6);                       // joint torque = kp * error
  EXPECT_NEAR(tau_q[1], 0.0, 1e-6);
}

TEST(ParallelLinkageController, MotorDirectionsAreApplied)
{
  // the same physical pose read through flipped motors gives the same joints, and commands come back flipped
  FakeAnkle hw(kPdIfaces, {-1.0, 1.0});
  hw.set_joint({0.15, 0.1});
  std::vector<hardware_interface::CommandInterface::SharedPtr> refs;
  std::vector<hardware_interface::StateInterface::ConstSharedPtr> st;
  auto c = load(hw, params("pd_in_drive", kPdIfaces, {-1.0, 1.0}), refs, st);
  tick(*c);
  EXPECT_NEAR(get(st, "linkage/left_ankle_pitch/position"), 0.15, 1e-9);
  EXPECT_NEAR(get(st, "linkage/left_ankle_roll/position"), 0.1, 1e-9);
  for (int m = 0; m < 2; ++m) {EXPECT_NEAR(hw.command(m, "position"), hw.state[3 * m], 1e-9);}
}

TEST(ParallelLinkageController, ImpossibleReadingsHoldTheMotors)
{
  FakeAnkle hw(kPdIfaces);
  hw.set_joint({0.0, 0.0});
  std::vector<hardware_interface::CommandInterface::SharedPtr> refs;
  std::vector<hardware_interface::StateInterface::ConstSharedPtr> st;
  auto c = load(hw, params("pd_in_drive", kPdIfaces), refs, st);
  hw.state[0] = 3.0;                                              // motor A far past what the rods allow
  hw.state[3] = -3.0;
  tick(*c);
  EXPECT_EQ(c->solve_failures(), 1u);
  EXPECT_NEAR(hw.command(0, "position"), 3.0, 1e-12);            // held where it is, no push
  EXPECT_NEAR(hw.command(1, "position"), -3.0, 1e-12);
  EXPECT_NEAR(hw.command(0, "effort"), 0.0, 1e-12);
}

TEST(ParallelLinkageController, ReferencesAreClampedToTheJointLimits)
{
  FakeAnkle hw(kPdIfaces);
  hw.set_joint({0.0, 0.0});
  std::vector<hardware_interface::CommandInterface::SharedPtr> refs;
  std::vector<hardware_interface::StateInterface::ConstSharedPtr> st;
  auto c = load(hw, params("pd_in_drive", kPdIfaces), refs, st);
  set(refs, "linkage/left_ankle_roll/position", 0.6);              // limit 0.2618
  tick(*c);
  lk::Vec2 th, q;
  for (int m = 0; m < 2; ++m) {th[m] = hw.command(m, "position") + hw.offset[m];}
  ASSERT_TRUE(hw.mech.joint_angles(th, {0.0, 0.0}, q));
  EXPECT_NEAR(q[1], 0.2618, 1e-9);
}

TEST(ParallelLinkageController, CommandsTopicDrivesItWhenNotChained)
{
  FakeAnkle hw(kPdIfaces);
  hw.set_joint({0.0, 0.0});
  std::vector<hardware_interface::CommandInterface::SharedPtr> refs;
  std::vector<hardware_interface::StateInterface::ConstSharedPtr> st;
  auto c = load(hw, params("pd_in_drive", kPdIfaces), refs, st, /*chained=*/false);
  auto pub_node = std::make_shared<rclcpp::Node>("linkage_test_pub");
  auto pub = pub_node->create_publisher<std_msgs::msg::Float64MultiArray>("/linkage/commands", 10);
  rclcpp::executors::SingleThreadedExecutor ex;
  ex.add_node(c->get_node()->get_node_base_interface());
  ex.add_node(pub_node);
  std_msgs::msg::Float64MultiArray msg;
  msg.data = {0.25, 0.05};
  bool moved = false;
  for (int i = 0; i < 100 && !moved; ++i) {
    pub->publish(msg);
    ex.spin_some(std::chrono::milliseconds(10));
    tick(*c);
    lk::Vec2 th, q;
    for (int m = 0; m < 2; ++m) {th[m] = hw.command(m, "position") + hw.offset[m];}
    moved = hw.mech.joint_angles(th, {0.0, 0.0}, q) && std::fabs(q[0] - 0.25) < 1e-9 && std::fabs(q[1] - 0.05) < 1e-9;
  }
  EXPECT_TRUE(moved);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int r = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return r;
}
