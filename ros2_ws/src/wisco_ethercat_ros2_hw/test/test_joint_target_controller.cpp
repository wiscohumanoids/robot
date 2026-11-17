// JointTargetController against fake interfaces: a single-motor joint (left_knee: position + kp + kd) and a two-motor
// joint reached through ParallelLinkageController's reference interface (parallel_joints/left_ankle_pitch/position).
// Hold on start, ramp-in, limits, the policy manifest's gains, timeout (damp / hold) and recovery.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "hardware_interface/loaned_command_interface.hpp"
#include "hardware_interface/loaned_state_interface.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "wisco_ethercat_ros2_hw/joint_target_controller.hpp"

using wisco_ethercat_ros2_hw::JointTargetController;
using Phase = JointTargetController::Phase;

namespace
{
struct FakeBus
{
  // commands: knee position, knee kp, knee kd, ankle-pitch position (chained); states: knee, ankle pitch
  double knee_pos = 0.0, knee_kp = 0.0, knee_kd = 0.0, ankle_cmd = 0.0;
  double knee_meas = 0.6, ankle_meas = 0.1;
  std::vector<hardware_interface::CommandInterface> cmd;
  std::vector<hardware_interface::StateInterface> st;
  FakeBus()
  {
    cmd.reserve(4);
    cmd.emplace_back("left_knee", "position", &knee_pos);
    cmd.emplace_back("left_knee", "kp", &knee_kp);
    cmd.emplace_back("left_knee", "kd", &knee_kd);
    cmd.emplace_back("parallel_joints/left_ankle_pitch", "position", &ankle_cmd);
    st.reserve(2);
    st.emplace_back("left_knee", "position", &knee_meas);
    st.emplace_back("parallel_joints/left_ankle_pitch", "position", &ankle_meas);
  }
};

std::string read_file(const std::string & path)
{
  std::ifstream f(path);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

std::vector<rclcpp::Parameter> params(const std::string & action = "damp", bool urdf_limits = false)
{
  std::vector<rclcpp::Parameter> p{
    rclcpp::Parameter("joints", std::vector<std::string>{"left_knee", "left_ankle_pitch"}),
    rclcpp::Parameter("chained_joints", std::vector<std::string>{"left_ankle_pitch"}),
    rclcpp::Parameter("topic", std::string("/test/joint_targets")),
    rclcpp::Parameter("manifest_file", std::string(MANIFEST)),
    rclcpp::Parameter("gain_interfaces", std::vector<std::string>{"kp", "kd"}),
    rclcpp::Parameter("timeout", 0.05),
    rclcpp::Parameter("ramp_time", 0.1),
    rclcpp::Parameter("timeout_action", action),
    rclcpp::Parameter("limit_margin", 0.0),
  };
  if (!urdf_limits) {
    p.emplace_back("position_min", std::vector<double>{0.0, -0.5});
    p.emplace_back("position_max", std::vector<double>{2.0, 0.7});
  }
  return p;
}

struct Rig
{
  FakeBus bus;
  std::shared_ptr<JointTargetController> c = std::make_shared<JointTargetController>();
  rclcpp::Node::SharedPtr pub_node = std::make_shared<rclcpp::Node>("jtc_test_pub");
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub =
    pub_node->create_publisher<sensor_msgs::msg::JointState>("/test/joint_targets", 10);
  rclcpp::executors::SingleThreadedExecutor ex;

  explicit Rig(const std::vector<rclcpp::Parameter> & p, const std::string & urdf = "")
  {
    controller_interface::ControllerInterfaceParams ip;
    ip.controller_name = "leg_targets";
    ip.robot_description = urdf;
    ip.update_rate = 500;
    ip.controller_manager_update_rate = 500;
    ip.node_options = rclcpp::NodeOptions().parameter_overrides(p).allow_undeclared_parameters(true)
      .automatically_declare_parameters_from_overrides(true);
    EXPECT_EQ(c->init(ip), controller_interface::return_type::OK);
    EXPECT_EQ(c->configure().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
    std::vector<hardware_interface::LoanedCommandInterface> cmds;
    for (const auto & want : c->command_interface_configuration().names) {
      for (auto & ci : bus.cmd) {if (ci.get_name() == want) {cmds.emplace_back(ci);}}
    }
    std::vector<hardware_interface::LoanedStateInterface> sts;
    for (const auto & want : c->state_interface_configuration().names) {
      for (auto & si : bus.st) {if (si.get_name() == want) {sts.emplace_back(si);}}
    }
    EXPECT_EQ(cmds.size(), 4u);
    EXPECT_EQ(sts.size(), 2u);
    c->assign_interfaces(std::move(cmds), std::move(sts));
    EXPECT_EQ(c->get_node()->activate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
    ex.add_node(c->get_node()->get_node_base_interface());
    ex.add_node(pub_node);
  }
  void send(const std::vector<std::string> & names, const std::vector<double> & pos)
  {
    sensor_msgs::msg::JointState m;
    m.name = names;
    m.position = pos;
    pub->publish(m);
    for (int i = 0; i < 20; ++i) {ex.spin_some(std::chrono::milliseconds(2));}
  }
  void tick(double dt = 0.01)
  {
    ASSERT_EQ(c->update(rclcpp::Time(0, 0, RCL_ROS_TIME), rclcpp::Duration::from_seconds(dt)),
      controller_interface::return_type::OK);
  }
};
}  // namespace

TEST(JointTargetController, HoldsTheMeasuredPoseUntilTargetsArrive)
{
  Rig r(params());
  r.tick();
  EXPECT_EQ(r.c->phase(), Phase::kHold);
  EXPECT_DOUBLE_EQ(r.bus.knee_pos, 0.6);
  EXPECT_DOUBLE_EQ(r.bus.ankle_cmd, 0.1);
  EXPECT_DOUBLE_EQ(r.bus.knee_kp, 120.0);                 // the trained policy's gains, from its manifest
  EXPECT_GT(r.bus.knee_kd, 0.0);
}

TEST(JointTargetController, RampsInThenTracks)
{
  Rig r(params());
  r.send({"left_knee", "left_ankle_pitch"}, {1.0, 0.3});
  r.tick(0.05);                                            // half of the 0.1 s ramp
  EXPECT_EQ(r.c->phase(), Phase::kRamp);
  EXPECT_NEAR(r.bus.knee_pos, 0.6 + 0.5 * (1.0 - 0.6), 1e-9);
  EXPECT_NEAR(r.bus.ankle_cmd, 0.1 + 0.5 * (0.3 - 0.1), 1e-9);
  r.send({"left_knee", "left_ankle_pitch"}, {1.0, 0.3});
  r.tick(0.05);
  EXPECT_EQ(r.c->phase(), Phase::kTrack);
  EXPECT_NEAR(r.bus.knee_pos, 1.0, 1e-12);
  EXPECT_NEAR(r.bus.ankle_cmd, 0.3, 1e-12);
}

TEST(JointTargetController, ClampsToTheJointLimitsAndKeepsMissingJoints)
{
  Rig r(params());
  r.send({"left_knee", "left_ankle_pitch"}, {5.0, 0.2});
  r.tick(0.2);
  EXPECT_NEAR(r.bus.knee_pos, 2.0, 1e-12);                // limit
  EXPECT_EQ(r.c->clamped(), 1u);
  r.send({"left_knee", "some_arm_joint"}, {1.5, 9.9});    // ankle left out, unknown name ignored
  r.tick();
  EXPECT_NEAR(r.bus.knee_pos, 1.5, 1e-12);
  EXPECT_NEAR(r.bus.ankle_cmd, 0.2, 1e-12);
}

TEST(JointTargetController, TimeoutDampsThenRampsBackIn)
{
  Rig r(params("damp"));
  r.send({"left_knee", "left_ankle_pitch"}, {1.0, 0.3});
  r.tick(0.2);
  ASSERT_EQ(r.c->phase(), Phase::kTrack);
  r.bus.knee_meas = 0.9;
  r.tick(0.08);                                           // 80 ms of control loop with no new targets (timeout 50 ms)
  EXPECT_EQ(r.c->phase(), Phase::kSafe);
  EXPECT_DOUBLE_EQ(r.bus.knee_kp, 0.0);                   // limp ...
  EXPECT_DOUBLE_EQ(r.bus.knee_kd, 8.0);                   // ... and damped
  EXPECT_NEAR(r.bus.ankle_cmd, 0.1, 1e-12);               // two-motor joint: holds where it is (measured 0.1)
  r.send({"left_knee", "left_ankle_pitch"}, {1.2, 0.3});  // targets return
  r.tick(0.05);
  EXPECT_EQ(r.c->phase(), Phase::kRamp);
  EXPECT_NEAR(r.bus.knee_pos, 0.9 + 0.5 * (1.2 - 0.9), 1e-9);    // from where the joint actually is
  EXPECT_DOUBLE_EQ(r.bus.knee_kp, 120.0);
}

TEST(JointTargetController, TimeoutCanHoldInstead)
{
  Rig r(params("hold"));
  r.send({"left_knee", "left_ankle_pitch"}, {1.0, 0.3});
  r.tick(0.2);
  r.bus.knee_meas = 0.95;
  r.tick(0.08);
  EXPECT_EQ(r.c->phase(), Phase::kSafe);
  EXPECT_NEAR(r.bus.knee_pos, 0.95, 1e-12);
  EXPECT_DOUBLE_EQ(r.bus.knee_kp, 120.0);
}

TEST(JointTargetController, LimitsComeFromTheUrdf)
{
  Rig r(params("damp", /*urdf_limits=*/true), read_file(WISCO_URDF));
  r.send({"left_knee", "left_ankle_pitch"}, {9.0, -9.0});
  r.tick(0.2);
  // robots/wisco_v1/robot.urdf: knee upper limit, ankle pitch lower limit
  EXPECT_LT(r.bus.knee_pos, 9.0);
  EXPECT_GT(r.bus.ankle_cmd, -9.0);
  EXPECT_EQ(r.c->clamped(), 2u);
}

TEST(JointTargetController, HoldServiceRearmsAtTheMeasuredPose)
{
  // the robot is moved by something else (a world reset): the controller must hold where it now is, not pull it
  // back to the old targets, and ramp in again only from messages sent after the request
  Rig r(params());
  r.send({"left_knee", "left_ankle_pitch"}, {1.0, 0.3});
  r.tick(0.2);
  ASSERT_EQ(r.c->phase(), Phase::kTrack);
  auto client = r.pub_node->create_client<std_srvs::srv::Trigger>("/leg_targets/hold");
  ASSERT_TRUE(client->wait_for_service(std::chrono::seconds(2)));
  auto fut = client->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
  for (int i = 0; i < 50 && fut.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready; ++i) {
    r.ex.spin_some(std::chrono::milliseconds(5));
  }
  ASSERT_TRUE(fut.get()->success);
  r.bus.knee_meas = 0.5;                                  // the world was reset: the knee is somewhere else now
  r.bus.ankle_meas = -0.1;
  r.tick();
  EXPECT_EQ(r.c->phase(), Phase::kHold);                  // the last message is older than the request
  EXPECT_DOUBLE_EQ(r.bus.knee_pos, 0.5);
  EXPECT_DOUBLE_EQ(r.bus.ankle_cmd, -0.1);
  r.send({"left_knee", "left_ankle_pitch"}, {1.0, 0.3});  // new targets: ramp from the new pose
  r.tick(0.05);
  EXPECT_EQ(r.c->phase(), Phase::kRamp);
  EXPECT_NEAR(r.bus.knee_pos, 0.5 + 0.5 * (1.0 - 0.5), 1e-9);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int r = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return r;
}
