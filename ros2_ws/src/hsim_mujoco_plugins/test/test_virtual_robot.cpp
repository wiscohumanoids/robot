// VirtualRobotSystem on Wisco V1 in the test room (assets/envs/test/table_room, built by tools/build/env_build.py; the
// test skips when it is not built): the drives come up through the CiA402 state machine into mode 5, the two-motor
// mechanisms read what the rod geometry says, and mode-5 holds from the bus keep the robot upright.
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "hardware_interface/component_parser.hpp"
#include "hsim_mujoco_plugins/virtual_robot_system.hpp"
#include "rclcpp/rclcpp.hpp"
#include "wisco_ethercat_ros2_hw/mock/cia402.hpp"
#include "wisco_ethercat_ros2_hw/parallel_linkage_yaml.hpp"

namespace
{

const std::string kWorld = std::string(REPO) + "/assets/envs/test/table_room/build_wisco/world_ros2_control/world.xml";
const std::string kLinkage = std::string(REPO) + "/robots/wisco_v1/linkage.json";

struct MotorSpec
{
  std::string name, mechanism;
  int crank = -1;
  double peak = 0.0, kp = 0.0, kd = 0.0;
};

// Wisco's motors: one per joint, except the ankles and the waist's roll/pitch (two per mechanism, X4-36 / X6-60).
std::vector<MotorSpec> motors_of(const mjModel * m)
{
  const std::map<std::string, std::pair<std::string, int>> linkage_joint = {
    {"left_ankle_pitch", {"left_ankle", 0}}, {"left_ankle_roll", {"left_ankle", 1}},
    {"right_ankle_pitch", {"right_ankle", 0}}, {"right_ankle_roll", {"right_ankle", 1}},
    {"torso_roll", {"waist", 0}}, {"torso_pitch", {"waist", 1}}};
  std::vector<MotorSpec> out;
  for (int a = 0; a < m->nu; ++a) {
    const int j = m->actuator_trnid[2 * a];
    const std::string name = m->names + m->name_jntadr[j];
    const double kp = m->actuator_gainprm[a * mjNGAIN], kd = -m->actuator_biasprm[a * mjNBIAS + 2];
    const auto it = linkage_joint.find(name);
    if (it == linkage_joint.end()) {
      out.push_back({name, "", -1, m->actuator_forcerange[2 * a + 1], kp, kd});
    } else {
      const auto & [mech, crank] = it->second;
      const double peak = mech == "waist" ? 60.0 : 34.0;
      out.push_back({mech + (crank == 0 ? "_motor_a" : "_motor_b"), mech, crank, peak, kp, kd});
    }
  }
  return out;
}

std::string urdf(const std::vector<MotorSpec> & motors, bool hold_until_released)
{
  std::ostringstream u;
  u << "<robot name='t'><link name='base'/>";
  for (const auto & mo : motors) {                          // the parser wants each ros2_control joint in the URDF
    u << "<link name='l_" << mo.name << "'/><joint name='" << mo.name << "' type='continuous'><parent link='base'/>"
      << "<child link='l_" << mo.name << "'/><axis xyz='0 0 1'/></joint>";
  }
  u << "<ros2_control name='VirtualRobot' type='system'><hardware>"
    << "<plugin>hsim_mujoco_plugins/VirtualRobotSystem</plugin>"
    << "<param name='mujoco_model'>" << kWorld << "</param>"
    << "<param name='linkage_file'>" << kLinkage << "</param>"
    << "<param name='control_period'>0.005</param><param name='real_time_factor'>0</param>"
    << "<param name='state_rate'>0</param><param name='node_name'>virtual_robot_test</param>"
    << "<param name='hold_until_released'>" << (hold_until_released ? "true" : "false") << "</param>"
    << "</hardware>";
  for (const auto & mo : motors) {
    u << "<joint name='" << mo.name << "'><command_interface name='position'/><state_interface name='position'/>"
      << "<param name='peak_effort_nm'>" << mo.peak << "</param>"
      << "<param name='effort_nm_per_permille'>" << mo.peak / 3000.0 << "</param>";
    if (!mo.mechanism.empty()) {
      u << "<param name='mechanism'>" << mo.mechanism << "</param><param name='crank'>" << mo.crank << "</param>";
    }
    u << "</joint>";
  }
  u << "<sensor name='imu'><state_interface name='orientation.x'/></sensor></ros2_control></robot>";
  return u.str();
}

class VirtualRobotTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void start(bool hold_until_released)
  {
    if (!std::filesystem::exists(kWorld)) {GTEST_SKIP() << kWorld << " not built (tools/build/env_build.py)";}
    char err[1000];
    model_ = mj_loadXML(kWorld.c_str(), nullptr, err, sizeof(err));
    ASSERT_NE(model_, nullptr) << err;
    motors_ = motors_of(model_);
    hardware_interface::HardwareComponentParams params;
    params.hardware_info = hardware_interface::parse_control_resources_from_urdf(
      urdf(motors_, hold_until_released)).front();
    params.clock = std::make_shared<rclcpp::Clock>();
    params.logger = rclcpp::get_logger("test_virtual_robot");
    params.executor = executor_;
    ASSERT_EQ(sys_.init(params), hardware_interface::CallbackReturn::SUCCESS);
    for (auto & s : sys_.export_state_interfaces()) {
      const auto name = s.get_name();
      states_.emplace(name, std::move(s));
    }
    for (auto & c : sys_.export_command_interfaces()) {
      const auto name = c.get_name();
      commands_.emplace(name, std::move(c));
    }
  }

  void TearDown() override {if (model_) {mj_deleteModel(model_);}}

  double state(const std::string & name) {return states_.at(name).get_optional().value();}
  void command(const std::string & name, double v) {(void)commands_.at(name).set_value(v);}
  void cycle()
  {
    const rclcpp::Time t(0, 0, RCL_ROS_TIME);
    const rclcpp::Duration dt = rclcpp::Duration::from_seconds(0.005);
    ASSERT_EQ(sys_.read(t, dt), hardware_interface::return_type::OK);
    ASSERT_EQ(sys_.write(t, dt), hardware_interface::return_type::OK);
  }

  std::shared_ptr<rclcpp::Executor> executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  hsim_mujoco_plugins::VirtualRobotSystem sys_;
  mjModel * model_ = nullptr;
  std::vector<MotorSpec> motors_;
  std::map<std::string, hardware_interface::StateInterface> states_;
  std::map<std::string, hardware_interface::CommandInterface> commands_;
};

// The master's automatic transitions take every drive to OPERATION_ENABLED in mode 5; the two-motor mechanisms
// report the motor angles the rod geometry gives for the keyframe's joint angles.
TEST_F(VirtualRobotTest, DrivesEnableAndLinkagesMap)
{
  start(true);
  if (IsSkipped()) {return;}
  for (int i = 0; i < 20; ++i) {cycle();}                    // enable within 20 cycles, as test_mock_slave expects
  for (const auto & mo : motors_) {
    EXPECT_TRUE(wisco_ethercat_ros2_hw::mock::cia402::operation_enabled(
      static_cast<uint16_t>(state(mo.name + "/status_word")))) << mo.name;
    EXPECT_EQ(state(mo.name + "/mode_of_operation"), 5.0) << mo.name;
  }
  mjData * d = mj_makeData(model_);
  mj_resetDataKeyframe(model_, d, mj_name2id(model_, mjOBJ_KEY, "init_state"));
  for (const std::string mech : {"left_ankle", "right_ankle", "waist"}) {
    const auto loaded = wisco_ethercat_ros2_hw::linkage::load_mechanism(kLinkage, mech);
    const wisco_ethercat_ros2_hw::linkage::Mechanism m(loaded.geometry);
    wisco_ethercat_ros2_hw::linkage::Vec2 zero, th, q;
    for (int k = 0; k < 2; ++k) {
      q[k] = d->qpos[model_->jnt_qposadr[mj_name2id(model_, mjOBJ_JOINT, loaded.joints[k].c_str())]];
    }
    ASSERT_TRUE(m.motor_angles({0.0, 0.0}, zero));
    ASSERT_TRUE(m.motor_angles(q, th, zero));
    EXPECT_NEAR(state(mech + "_motor_a/position"), th[0] - zero[0], 1e-4) << mech;
    EXPECT_NEAR(state(mech + "_motor_b/position"), th[1] - zero[1], 1e-4) << mech;
  }
  mj_deleteData(d);
  EXPECT_NEAR(state("imu/orientation.w"), 1.0, 0.05);       // frozen upright at the keyframe
}

// Mode-5 holds sent over the bus (each motor's position at start, the trained gains) keep the joints on target
// and the robot upright: the torque path drive -> (J^T) -> MuJoCo works for single and two-motor joints alike.
// The world is frozen while the drives enable (as on a gantry) and released through ~/set_pause.
TEST_F(VirtualRobotTest, ModeFiveHoldKeepsTheRobotUp)
{
  start(true);
  if (IsSkipped()) {return;}
  for (int i = 0; i < 20; ++i) {cycle();}                   // drives enabled, positions known
  std::map<std::string, double> hold;
  for (const auto & mo : motors_) {hold[mo.name] = state(mo.name + "/position");}
  auto send = [&]() {
      for (const auto & mo : motors_) {
        command(mo.name + "/position", hold[mo.name]);
        command(mo.name + "/kp", mo.kp);
        command(mo.name + "/kd", mo.kd);
      }
    };
  for (int i = 0; i < 3; ++i) {send(); cycle();}           // the targets reach the drives before the release

  using mujoco_ros2_control_msgs::srv::SetPause;
  auto client_node = std::make_shared<rclcpp::Node>("test_client");
  executor_->add_node(client_node);
  auto client = client_node->create_client<SetPause>("/virtual_robot_test/set_pause");
  ASSERT_TRUE(client->wait_for_service(std::chrono::seconds(5)));
  auto req = std::make_shared<SetPause::Request>();
  req->paused = false;
  auto future = client->async_send_request(req);
  ASSERT_EQ(executor_->spin_until_future_complete(future, std::chrono::seconds(5)),
    rclcpp::FutureReturnCode::SUCCESS);
  ASSERT_TRUE(future.get()->success);
  executor_->remove_node(client_node);

  for (int i = 0; i < 60; ++i) {send(); cycle();}          // 0.3 s
  for (const auto & mo : motors_) {
    EXPECT_NEAR(state(mo.name + "/position"), hold[mo.name], 0.1) << mo.name;
  }
  const double x = state("imu/orientation.x"), y = state("imu/orientation.y");
  EXPECT_LT(std::hypot(x, y), 0.1) << "tilted";             // sin(tilt / 2)
}

}  // namespace
