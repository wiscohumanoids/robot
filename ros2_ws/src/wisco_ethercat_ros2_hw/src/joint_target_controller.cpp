// JointTargetController: see include/wisco_ethercat_ros2_hw/joint_target_controller.hpp.
#include "wisco_ethercat_ros2_hw/joint_target_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_map>

#include <yaml-cpp/yaml.h>

#include "pluginlib/class_list_macros.hpp"
#include "urdf/model.h"

namespace wisco_ethercat_ros2_hw
{

using controller_interface::CallbackReturn;
using controller_interface::InterfaceConfiguration;
using controller_interface::interface_configuration_type;
using controller_interface::return_type;

namespace
{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr uint64_t kStatusEvery = 50;   // cycles between status messages
}  // namespace

const char * JointTargetController::phase_name(Phase p)
{
  switch (p) {
    case Phase::kHold: return "hold";
    case Phase::kRamp: return "ramp";
    case Phase::kTrack: return "track";
    case Phase::kSafe: return "safe";
  }
  return "?";
}

int64_t JointTargetController::steady_now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

CallbackReturn JointTargetController::on_init()
{
  try {
    auto_declare<std::vector<std::string>>("joints", {});
    auto_declare<std::vector<std::string>>("chained_joints", {});
    auto_declare<std::string>("chain_prefix", "parallel_joints");
    auto_declare<std::string>("topic", "/locomotion/joint_targets");
    auto_declare<std::string>("manifest_file", "");
    auto_declare<std::vector<double>>("kp", {});
    auto_declare<std::vector<double>>("kd", {});
    auto_declare<std::vector<std::string>>("gain_override_joints", {});
    auto_declare<std::vector<double>>("gain_override_kp", {});
    auto_declare<std::vector<double>>("gain_override_kd", {});
    auto_declare<std::vector<std::string>>("gain_interfaces", {});
    auto_declare<std::vector<double>>("position_min", {});
    auto_declare<std::vector<double>>("position_max", {});
    auto_declare<double>("limit_margin", 0.02);
    auto_declare<double>("timeout", 0.05);
    auto_declare<double>("ramp_time", 2.0);
    auto_declare<std::string>("timeout_action", "damp");
    auto_declare<double>("damping_kd", 8.0);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "parameter declaration failed: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointTargetController::on_configure(const rclcpp_lifecycle::State &)
{
  auto node = get_node();
  const auto log = node->get_logger();
  const auto names = node->get_parameter("joints").as_string_array();
  const auto chained = node->get_parameter("chained_joints").as_string_array();
  chain_prefix_ = node->get_parameter("chain_prefix").as_string();
  gain_ifaces_ = node->get_parameter("gain_interfaces").as_string_array();
  timeout_ns_ = static_cast<int64_t>(node->get_parameter("timeout").as_double() * 1e9);
  ramp_time_ = node->get_parameter("ramp_time").as_double();
  damping_kd_ = node->get_parameter("damping_kd").as_double();
  const auto action = node->get_parameter("timeout_action").as_string();
  const double margin = node->get_parameter("limit_margin").as_double();
  if (names.empty()) {
    RCLCPP_ERROR(log, "no joints");
    return CallbackReturn::ERROR;
  }
  if (action != "damp" && action != "hold") {
    RCLCPP_ERROR(log, "timeout_action '%s': damp or hold", action.c_str());
    return CallbackReturn::ERROR;
  }
  damp_on_timeout_ = action == "damp";
  if (!gain_ifaces_.empty() && gain_ifaces_.size() != 2) {
    RCLCPP_ERROR(log, "gain_interfaces: none, or two names (kp, kd)");
    return CallbackReturn::ERROR;
  }
  if (!(timeout_ns_ > 0) || ramp_time_ < 0.0) {
    RCLCPP_ERROR(log, "timeout must be > 0 and ramp_time >= 0");
    return CallbackReturn::ERROR;
  }

  // gains: the policy manifest (the PD the policy was trained with), else the kp / kd parameters
  std::unordered_map<std::string, std::pair<double, double>> gains;
  const auto manifest = node->get_parameter("manifest_file").as_string();
  if (!manifest.empty()) {
    try {
      for (const auto & j : YAML::LoadFile(manifest)["joints"]) {
        gains[j["name"].as<std::string>()] = {j["kp"].as<double>(), j["kd"].as<double>()};
      }
    } catch (const std::exception & e) {
      RCLCPP_ERROR(log, "manifest %s: %s", manifest.c_str(), e.what());
      return CallbackReturn::ERROR;
    }
  }
  const auto kp = node->get_parameter("kp").as_double_array();
  const auto kd = node->get_parameter("kd").as_double_array();
  if (!kp.empty() || !kd.empty()) {
    if (kp.size() != names.size() || kd.size() != names.size()) {
      RCLCPP_ERROR(log, "kp and kd need one value per joint");
      return CallbackReturn::ERROR;
    }
    for (std::size_t i = 0; i < names.size(); ++i) {gains[names[i]] = {kp[i], kd[i]};}
  }
  // joints the policy does not train (e.g. Wisco's head, welded in training) get their gains here
  const auto oj = node->get_parameter("gain_override_joints").as_string_array();
  const auto okp = node->get_parameter("gain_override_kp").as_double_array();
  const auto okd = node->get_parameter("gain_override_kd").as_double_array();
  if (okp.size() != oj.size() || okd.size() != oj.size()) {
    RCLCPP_ERROR(log, "gain_override_kp / gain_override_kd need one value per gain_override_joints entry");
    return CallbackReturn::ERROR;
  }
  for (std::size_t i = 0; i < oj.size(); ++i) {gains[oj[i]] = {okp[i], okd[i]};}

  // limits: the parameters, else the URDF
  auto lo = node->get_parameter("position_min").as_double_array();
  auto hi = node->get_parameter("position_max").as_double_array();
  urdf::Model model;
  const bool have_urdf = lo.empty() && !get_robot_description().empty() &&
    model.initString(get_robot_description());

  joints_.clear();
  for (std::size_t i = 0; i < names.size(); ++i) {
    Joint j;
    j.name = names[i];
    j.chained = std::find(chained.begin(), chained.end(), j.name) != chained.end();
    const auto g = gains.find(j.name);
    if (g == gains.end() && !j.chained && !gain_ifaces_.empty()) {
      RCLCPP_ERROR(log, "%s: no kp / kd (manifest_file or kp, kd parameters)", j.name.c_str());
      return CallbackReturn::ERROR;
    }
    if (g != gains.end()) {
      j.kp = g->second.first;
      j.kd = g->second.second;
    }
    if (!lo.empty()) {
      if (lo.size() != names.size() || hi.size() != names.size()) {
        RCLCPP_ERROR(log, "position_min / position_max need one value per joint");
        return CallbackReturn::ERROR;
      }
      j.lower = lo[i];
      j.upper = hi[i];
    } else if (have_urdf) {
      const auto uj = model.getJoint(j.name);
      if (!uj || !uj->limits || uj->type == urdf::Joint::CONTINUOUS) {
        RCLCPP_ERROR(log, "%s: no position limits in robot_description", j.name.c_str());
        return CallbackReturn::ERROR;
      }
      j.lower = uj->limits->lower;
      j.upper = uj->limits->upper;
    } else {
      RCLCPP_ERROR(log, "joint limits: set position_min / position_max or provide robot_description");
      return CallbackReturn::ERROR;
    }
    j.lower += margin;
    j.upper -= margin;
    if (!(j.lower < j.upper)) {
      RCLCPP_ERROR(log, "%s: empty range after limit_margin", j.name.c_str());
      return CallbackReturn::ERROR;
    }
    joints_.push_back(j);
  }
  for (const auto & c : chained) {
    if (std::find(names.begin(), names.end(), c) == names.end()) {
      RCLCPP_ERROR(log, "chained joint %s is not in joints", c.c_str());
      return CallbackReturn::ERROR;
    }
  }

  // name -> index, for the (non-real-time) subscription callback
  auto index = std::make_shared<std::unordered_map<std::string, std::size_t>>();
  for (std::size_t i = 0; i < joints_.size(); ++i) {(*index)[joints_[i].name] = i;}
  const std::size_t n = joints_.size();
  buffer_.writeFromNonRT(nullptr);
  subscription_ = node->create_subscription<sensor_msgs::msg::JointState>(
    node->get_parameter("topic").as_string(), rclcpp::SystemDefaultsQoS(),
    [this, index, n](const sensor_msgs::msg::JointState::SharedPtr msg) {
      auto t = std::make_shared<Targets>();
      t->position.assign(n, kNaN);
      const std::size_t m = std::min(msg->name.size(), msg->position.size());
      for (std::size_t k = 0; k < m; ++k) {
        const auto it = index->find(msg->name[k]);
        if (it != index->end() && std::isfinite(msg->position[k])) {t->position[it->second] = msg->position[k];}
      }
      t->received_ns = steady_now_ns();
      buffer_.writeFromNonRT(t);
    });
  hold_service_ = node->create_service<std_srvs::srv::Trigger>(
    "~/hold", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
      hold_request_ns_.store(steady_now_ns());
      res->success = true;
      res->message = "holding the measured pose; ramping in again when new targets arrive";
    });
  status_ = std::make_shared<realtime_tools::RealtimePublisher<diagnostic_msgs::msg::DiagnosticArray>>(
    node->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("~/status", 10));
  RCLCPP_INFO(log, "%zu joints (%zu through %s), timeout %.0f ms, ramp %.1f s, on timeout: %s",
    joints_.size(), chained.size(), chain_prefix_.c_str(), timeout_ns_ / 1e6, ramp_time_, action.c_str());
  return CallbackReturn::SUCCESS;
}

std::string JointTargetController::position_name(const Joint & j) const
{
  return (j.chained ? chain_prefix_ + "/" : std::string()) + j.name + "/position";
}

InterfaceConfiguration JointTargetController::command_interface_configuration() const
{
  InterfaceConfiguration c{interface_configuration_type::INDIVIDUAL, {}};
  for (const auto & j : joints_) {
    c.names.push_back(position_name(j));
    if (!j.chained) {
      for (const auto & g : gain_ifaces_) {c.names.push_back(j.name + "/" + g);}
    }
  }
  return c;
}

InterfaceConfiguration JointTargetController::state_interface_configuration() const
{
  InterfaceConfiguration c{interface_configuration_type::INDIVIDUAL, {}};
  for (const auto & j : joints_) {c.names.push_back(position_name(j));}
  return c;
}

CallbackReturn JointTargetController::on_activate(const rclcpp_lifecycle::State &)
{
  // the loaned interfaces arrive in configuration order: map each joint to its slots
  std::size_t k = 0;
  for (std::size_t i = 0; i < joints_.size(); ++i) {
    Joint & j = joints_[i];
    j.cmd_pos = k++;
    if (!j.chained && !gain_ifaces_.empty()) {
      j.cmd_kp = k++;
      j.cmd_kd = k++;
    }
    j.state_pos = i;
    if (!std::isfinite(state_interfaces_[i].get_value())) {
      RCLCPP_ERROR(get_node()->get_logger(), "%s: no position reading", j.name.c_str());
      return CallbackReturn::ERROR;
    }
  }
  buffer_.writeFromNonRT(nullptr);
  phase_ = Phase::kHold;
  last_seen_ns_ = 0;
  hold_floor_ns_ = 0;
  since_fresh_ = std::numeric_limits<double>::infinity();
  hold_request_ns_.store(0);
  clamped_ = cycles_ = 0;
  hold_measured();
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointTargetController::on_deactivate(const rclcpp_lifecycle::State &)
{
  hold_measured();
  return CallbackReturn::SUCCESS;
}

void JointTargetController::write(std::size_t i, double position, double kp, double kd)
{
  const Joint & j = joints_[i];
  (void)command_interfaces_[j.cmd_pos].set_value(position);
  if (j.cmd_kp != SIZE_MAX) {
    (void)command_interfaces_[j.cmd_kp].set_value(kp);
    (void)command_interfaces_[j.cmd_kd].set_value(kd);
  }
}

void JointTargetController::hold_measured()
{
  for (std::size_t i = 0; i < joints_.size(); ++i) {
    Joint & j = joints_[i];
    j.target = std::clamp(state_interfaces_[j.state_pos].get_value(), j.lower, j.upper);
    write(i, j.target, j.kp, j.kd);
  }
}

return_type JointTargetController::update(const rclcpp::Time &, const rclcpp::Duration & period)
{
  const int64_t hold_ns = hold_request_ns_.exchange(0);
  if (hold_ns != 0) {                                     // ~/hold: re-arm at the measured pose
    hold_measured();
    phase_ = Phase::kHold;
    last_seen_ns_ = std::max(last_seen_ns_, hold_ns);     // only messages after the request count
    hold_floor_ns_ = hold_ns;
    since_fresh_ = std::numeric_limits<double>::infinity();
  }
  const auto latest = buffer_.readFromRT();
  const std::shared_ptr<Targets> msg = latest ? *latest : nullptr;
  const bool fresh = msg && msg->received_ns > last_seen_ns_;
  // the timeout counts the control loop's own time (its update periods): real time on the robot, simulated time in
  // simulation, where a busy computer slows the world and the policy together
  since_fresh_ = fresh ? 0.0 : since_fresh_ + period.seconds();
  const bool alive = msg && msg->received_ns > hold_floor_ns_ && since_fresh_ * 1e9 <= static_cast<double>(timeout_ns_);

  if (fresh) {
    last_seen_ns_ = msg->received_ns;
    for (std::size_t i = 0; i < joints_.size(); ++i) {
      const double p = msg->position[i];
      if (!std::isfinite(p)) {continue;}                   // not in this message: keep the last target
      const double c = std::clamp(p, joints_[i].lower, joints_[i].upper);
      if (c != p) {++clamped_;}
      joints_[i].target = c;
    }
  }

  // phase transitions
  if (alive && (phase_ == Phase::kHold || phase_ == Phase::kSafe)) {
    phase_ = ramp_time_ > 0.0 ? Phase::kRamp : Phase::kTrack;
    ramp_elapsed_ = 0.0;
    for (auto & j : joints_) {j.ramp_from = state_interfaces_[j.state_pos].get_value();}
  } else if (!alive && (phase_ == Phase::kRamp || phase_ == Phase::kTrack)) {
    phase_ = Phase::kSafe;
    RCLCPP_WARN(get_node()->get_logger(), "no joint targets for %.0f ms: %s",
      timeout_ns_ / 1e6, damp_on_timeout_ ? "damping" : "holding");
    for (auto & j : joints_) {j.target = std::clamp(state_interfaces_[j.state_pos].get_value(), j.lower, j.upper);}
  }

  switch (phase_) {
    case Phase::kHold:
      for (std::size_t i = 0; i < joints_.size(); ++i) {write(i, joints_[i].target, joints_[i].kp, joints_[i].kd);}
      break;
    case Phase::kRamp: {
      ramp_elapsed_ += period.seconds();
      const double a = std::min(1.0, ramp_elapsed_ / ramp_time_);
      for (std::size_t i = 0; i < joints_.size(); ++i) {
        const Joint & j = joints_[i];
        write(i, j.ramp_from + a * (j.target - j.ramp_from), j.kp, j.kd);
      }
      if (a >= 1.0) {phase_ = Phase::kTrack;}
      break;
    }
    case Phase::kTrack:
      for (std::size_t i = 0; i < joints_.size(); ++i) {write(i, joints_[i].target, joints_[i].kp, joints_[i].kd);}
      break;
    case Phase::kSafe:
      for (std::size_t i = 0; i < joints_.size(); ++i) {
        const Joint & j = joints_[i];
        if (damp_on_timeout_ && !j.chained && j.cmd_kp != SIZE_MAX) {
          write(i, state_interfaces_[j.state_pos].get_value(), 0.0, damping_kd_);   // limp, damped
        } else {
          write(i, j.target, j.kp, j.kd);                                           // hold where it stopped
        }
      }
      break;
  }

  if (++cycles_ % kStatusEvery == 0) {publish_status();}
  return return_type::OK;
}

void JointTargetController::publish_status()
{
  if (!status_ || !status_->trylock()) {return;}
  auto & arr = status_->msg_;
  arr.status.resize(1);
  auto & s = arr.status[0];
  s.name = std::string(get_node()->get_name()) + ": joint targets";
  s.level = phase_ == Phase::kSafe ? diagnostic_msgs::msg::DiagnosticStatus::WARN :
    diagnostic_msgs::msg::DiagnosticStatus::OK;
  s.message = phase_name(phase_);
  s.values.resize(2);
  s.values[0].key = "clamped_targets";
  s.values[0].value = std::to_string(clamped_);
  s.values[1].key = "joints";
  s.values[1].value = std::to_string(joints_.size());
  arr.header.stamp = get_node()->now();
  status_->unlockAndPublish();
}

}  // namespace wisco_ethercat_ros2_hw

PLUGINLIB_EXPORT_CLASS(wisco_ethercat_ros2_hw::JointTargetController, controller_interface::ControllerInterface)
