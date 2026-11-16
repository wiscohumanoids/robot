#include "wisco_ethercat_ros2_hw/arm_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <limits>
#include <map>

#include "controller_interface/helpers.hpp"
#include "wisco_ethercat_ros2_hw/gravity_model.hpp"

namespace wisco_ethercat_ros2_hw
{

ArmController::ArmController() = default;
ArmController::~ArmController() = default;

int64_t ArmController::steady_now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

controller_interface::CallbackReturn ArmController::on_init()
{
  try {
    auto_declare<std::vector<std::string>>("joints", std::vector<std::string>{});
    auto_declare<std::vector<double>>("max_torque", std::vector<double>{});
    auto_declare<std::vector<double>>("max_velocity", std::vector<double>{});
    auto_declare<std::vector<double>>("soft_lower", std::vector<double>{});
    auto_declare<std::vector<double>>("soft_upper", std::vector<double>{});
    auto_declare<std::vector<double>>("kp", std::vector<double>{});
    auto_declare<std::vector<double>>("kd", std::vector<double>{});
    auto_declare<std::vector<double>>("kd_fallback", std::vector<double>{});
    auto_declare<double>("watchdog_s", 0.05);
    auto_declare<double>("soft_margin", 0.05);
    auto_declare<double>("blend_s", 0.02);
    auto_declare<bool>("drive_feedback", false);
    auto_declare<bool>("hold_on_startup", false);
    auto_declare<bool>("hold_on_timeout", false);
    auto_declare<bool>("gravity_enabled", true);
    auto_declare<bool>("hold_only", false);
    auto_declare<bool>("position_only", false);
    for (const char * name : {"local_center", "local_bias", "local_jacobian", "local_lower", "local_upper"}) {
      auto_declare<std::vector<double>>(name, std::vector<double>{});
    }
    auto_declare<bool>("write_max_torque", false);
    auto_declare<std::vector<bool>>("torque_offset_enabled", std::vector<bool>{});
    auto_declare<std::string>("calibration_file", "");
    auto_declare<std::string>("robot_description", "");
    auto_declare<std::vector<std::string>>("link_mass_names", std::vector<std::string>{});
    auto_declare<std::vector<double>>("link_mass_values", std::vector<double>{});
    auto_declare<std::string>("payload_frame", "tool0");
  } catch (const std::exception & e) {
    std::fprintf(stderr, "ArmController: exception in on_init: %s\n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ArmController::on_configure(const rclcpp_lifecycle::State &)
{
  auto node = get_node();
  auto logger = node->get_logger();
  joints_ = node->get_parameter("joints").as_string_array();
  const size_t n = joints_.size();
  if (n == 0) {
    RCLCPP_ERROR(logger, "'joints' is empty");
    return controller_interface::CallbackReturn::ERROR;
  }
  auto vec = [&](const char * name, std::vector<double> & out) {
      out = node->get_parameter(name).as_double_array();
      if (out.size() != n) {
        RCLCPP_ERROR(logger, "'%s' needs %zu entries, got %zu", name, n, out.size());
        return false;
      }
      return true;
    };
  std::vector<double> kp, kd;
  if (!vec("max_torque", cfg_.max_torque) || !vec("max_velocity", cfg_.max_velocity) ||
    !vec("soft_lower", cfg_.lower) || !vec("soft_upper", cfg_.upper) ||
    !vec("kd_fallback", cfg_.kd_fallback))
  {
    return controller_interface::CallbackReturn::ERROR;
  }
  for (size_t i = 0; i < n; ++i) {
    if (!(cfg_.max_torque[i] > 0) || !(cfg_.max_velocity[i] > 0) || !(cfg_.lower[i] < cfg_.upper[i])) {
      RCLCPP_ERROR(logger, "bad limits for joint '%s'", joints_[i].c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
  }
  cfg_.coulomb.assign(n, 0.0);
  cfg_.viscous.assign(n, 0.0);
  cfg_.watchdog_s = node->get_parameter("watchdog_s").as_double();
  cfg_.soft_margin = node->get_parameter("soft_margin").as_double();
  blend_s_ = node->get_parameter("blend_s").as_double();
  drive_feedback_ = node->get_parameter("drive_feedback").as_bool();
  hold_on_startup_ = node->get_parameter("hold_on_startup").as_bool();
  hold_on_timeout_ = node->get_parameter("hold_on_timeout").as_bool();
  hold_only_ = node->get_parameter("hold_only").as_bool();
  position_only_ = node->get_parameter("position_only").as_bool();
  hold_on_startup_ = hold_on_startup_ || hold_only_ || position_only_;
  hold_on_timeout_ = hold_on_timeout_ || hold_only_ || position_only_;
  gravity_enabled_ = node->get_parameter("gravity_enabled").as_bool() && !hold_only_ && !position_only_;
  if (hold_only_) {
    RCLCPP_WARN(logger, "Relative reference: position holding only; motion commands ignored, "
      "gravity disabled. Joint angles are not registered to the CAD model.");
  }
  local_center_ = node->get_parameter("local_center").as_double_array();
  local_bias_ = node->get_parameter("local_bias").as_double_array();
  local_jacobian_ = node->get_parameter("local_jacobian").as_double_array();
  local_lower_ = node->get_parameter("local_lower").as_double_array();
  local_upper_ = node->get_parameter("local_upper").as_double_array();
  local_gravity_enabled_ = !local_bias_.empty();
  if (local_gravity_enabled_) {
    const auto finite = [](const std::vector<double> & v) {
        return std::all_of(v.begin(), v.end(), [](double x) {return std::isfinite(x);});
      };
    if (!position_only_ || hold_only_ || local_center_.size() != n || local_bias_.size() != n ||
      local_jacobian_.size() != n*n || local_lower_.size() != n || local_upper_.size() != n ||
      !finite(local_center_) || !finite(local_bias_) || !finite(local_jacobian_) ||
      !finite(local_lower_) || !finite(local_upper_))
    {
      RCLCPP_ERROR(logger, "Invalid local gravity parameters");
      return controller_interface::CallbackReturn::ERROR;
    }
    for (size_t i = 0; i < n; ++i) {
      if (!(local_lower_[i] < local_center_[i] && local_center_[i] < local_upper_[i]) ||
        local_lower_[i] < cfg_.lower[i] || local_upper_[i] > cfg_.upper[i])
      {
        RCLCPP_ERROR(logger, "Local gravity domain must be inside the position bounds");
        return controller_interface::CallbackReturn::ERROR;
      }
    }
  }
  write_max_torque_ = node->get_parameter("write_max_torque").as_bool();
  torque_offset_enabled_ = node->get_parameter("torque_offset_enabled").as_bool_array();
  if (torque_offset_enabled_.empty()) {torque_offset_enabled_.assign(n, true);}
  if (torque_offset_enabled_.size() != n) {
    RCLCPP_ERROR(logger, "'torque_offset_enabled' needs %zu entries, got %zu", n,
      torque_offset_enabled_.size());
    return controller_interface::CallbackReturn::ERROR;
  }
  for (size_t i = 0; i < n; ++i) {
    if (!torque_offset_enabled_[i]) {
      RCLCPP_WARN(logger, "%s: legacy PDOs; CSP uses the drive position loop without torque "
        "feedforward, CST uses 0x6071", joints_[i].c_str());
    }
  }

  std::string urdf = node->get_parameter("robot_description").as_string();
  if (urdf.empty()) {
    RCLCPP_ERROR(logger, "'robot_description' parameter is empty (the launch file sets it)");
    return controller_interface::CallbackReturn::ERROR;
  }
  gravity_ = std::make_unique<GravityModel>();
  std::string error;
  if (!gravity_->build(
      urdf, joints_, node->get_parameter("calibration_file").as_string(),
      node->get_parameter("link_mass_names").as_string_array(),
      node->get_parameter("link_mass_values").as_double_array(), cfg_, error))
  {
    RCLCPP_ERROR(logger, "%s", error.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  const std::string payload_frame = node->get_parameter("payload_frame").as_string();
  payload_ok_ = gravity_->set_payload_frame(payload_frame);
  if (!payload_ok_) {
    RCLCPP_WARN(logger, "payload_frame '%s' is not in robot_description: payload blocks are ignored",
      payload_frame.c_str());
  }

  q_.assign(n, 0.0);
  q_hold_.assign(n, 0.0);
  qd_.assign(n, 0.0);
  tau_meas_.assign(n, 0.0);
  g_.assign(n, 0.0);
  tau_switch_.assign(n, 0.0);
  q_target_prev_.assign(n, 0.0);
  q_switch_.assign(n, 0.0);
  since_ease_.assign(n, 0.0);
  enabled_.assign(n, true);
  out_.q_target.assign(n, 0.0);
  out_.tau.assign(n, 0.0);

  const size_t expected = kCommandHeader + kCommandBlocks * n;
  subscription_ = node->create_subscription<std_msgs::msg::Float64MultiArray>(
    "~/command", rclcpp::SystemDefaultsQoS(),
    [this, n, expected](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
      if (hold_only_) {
        RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
          "Ignoring motion command: relative reference permits position holding only");
        return;
      }
      if (msg->data.size() != expected && msg->data.size() != expected + kPayloadBlock) {
        RCLCPP_WARN_THROTTLE(
          get_node()->get_logger(), *get_node()->get_clock(), 2000,
          "ignoring command with %zu values; expected %zu (or %zu with a payload)", msg->data.size(),
          expected, expected + kPayloadBlock);
        return;
      }
      auto c = std::make_shared<Command>();
      const auto & d = msg->data;
      c->law.present = true;
      c->law.mode = d[0] > 0.5 ? arm::kModeT : arm::kModeP;
      const int flags = std::isfinite(d[1]) ? static_cast<int>(d[1]) : 1;
      c->law.gravity_ff = flags & 1;
      c->law.ignore_soft_limits = flags & 2;
      if (position_only_ && c->law.mode != arm::kModeP) {
        RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
          "Relative calibration only permits bounded position commands");
        return;
      }
      if (position_only_) {c->law.ignore_soft_limits = false;}
      c->law.friction_ff_gain = std::isfinite(d[2]) ? d[2] : 1.0;
      auto block = [&](size_t b) {
          return std::vector<double>(d.begin() + kCommandHeader + b * n,
            d.begin() + kCommandHeader + (b + 1) * n);
        };
      c->law.q = block(0);
      c->law.qd = block(1);
      c->law.kp = block(2);
      c->law.kd = block(3);
      c->law.tau_ff = block(4);
      if (position_only_) {std::fill(c->law.tau_ff.begin(), c->law.tau_ff.end(), 0.0);}
      c->law.torque_clamp = block(5);
      if (d.size() == expected + kPayloadBlock && std::isfinite(d[expected])) {
        c->has_payload = true;
        for (size_t k = 0; k < kPayloadBlock; ++k) {
          c->payload[k] = std::isfinite(d[expected + k]) ? d[expected + k] : 0.0;
        }
      }
      c->received_ns = steady_now_ns();
      command_buffer_.writeFromNonRT(c);
    });
  state_pub_ = node->create_publisher<std_msgs::msg::Float64MultiArray>(
    "~/state", rclcpp::SystemDefaultsQoS());
  rt_state_ = std::make_unique<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(
    state_pub_);
  rt_state_->msg_.data.assign(kStateHeader + (kStateBlocks + 2) * n, 0.0);

  RCLCPP_INFO(logger, "%zu joints, watchdog %.0f ms; Mode P (CSP) / Mode T (CST)%s", n,
    cfg_.watchdog_s * 1e3, drive_feedback_ ? "; drive feedback (0x6041, 0x6061)" : "");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration ArmController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const char * iface : {"position", "effort", "torque_offset", "mode_of_operation"}) {
    for (size_t i = 0; i < joints_.size(); ++i) {
      if (std::string(iface) != "torque_offset" || torque_offset_enabled_[i]) {
        config.names.push_back(joints_[i] + "/" + iface);
      }
    }
  }
  if (write_max_torque_) {
    for (const auto & j : joints_) {config.names.push_back(j + "/max_torque");}
  }
  return config;
}

controller_interface::InterfaceConfiguration ArmController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const char * iface : {"position", "velocity", "effort"}) {
    for (const auto & j : joints_) {config.names.push_back(j + "/" + iface);}
  }
  if (drive_feedback_) {
    for (const char * iface : {"status_word", "mode_of_operation"}) {
      for (const auto & j : joints_) {config.names.push_back(j + "/" + iface);}
    }
  }
  return config;
}

controller_interface::CallbackReturn ArmController::on_activate(const rclcpp_lifecycle::State &)
{
  cmd_pos_.clear(); cmd_eff_.clear(); cmd_off_.clear(); cmd_mode_.clear(); cmd_max_.clear();
  st_pos_.clear(); st_vel_.clear(); st_eff_.clear(); st_status_.clear(); st_mode_.clear();
  bool ok =
    controller_interface::get_ordered_interfaces(command_interfaces_, joints_, "position", cmd_pos_) &&
    controller_interface::get_ordered_interfaces(command_interfaces_, joints_, "effort", cmd_eff_) &&
    controller_interface::get_ordered_interfaces(command_interfaces_, joints_, "mode_of_operation", cmd_mode_) &&
    controller_interface::get_ordered_interfaces(state_interfaces_, joints_, "position", st_pos_) &&
    controller_interface::get_ordered_interfaces(state_interfaces_, joints_, "velocity", st_vel_) &&
    controller_interface::get_ordered_interfaces(state_interfaces_, joints_, "effort", st_eff_);
  cmd_off_.assign(joints_.size(), nullptr);
  for (size_t i = 0; i < joints_.size(); ++i) {
    if (!torque_offset_enabled_[i]) {continue;}
    const auto name = joints_[i] + "/torque_offset";
    const auto iface = std::find_if(command_interfaces_.begin(), command_interfaces_.end(),
      [&](const auto & candidate) {return candidate.get_name() == name;});
    if (iface == command_interfaces_.end()) {
      ok = false;
      break;
    }
    cmd_off_[i] = &*iface;
  }
  if (ok && write_max_torque_) {
    ok = controller_interface::get_ordered_interfaces(command_interfaces_, joints_, "max_torque", cmd_max_);
  }
  if (ok && drive_feedback_) {
    ok = controller_interface::get_ordered_interfaces(
      state_interfaces_, joints_, "status_word", st_status_) &&
      controller_interface::get_ordered_interfaces(
      state_interfaces_, joints_, "mode_of_operation", st_mode_);
  }
  if (!ok) {
    RCLCPP_ERROR(get_node()->get_logger(), "could not get the expected interfaces");
    return controller_interface::CallbackReturn::ERROR;
  }
  // Start held in CSP at the actual position; no command -> watchdog fallback
  // next cycle (gravity comp + damping) until Python sends targets.
  command_buffer_.writeFromNonRT(std::shared_ptr<Command>());
  for (size_t i = 0; i < joints_.size(); ++i) {
    q_target_prev_[i] = st_pos_[i].get().get_value();
    cmd_pos_[i].get().set_value(q_target_prev_[i]);
    cmd_eff_[i].get().set_value(0.0);
    if (cmd_off_[i]) {cmd_off_[i]->set_value(0.0);}
    cmd_mode_[i].get().set_value(static_cast<double>(arm::kCsp));
    if (write_max_torque_) {cmd_max_[i].get().set_value(cfg_.max_torque[i]);}
  }
  last_mode_ = arm::kModeP;
  holding_ = false;
  since_switch_ = 0.0;
  for (size_t i = 0; i < joints_.size(); ++i) {
    q_switch_[i] = q_target_prev_[i];
    since_ease_[i] = 0.0;
  }
  std::fill(enabled_.begin(), enabled_.end(), true);
  std::fill(std::begin(payload_), std::end(payload_), 0.0);
  gravity_->set_payload(0.0, 0.0, 0.0, 0.0);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ArmController::on_deactivate(const rclcpp_lifecycle::State &)
{
  // Hold position in CSP with no feedforward.
  for (size_t i = 0; i < cmd_pos_.size(); ++i) {
    cmd_pos_[i].get().set_value(st_pos_[i].get().get_value());
    cmd_eff_[i].get().set_value(0.0);
    if (cmd_off_[i]) {cmd_off_[i]->set_value(0.0);}
    cmd_mode_[i].get().set_value(static_cast<double>(arm::kCsp));
  }
  command_buffer_.writeFromNonRT(std::shared_ptr<Command>());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type ArmController::update(
  const rclcpp::Time &, const rclcpp::Duration & period)
{
  const size_t n = joints_.size();
  const int64_t now = steady_now_ns();   // command age (the watchdog) only
  // Blends and eases count control periods, not wall time: the drives step
  // once per bus cycle, so a late cycle must not skip part of a ramp.
  const double dt = std::max(period.seconds(), 1e-4);
  since_switch_ += dt;
  for (size_t i = 0; i < n; ++i) {
    since_ease_[i] += dt;
    q_[i] = st_pos_[i].get().get_value();
    qd_[i] = st_vel_[i].get().get_value();
    tau_meas_[i] = st_eff_[i].get().get_value();
  }

  // A drive that is not OPERATION_ENABLED follows nothing: hold the whole arm
  // in the fallback until every drive is back.
  bool all_enabled = true;
  if (drive_feedback_) {
    for (size_t i = 0; i < n; ++i) {
      const double sw = st_status_[i].get().get_value();
      enabled_[i] = std::isfinite(sw) && (static_cast<uint16_t>(sw) & 0x6F) == 0x27;
      all_enabled = all_enabled && enabled_[i];
    }
  }

  static const arm::LawCommand kNone;
  const std::shared_ptr<Command> cmd = *command_buffer_.readFromRT();
  if (cmd) {
    cmd->law.age_s = (now - cmd->received_ns) * 1e-9;
    // the payload is sticky: kept after the command goes stale (the watchdog
    // fallback still carries it) and by later commands without the block
    if (cmd->has_payload && payload_ok_ &&
      !std::equal(std::begin(payload_), std::end(payload_), std::begin(cmd->payload)))
    {
      std::copy(std::begin(cmd->payload), std::end(cmd->payload), std::begin(payload_));
      gravity_->set_payload(payload_[0], payload_[1], payload_[2], payload_[3]);
    }
  }
  if (local_gravity_enabled_) {
    bool inside = true;
    for (size_t i = 0; i < n; ++i) {
      inside = inside && std::isfinite(q_[i]) && q_[i] >= local_lower_[i] && q_[i] <= local_upper_[i];
    }
    for (size_t i = 0; i < n; ++i) {
      g_[i] = inside ? local_bias_[i] : 0.0;
      if (inside) {
        for (size_t j = 0; j < n; ++j) {g_[i] += local_jacobian_[i*n+j]*(q_[j]-local_center_[j]);}
      }
    }
  } else if (gravity_enabled_) {gravity_->gravity(q_, g_);} else {std::fill(g_.begin(), g_.end(), 0.0);}
  arm::compute(cfg_, cmd && all_enabled ? cmd->law : kNone, q_, qd_, g_, out_);
  const bool hold = out_.fallback && (hold_on_timeout_ || (hold_on_startup_ && !cmd));
  if (hold) {
    if (!holding_ || ((hold_only_ || position_only_) && !all_enabled)) {q_hold_ = q_;}
    out_.mode = arm::kModeP;
    out_.q_target = q_hold_;
    std::fill(out_.tau.begin(), out_.tau.end(), 0.0);
  }
  holding_ = hold;

  if (out_.mode != last_mode_) {
    since_switch_ = 0.0;
    for (size_t i = 0; i < n; ++i) {
      tau_switch_[i] = tau_meas_[i];
      q_target_prev_[i] = q_[i];   // CSP re-entry starts at the actual position
      q_switch_[i] = q_[i];
      since_ease_[i] = 0.0;
    }
    last_mode_ = out_.mode;
  }
  for (size_t i = 0; i < n; ++i) {
    const bool mode_p = out_.mode == arm::kModeP;
    const int requested = mode_p ? arm::kCsp : arm::kCst;
    bool pending = false;   // drive has not confirmed the requested mode yet
    if (drive_feedback_) {
      const double shown = st_mode_[i].get().get_value();
      pending = (shown == arm::kCsp || shown == arm::kCst) && shown != requested;
    }
    if (!enabled_[i] || (mode_p && pending)) {
      // Not following a CSP target yet (disabled, or the master still sends
      // the actual position as 0x607A): start, and keep restarting, the
      // ease-in from where the joint is, so it never resumes at a stale target.
      q_target_prev_[i] = q_[i];
      q_switch_[i] = q_[i];
      since_ease_[i] = 0.0;
    }
    if (!enabled_[i]) {
      // Re-enabling goes straight to the law's torque: the measured torque is
      // 0 while disabled, and blending up from 0 would let the arm drop.
      tau_switch_[i] = out_.tau[i];
    }
    double target = q_[i];   // Mode T: keeps CSP re-entry bumpless
    double tau = out_.tau[i];
    if (mode_p) {
      const double goal =
        arm::ease_in(q_switch_[i], out_.q_target[i], since_ease_[i], blend_s_);
      target = arm::rate_limit(q_target_prev_[i], goal, cfg_.max_velocity[i] * dt);
      q_target_prev_[i] = target;
    } else {
      tau = arm::blend(tau_switch_[i], out_.tau[i], since_switch_, blend_s_);
      out_.tau[i] = tau;
    }
    // Torque channel: 0x60B2 offset in CSP, 0x6071 target in CST. While a
    // switch is pending, 0x60B2, which CiA402 adds in both modes, so the
    // torque is applied whichever mode the drive runs this cycle.
    cmd_mode_[i].get().set_value(static_cast<double>(requested));
    cmd_pos_[i].get().set_value(target);
    double cap = cfg_.max_torque[i];
    if (cmd && !out_.fallback && i < cmd->law.torque_clamp.size() &&
      std::isfinite(cmd->law.torque_clamp[i]))
    {
      cap = std::min(cap, std::fabs(cmd->law.torque_clamp[i]));
    }
    if (write_max_torque_) {
      // 0x6072 also caps the drive's own CSP position loop, e.g. during a
      // limit search pushing into a hard stop.
      // 0x6072 = 0 is NOT zero torque on the RMD-X: the drive falls back to its
      // stall-current limit. Never let a zero (or tiny, rounding to 0 per mille)
      // clamp reach the drive: floor it at 1% of the configured max.
      cmd_max_[i].get().set_value(std::max(cap, kMinTorqueCapFraction * cfg_.max_torque[i]));
    }
    if (!cmd_off_[i]) {
      // 0x6071 is ignored in CSP. Keep the measured switching torque there
      // during a delayed CST -> CSP transition, then the drive's position
      // loop takes over. No torque-offset object is needed on legacy drives.
      const double effort = mode_p ? std::clamp(tau_switch_[i], -cap, cap) : tau;
      cmd_eff_[i].get().set_value(effort);
      out_.tau[i] = mode_p ? 0.0 : tau;
    } else if (requested == arm::kCsp || pending) {
      cmd_off_[i]->set_value(tau);
      cmd_eff_[i].get().set_value(0.0);
    } else {
      cmd_eff_[i].get().set_value(tau);
      cmd_off_[i]->set_value(0.0);
    }
  }

  if (rt_state_ && rt_state_->trylock()) {
    auto & d = rt_state_->msg_.data;
    d[0] = now * 1e-9;
    d[1] = out_.mode;
    d[2] = out_.fallback;
    d[3] = out_.limit_hit;
    d[4] = out_.clamped;
    for (size_t i = 0; i < n; ++i) {
      d[kStateHeader + i] = q_[i];
      d[kStateHeader + n + i] = qd_[i];
      d[kStateHeader + 2 * n + i] = tau_meas_[i];
      d[kStateHeader + 3 * n + i] = out_.tau[i];
      d[kStateHeader + 4 * n + i] = out_.mode == arm::kModeP ? q_target_prev_[i] : q_[i];
      d[kStateHeader + 5 * n + i] = drive_feedback_ ? st_status_[i].get().get_value() :
        std::numeric_limits<double>::quiet_NaN();
      d[kStateHeader + 6 * n + i] = drive_feedback_ ? st_mode_[i].get().get_value() :
        std::numeric_limits<double>::quiet_NaN();
    }
    rt_state_->unlockAndPublish();
  }
  return controller_interface::return_type::OK;
}

}  // namespace wisco_ethercat_ros2_hw

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(wisco_ethercat_ros2_hw::ArmController, controller_interface::ControllerInterface)
