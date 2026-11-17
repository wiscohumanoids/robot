// VirtualRobotSystem: see include/hsim_mujoco_plugins/virtual_robot_system.hpp.
#include "hsim_mujoco_plugins/virtual_robot_system.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <thread>

#include "pluginlib/class_list_macros.hpp"
#include "wisco_ethercat_ros2_hw/parallel_linkage_yaml.hpp"

namespace hsim_mujoco_plugins
{

namespace cia402 = wisco_ethercat_ros2_hw::mock::cia402;
namespace linkage = wisco_ethercat_ros2_hw::linkage;
using hardware_interface::return_type;

namespace
{
std::string get(const std::unordered_map<std::string, std::string> & p, const std::string & key,
  const std::string & fallback)
{
  const auto it = p.find(key);
  return it == p.end() ? fallback : it->second;
}

double num(const std::unordered_map<std::string, std::string> & p, const std::string & key, double fallback)
{
  const auto it = p.find(key);
  return it == p.end() ? fallback : std::stod(it->second);
}

int id_of(const mjModel * m, mjtObj type, const std::string & name, const std::string & what)
{
  const int id = mj_name2id(m, type, name.c_str());
  if (id < 0) {throw std::runtime_error(what + " '" + name + "' is not in the model");}
  return id;
}
}  // namespace

VirtualRobotSystem::~VirtualRobotSystem()
{
  {
    std::lock_guard<std::mutex> lk(pause_mutex_);
    paused_ = false;
  }
  pause_cv_.notify_all();
  if (auto ex = executor_.lock(); ex && node_) {ex->remove_node(node_);}
  ground_truth_.reset();
  if (data_) {mj_deleteData(data_);}
  if (model_) {mj_deleteModel(model_);}
}

VirtualRobotSystem::CallbackReturn VirtualRobotSystem::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (SystemInterface::on_init(params) != CallbackReturn::SUCCESS) {return CallbackReturn::ERROR;}
  const auto & hw = info_.hardware_parameters;
  auto log = get_logger();
  try {
    // ---- the world ----
    const std::string path = get(hw, "mujoco_model", "");
    char err[1000] = "";
    model_ = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
    if (!model_) {throw std::runtime_error("mujoco_model " + path + ": " + err);}
    const double dt = num(hw, "physics_timestep", 0.0);
    if (dt > 0.0) {model_->opt.timestep = dt;}
    control_period_ = num(hw, "control_period", 0.005);
    substeps_ = static_cast<int>(std::lround(control_period_ / model_->opt.timestep));
    if (substeps_ < 1 || std::fabs(substeps_ * model_->opt.timestep - control_period_) > 1e-9) {
      throw std::runtime_error("control_period " + std::to_string(control_period_) +
              " s is not a whole number of physics steps (" + std::to_string(model_->opt.timestep) + " s)");
    }
    real_time_factor_ = num(hw, "real_time_factor", 1.0);
    settle_cycles_ = static_cast<int>(num(hw, "settle_cycles", 3.0));
    released_ = get(hw, "hold_until_released", "true") != "true";
    data_ = mj_makeData(model_);
    const std::string key = get(hw, "keyframe", "init_state");
    keyframe_ = mj_name2id(model_, mjOBJ_KEY, key.c_str());
    if (keyframe_ >= 0) {mj_resetDataKeyframe(model_, data_, keyframe_);} else {mj_resetData(model_, data_);}
    mj_forward(model_, data_);

    // ---- motors: one drive each, on a hinge or on a crank of a two-motor mechanism ----
    const auto mode = static_cast<int8_t>(num(hw, "mode_of_operation", cia402::kModeMit));
    const std::string linkage_file = get(hw, "linkage_file", "");
    std::map<std::string, int> linkage_index;
    std::set<int> driven_joints;
    motors_.resize(info_.joints.size());
    for (std::size_t i = 0; i < info_.joints.size(); ++i) {
      const auto & ji = info_.joints[i];
      const auto & p = ji.parameters;
      Motor & m = motors_[i];
      wisco_ethercat_ros2_hw::mock::RmdxParams sp;
      sp.dt = control_period_;
      sp.external_physics = true;
      sp.counts_per_rev = num(p, "counts_per_rev", 131072.0);
      const double nm_per_permille = num(p, "effort_nm_per_permille", 0.01);
      sp.rated_torque_nm = 1000.0 * nm_per_permille;
      sp.peak_torque_nm = num(p, "peak_effort_nm", sp.rated_torque_nm);
      m.drive = Drive(sp);
      m.master.mode_of_operation = mode;
      // 0x6072 at the motor's peak torque: the drive caps at what the motor can do, not at its rated torque
      m.master.max_torque_permille = static_cast<int16_t>(
        std::min(32767.0, std::ceil(sp.peak_torque_nm / nm_per_permille)));
      m.master.scaling = wisco_ethercat_ros2_hw::mock::DriveScaling::from_joint(
        sp.counts_per_rev, nm_per_permille, 1, 0.0);
      m.master.scaling.kp_command_factor = 1.0 / sp.kp_per_raw;
      m.master.scaling.kd_command_factor = 1.0 / sp.kd_per_raw;

      const std::string mech = get(p, "mechanism", "");
      if (mech.empty()) {
        const int j = id_of(model_, mjOBJ_JOINT, ji.name, "joint");
        if (model_->jnt_type[j] != mjJNT_HINGE) {throw std::runtime_error(ji.name + ": not a hinge joint");}
        m.qadr = model_->jnt_qposadr[j];
        m.dof = model_->jnt_dofadr[j];
        driven_joints.insert(j);
        continue;
      }
      if (linkage_file.empty()) {throw std::runtime_error(ji.name + ": mechanism given but no linkage_file");}
      auto [it, added] = linkage_index.emplace(mech, static_cast<int>(linkages_.size()));
      if (added) {
        Linkage l;
        l.name = mech;
        const auto loaded = linkage::load_mechanism(linkage_file, mech);
        l.mech = linkage::Mechanism(loaded.geometry);
        for (int k = 0; k < 2; ++k) {
          const int j = id_of(model_, mjOBJ_JOINT, loaded.joints[k], mech + " joint");
          l.qadr[k] = model_->jnt_qposadr[j];
          l.dof[k] = model_->jnt_dofadr[j];
          driven_joints.insert(j);
        }
        // the CAD branch, as ParallelLinkageController picks it
        if (!l.mech.motor_angles({0.0, 0.0}, l.offset)) {
          throw std::runtime_error(mech + ": the linkage does not close at the zero pose");
        }
        l.crank = l.offset;
        linkages_.push_back(l);
      }
      Linkage & l = linkages_[it->second];
      m.linkage = it->second;
      m.crank = static_cast<int>(num(p, "crank", -1.0));
      if (m.crank != 0 && m.crank != 1) {throw std::runtime_error(ji.name + ": crank must be 0 or 1");}
      if (l.motor[m.crank] >= 0) {throw std::runtime_error(ji.name + ": " + mech + " crank taken twice");}
      l.motor[m.crank] = static_cast<int>(i);
      l.direction[m.crank] = num(p, "direction", 1.0) < 0.0 ? -1.0 : 1.0;
    }
    for (const auto & l : linkages_) {
      if (l.motor[0] < 0 || l.motor[1] < 0) {throw std::runtime_error(l.name + ": needs a motor on each crank");}
    }

    // the drives are the only actuation: MuJoCo's position actuators on the driven joints exert nothing
    int disabled = 0;
    for (int a = 0; a < model_->nu; ++a) {
      if (model_->actuator_trntype[a] == mjTRN_JOINT && driven_joints.count(model_->actuator_trnid[2 * a])) {
        std::fill_n(model_->actuator_gainprm + a * mjNGAIN, mjNGAIN, 0.0);
        std::fill_n(model_->actuator_biasprm + a * mjNBIAS, mjNBIAS, 0.0);
        ++disabled;
      }
    }

    // ---- IMU and floating base ----
    if (!info_.sensors.empty()) {
      const auto & sp = info_.sensors.front().parameters;
      imu_quat_ = id_of(model_, mjOBJ_SENSOR, get(sp, "quat_sensor", "imu_quat"), "IMU sensor");
      imu_gyro_ = id_of(model_, mjOBJ_SENSOR, get(sp, "gyro_sensor", "imu_gyro"), "IMU sensor");
      imu_accel_ = id_of(model_, mjOBJ_SENSOR, get(sp, "accel_sensor", "imu_accel"), "IMU sensor");
    }
    const std::string free_joint = get(hw, "free_joint", "");
    if (!free_joint.empty()) {
      base_body_ = model_->jnt_bodyid[id_of(model_, mjOBJ_JOINT, free_joint, "free joint")];
    }

    // ---- ROS: an own node on the controller manager's executor ----
    executor_ = params.executor;
    node_ = std::make_shared<rclcpp::Node>(get(hw, "node_name", "virtual_robot"));
    if (auto ex = executor_.lock()) {ex->add_node(node_);} else {
      RCLCPP_WARN(log, "no executor: reset_world / set_pause will not be served");
    }
    clock_pub_ = node_->create_publisher<rosgraph_msgs::msg::Clock>("/clock", rclcpp::ClockQoS());
    if (base_body_ >= 0) {
      odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>(
        get(hw, "odom_topic", "/sim/ground_truth/odom"), 10);
      odom_.header.frame_id = "odom";
      odom_.child_frame_id = "base_link";
    }
    GroundTruthSettings gs;
    gs.state_topic = get(hw, "state_topic", gs.state_topic);
    gs.state_rate = num(hw, "state_rate", gs.state_rate);
    gs.viewer_host = get(hw, "viewer_host", gs.viewer_host);
    gs.viewer_port = static_cast<int>(num(hw, "viewer_port", gs.viewer_port));
    gs.viewer_rate = num(hw, "viewer_rate", gs.viewer_rate);
    ground_truth_.configure(node_, gs, static_cast<std::size_t>(model_->nq));
    using mujoco_ros2_control_msgs::srv::ResetWorld;
    using mujoco_ros2_control_msgs::srv::SetPause;
    reset_srv_ = node_->create_service<ResetWorld>("~/reset_world",
        [this](const std::shared_ptr<ResetWorld::Request> req, std::shared_ptr<ResetWorld::Response> res) {
          on_reset(req, res);
        });
    pause_srv_ = node_->create_service<SetPause>("~/set_pause",
        [this](const std::shared_ptr<SetPause::Request> req, std::shared_ptr<SetPause::Response> res) {
          on_pause(req, res);
        });

    // the controller manager sleeps on the ROS clock: tick it (frozen world) until its loop reaches write()
    startup_clock_ = node_->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(control_period_)),
      [this]() {
        if (writing_) {
          startup_clock_->cancel();
          return;
        }
        std::lock_guard<std::mutex> lk(data_mutex_);
        data_->time += control_period_;
        publish(data_->time);
      });

    RCLCPP_INFO(log, "%s: %zu drives (mode %d, %zu two-motor mechanisms), %d MuJoCo actuators off, %d physics "
      "step(s) of %.4f s per bus cycle, %s", path.c_str(), motors_.size(), mode, linkages_.size(), disabled,
      substeps_, model_->opt.timestep, released_ ? "running" : "frozen at the keyframe until unpaused");
  } catch (const std::exception & e) {
    RCLCPP_ERROR(log, "%s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> VirtualRobotSystem::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> out;
  for (std::size_t i = 0; i < motors_.size(); ++i) {
    const auto & j = info_.joints[i].name;
    auto & s = motors_[i].st;
    out.emplace_back(j, "position", &s.position);
    out.emplace_back(j, "velocity", &s.velocity);
    out.emplace_back(j, "effort", &s.effort);
    out.emplace_back(j, "mode_of_operation", &s.mode_of_operation);
    out.emplace_back(j, "status_word", &s.status_word);
    out.emplace_back(j, "error_code", &s.error_code);
  }
  if (imu_quat_ >= 0) {
    const auto & n = info_.sensors.front().name;
    const char * names[] = {"orientation.x", "orientation.y", "orientation.z", "orientation.w",
      "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
      "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"};
    for (std::size_t k = 0; k < imu_.size(); ++k) {out.emplace_back(n, names[k], &imu_[k]);}
  }
  return out;
}

std::vector<hardware_interface::CommandInterface> VirtualRobotSystem::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> out;
  for (std::size_t i = 0; i < motors_.size(); ++i) {
    const auto & j = info_.joints[i].name;
    auto & c = motors_[i].cmd;
    out.emplace_back(j, "position", &c.position);
    out.emplace_back(j, "velocity", &c.velocity);
    out.emplace_back(j, "effort", &c.effort);
    out.emplace_back(j, "kp", &c.kp);
    out.emplace_back(j, "kd", &c.kd);
    out.emplace_back(j, "mode_of_operation", &c.mode_of_operation);
    out.emplace_back(j, "reset_fault", &c.reset_fault);
    out.emplace_back(j, "max_torque", &c.max_torque);
  }
  return out;
}

// The frame that came back from the last write(): TxPDO -> state interfaces, and the commands -> next RxPDO image
// (the master's processData, as EthercatDriver::read does on the robot).
return_type VirtualRobotSystem::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  std::lock_guard<std::mutex> lk(data_mutex_);
  for (auto & m : motors_) {m.master.process(have_frame_ ? &m.inbox : nullptr, m.cmd, m.st);}
  have_frame_ = false;
  if (imu_quat_ >= 0) {
    const double * q = data_->sensordata + model_->sensor_adr[imu_quat_];     // w x y z
    const double * g = data_->sensordata + model_->sensor_adr[imu_gyro_];
    const double * a = data_->sensordata + model_->sensor_adr[imu_accel_];
    imu_ = {q[1], q[2], q[3], q[0], g[0], g[1], g[2], a[0], a[1], a[2]};
  }
  return return_type::OK;
}

// One bus cycle (the frame reaches the drives, latching their state now), then one control period of physics with
// the drives' torques, then /clock and the ground truth.
return_type VirtualRobotSystem::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  writing_ = true;
  {
    std::unique_lock<std::mutex> lk(pause_mutex_);
    while (paused_ && rclcpp::ok()) {pause_cv_.wait_for(lk, std::chrono::milliseconds(100));}
  }
  double t;
  {
    std::lock_guard<std::mutex> lk(data_mutex_);
    measure_drives();
    for (auto & m : motors_) {m.inbox = m.drive.cycle(&m.master.rx());}
    have_frame_ = true;
    if (!released_ || frozen_cycles_ > 0) {
      // frozen: the pose holds, time runs (the drives enable and the bus carries the new targets meanwhile)
      if (frozen_cycles_ > 0) {--frozen_cycles_;}
      mju_zero(data_->qvel, model_->nv);
      mj_forward(model_, data_);
      data_->time += control_period_;
    } else {
      for (int s = 0; s < substeps_; ++s) {
        apply_drive_torques();
        mj_step(model_, data_);
      }
    }
    t = data_->time;
    publish(t);
  }
  pace();
  return return_type::OK;
}

bool VirtualRobotSystem::solve_linkage(Linkage & l)
{
  const Vec2 q{data_->qpos[l.qadr[0]], data_->qpos[l.qadr[1]]};
  Vec2 crank;
  l.ok = l.mech.motor_angles(q, crank, l.crank) && l.mech.jacobian(q, l.jacobian, crank);
  if (l.ok) {l.crank = crank;} else if (++unsolved_ % 1000 == 1) {
    RCLCPP_WARN(get_logger(), "%s: the rods cannot close at (%.3f, %.3f): no torque (%lu times)", l.name.c_str(),
      q[0], q[1], static_cast<unsigned long>(unsolved_));
  }
  return l.ok;
}

void VirtualRobotSystem::measure_drives()
{
  for (auto & l : linkages_) {
    if (!solve_linkage(l)) {continue;}
    const Vec2 qd{data_->qvel[l.dof[0]], data_->qvel[l.dof[1]]};
    const Vec2 crank_vel = linkage::mul(l.jacobian, qd);
    for (int c = 0; c < 2; ++c) {
      const double d = l.direction[c];
      motors_[l.motor[c]].drive.measure(d * (l.crank[c] - l.offset[c]), d * crank_vel[c]);
    }
  }
  for (auto & m : motors_) {
    if (m.linkage < 0) {m.drive.measure(data_->qpos[m.qadr], data_->qvel[m.dof]);}
  }
}

void VirtualRobotSystem::apply_drive_torques()
{
  for (auto & m : motors_) {
    if (m.linkage < 0) {data_->qfrc_applied[m.dof] = m.drive.servo(data_->qpos[m.qadr], data_->qvel[m.dof]);}
  }
  for (auto & l : linkages_) {
    if (!solve_linkage(l)) {
      data_->qfrc_applied[l.dof[0]] = data_->qfrc_applied[l.dof[1]] = 0.0;
      continue;
    }
    const Vec2 qd{data_->qvel[l.dof[0]], data_->qvel[l.dof[1]]};
    const Vec2 crank_vel = linkage::mul(l.jacobian, qd);
    Vec2 crank_torque{};
    for (int c = 0; c < 2; ++c) {
      const double d = l.direction[c];
      crank_torque[c] = d * motors_[l.motor[c]].drive.servo(d * (l.crank[c] - l.offset[c]), d * crank_vel[c]);
    }
    const Vec2 joint_torque = linkage::mul(linkage::transpose(l.jacobian), crank_torque);   // virtual work
    data_->qfrc_applied[l.dof[0]] = joint_torque[0];
    data_->qfrc_applied[l.dof[1]] = joint_torque[1];
  }
}

void VirtualRobotSystem::publish(double t)
{
  const rclcpp::Time stamp(static_cast<int64_t>(std::llround(t * 1e9)), RCL_ROS_TIME);
  rosgraph_msgs::msg::Clock clock;
  clock.clock = stamp;
  clock_pub_->publish(clock);
  ground_truth_.update(t, data_->qpos);
  if (odom_pub_) {
    odom_.header.stamp = stamp;
    const double * p = data_->xpos + 3 * base_body_;
    const double * q = data_->xquat + 4 * base_body_;
    odom_.pose.pose.position.x = p[0];
    odom_.pose.pose.position.y = p[1];
    odom_.pose.pose.position.z = p[2];
    odom_.pose.pose.orientation.w = q[0];
    odom_.pose.pose.orientation.x = q[1];
    odom_.pose.pose.orientation.y = q[2];
    odom_.pose.pose.orientation.z = q[3];
    double v[6];
    mj_objectVelocity(model_, data_, mjOBJ_BODY, base_body_, v, 1);      // [angular, linear], body frame
    odom_.twist.twist.angular.x = v[0];
    odom_.twist.twist.angular.y = v[1];
    odom_.twist.twist.angular.z = v[2];
    odom_.twist.twist.linear.x = v[3];
    odom_.twist.twist.linear.y = v[4];
    odom_.twist.twist.linear.z = v[5];
    odom_pub_->publish(odom_);
  }
}

// Keep simulated time at real_time_factor x wall time; when the physics falls behind, carry on from where it is.
void VirtualRobotSystem::pace()
{
  if (real_time_factor_ <= 0.0) {return;}
  const auto now = std::chrono::steady_clock::now();
  const double t = data_->time;
  if (!anchored_) {
    wall_anchor_ = now;
    sim_anchor_ = t;
    anchored_ = true;
    return;
  }
  const auto due = wall_anchor_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>((t - sim_anchor_) / real_time_factor_));
  if (due > now) {
    std::this_thread::sleep_until(due);
  } else if (now - due > std::chrono::milliseconds(100)) {
    anchored_ = false;                     // behind: no catching up in a burst
  }
}

void VirtualRobotSystem::on_reset(
  const std::shared_ptr<mujoco_ros2_control_msgs::srv::ResetWorld::Request> req,
  std::shared_ptr<mujoco_ros2_control_msgs::srv::ResetWorld::Response> res)
{
  std::lock_guard<std::mutex> lk(data_mutex_);
  const int key = req->keyframe.empty() ? keyframe_ : mj_name2id(model_, mjOBJ_KEY, req->keyframe.c_str());
  if (!req->keyframe.empty() && key < 0) {
    res->success = false;
    res->message = "no keyframe " + req->keyframe;
    return;
  }
  const double t = data_->time;                  // simulated time only moves forward
  if (key >= 0) {mj_resetDataKeyframe(model_, data_, key);} else {mj_resetData(model_, data_);}
  data_->time = t;
  mj_forward(model_, data_);
  for (auto & l : linkages_) {l.crank = l.offset;}
  frozen_cycles_ = settle_cycles_;
  anchored_ = false;
  res->success = true;
  res->message = "reset to " + (key >= 0 ? std::string(model_->names + model_->name_keyadr[key]) : "the model's default");
}

void VirtualRobotSystem::on_pause(
  const std::shared_ptr<mujoco_ros2_control_msgs::srv::SetPause::Request> req,
  std::shared_ptr<mujoco_ros2_control_msgs::srv::SetPause::Response> res)
{
  {
    std::lock_guard<std::mutex> lk(pause_mutex_);
    paused_ = req->paused;
    if (!paused_) {
      released_ = true;
      anchored_ = false;
    }
  }
  pause_cv_.notify_all();
  res->success = true;
  res->message = req->paused ? "paused" : "running";
}

}  // namespace hsim_mujoco_plugins

PLUGINLIB_EXPORT_CLASS(hsim_mujoco_plugins::VirtualRobotSystem, hardware_interface::SystemInterface)
