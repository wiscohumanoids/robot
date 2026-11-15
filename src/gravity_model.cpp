#include "wisco_ethercat_ros2_hw/gravity_model.hpp"

#include <algorithm>
#include <cmath>
#include <map>

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "yaml-cpp/yaml.h"

namespace wisco_ethercat_ros2_hw
{

bool GravityModel::build(
  const std::string & urdf_xml, const std::vector<std::string> & joints,
  const std::string & calibration_file, const std::vector<std::string> & mass_links,
  const std::vector<double> & mass_values, arm::LawConfig & cfg, std::string & error)
{
  pinocchio::Model full;
  try {
    pinocchio::urdf::buildModelFromXML(urdf_xml, full);
  } catch (const std::exception & e) {
    error = std::string("cannot parse robot_description: ") + e.what();
    return false;
  }
  std::vector<pinocchio::JointIndex> lock;
  for (pinocchio::JointIndex j = 1; j < static_cast<pinocchio::JointIndex>(full.njoints); ++j) {
    if (std::find(joints.begin(), joints.end(), full.names[j]) == joints.end()) {
      lock.push_back(j);
    }
  }
  model_ = lock.empty() ? full :
    pinocchio::buildReducedModel(full, lock, pinocchio::neutral(full));
  model_.gravity.linear(Eigen::Vector3d(0, 0, -9.81));
  for (const auto & name : joints) {
    if (!model_.existJointName(name)) {
      error = "joint '" + name + "' is not in robot_description";
      return false;
    }
    const auto id = model_.getJointId(name);
    jid_.push_back(id);
    vidx_.push_back(model_.joints[id].idx_v());
  }
  if (static_cast<size_t>(model_.nv) != joints.size()) {
    error = "robot_description has " + std::to_string(model_.nv) + " driven dofs, config lists " +
      std::to_string(joints.size());
    return false;
  }
  // child link of each driven joint, for the per-link calibration fields. As
  // ArmModel.link_names(): the joint's first body frame. Bodies merged in from
  // locked joints (the wrist crank and rod, on link5's joint) have no entry:
  // their mass is part of link5's, which stage 5 fits as a whole.
  std::map<std::string, pinocchio::JointIndex> link_joint;
  for (const auto id : jid_) {
    for (const auto & f : model_.frames) {
      if (f.type == pinocchio::BODY && f.parentJoint == id) {
        link_joint[f.name] = id;
        break;
      }
    }
  }
  auto set_mass = [&](const std::string & link, double mass, const Eigen::Vector3d * com) {
      auto it = link_joint.find(link);
      if (it == link_joint.end() || !(mass > 0)) {return;}
      auto & I = model_.inertias[it->second];
      const double scale = I.mass() > 0 ? mass / I.mass() : 1.0;
      I = pinocchio::Inertia(mass, com ? *com : I.lever(), I.inertia().matrix() * scale);
    };
  for (size_t k = 0; k < mass_links.size() && k < mass_values.size(); ++k) {
    if (std::isfinite(mass_values[k])) {set_mass(mass_links[k], mass_values[k], nullptr);}
  }
  if (!calibration_file.empty()) {
    try {
      const YAML::Node cal = YAML::LoadFile(calibration_file);
      if (cal["link_inertials"] && cal["link_inertials"].IsMap()) {
        for (const auto & kv : cal["link_inertials"]) {
          const auto link = kv.first.as<std::string>();
          const auto & e = kv.second;
          const double mass = e["mass_kg"] && !e["mass_kg"].IsNull() ? e["mass_kg"].as<double>() : -1;
          Eigen::Vector3d com;
          bool has_com = e["com_m"] && e["com_m"].IsSequence() && e["com_m"].size() == 3;
          if (has_com) {
            for (int i = 0; i < 3; ++i) {
              if (e["com_m"][i].IsNull()) {has_com = false;} else {com[i] = e["com_m"][i].as<double>();}
            }
          }
          auto it = link_joint.find(link);
          if (it != link_joint.end()) {
            set_mass(link, mass > 0 ? mass : model_.inertias[it->second].mass(), has_com ? &com : nullptr);
          }
        }
      }
      if (cal["friction"] && cal["friction"].IsMap()) {
        for (size_t i = 0; i < joints.size(); ++i) {
          const auto f = cal["friction"][joints[i]];
          if (!f) {continue;}
          if (f["coulomb_nm"] && !f["coulomb_nm"].IsNull()) {cfg.coulomb[i] = f["coulomb_nm"].as<double>();}
          if (f["viscous_nm_s_per_rad"] && !f["viscous_nm_s_per_rad"].IsNull()) {
            cfg.viscous[i] = f["viscous_nm_s_per_rad"].as<double>();
          }
        }
      }
      if (cal["kinematic_overlay"] && cal["kinematic_overlay"].IsMap()) {
        for (size_t i = 0; i < joints.size(); ++i) {
          const auto o = cal["kinematic_overlay"][joints[i]];
          if (!o) {continue;}
          Eigen::Vector3d dxyz = Eigen::Vector3d::Zero(), drot = Eigen::Vector3d::Zero();
          for (int a = 0; a < 3; ++a) {
            if (o["dxyz"] && o["dxyz"].size() == 3) {dxyz[a] = o["dxyz"][a].as<double>();}
            if (o["drot"] && o["drot"].size() == 3) {drot[a] = o["drot"][a].as<double>();}
          }
          auto & M = model_.jointPlacements[jid_[i]];
          M = pinocchio::SE3(M.rotation() * pinocchio::exp3(drot), M.translation() + dxyz);
        }
      }
    } catch (const std::exception & e) {
      error = "cannot read calibration file '" + calibration_file + "': " + e.what();
      return false;
    }
  }
  data_ = std::make_unique<pinocchio::Data>(model_);
  q_ = pinocchio::neutral(model_);
  return true;
}

void GravityModel::gravity(const std::vector<double> & q, std::vector<double> & g)
{
  for (size_t i = 0; i < jid_.size(); ++i) {q_[model_.joints[jid_[i]].idx_q()] = q[i];}
  const auto & tau = pinocchio::computeGeneralizedGravity(model_, *data_, q_);
  for (size_t i = 0; i < vidx_.size(); ++i) {g[i] = tau[vidx_[i]];}
}

bool GravityModel::frame_placement(
  const std::vector<double> & q, const std::string & frame, pinocchio::SE3 & out)
{
  if (!data_ || !model_.existFrame(frame)) {return false;}
  for (size_t i = 0; i < jid_.size(); ++i) {q_[model_.joints[jid_[i]].idx_q()] = q[i];}
  pinocchio::forwardKinematics(model_, *data_, q_);
  out = pinocchio::updateFramePlacement(model_, *data_, model_.getFrameId(frame));
  return true;
}

}  // namespace wisco_ethercat_ros2_hw
