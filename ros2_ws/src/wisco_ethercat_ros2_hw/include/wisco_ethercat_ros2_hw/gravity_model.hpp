#ifndef WISCO_ETHERCAT_ROS2_HW__GRAVITY_MODEL_HPP_
#define WISCO_ETHERCAT_ROS2_HW__GRAVITY_MODEL_HPP_

// Gravity torques from the URDF, with the calibration layered on top the same
// way arm_stack/model/robot.py does it: weighed masses and fitted COMs per
// link (stage 5) and the mocap kinematic overlay. Joints not driven (the
// wrist parallelogram's mimic joints) are locked at zero.
//
// Used by the ArmController's RT loop; test/test_gravity_parity.cpp checks it
// against the Python model (test/data/arm01_parity.json).

#include <memory>
#include <string>
#include <vector>

#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include "wisco_ethercat_ros2_hw/arm_law.hpp"

namespace wisco_ethercat_ros2_hw
{

class GravityModel
{
public:
  // Writes the calibration's friction into cfg.coulomb / cfg.viscous (sized to
  // joints by the caller).
  bool build(
    const std::string & urdf_xml, const std::vector<std::string> & joints,
    const std::string & calibration_file, const std::vector<std::string> & mass_links,
    const std::vector<double> & mass_values, arm::LawConfig & cfg, std::string & error);

  // Not allocating: q_ and data_ are preallocated.
  void gravity(const std::vector<double> & q, std::vector<double> & g);

  // Placement of a frame in the base frame at joint positions q (config
  // order). Not for the RT loop. False if the frame does not exist.
  bool frame_placement(const std::vector<double> & q, const std::string & frame, pinocchio::SE3 & out);

  const pinocchio::Model & model() const {return model_;}

  // Payload: a point mass fixed in `frame` (tool0), carried as extra inertia on
  // the frame's parent joint body, so gravity() includes it. Call after load(),
  // so the calibrated inertia is the base the payload is added to.
  bool set_payload_frame(const std::string & frame)
  {
    if (!model_.existFrame(frame)) {return false;}
    const auto & f = model_.frames[model_.getFrameId(frame)];
    if (f.parentJoint == 0) {return false;}
    payload_joint_ = f.parentJoint;
    payload_placement_ = f.placement;
    payload_base_ = model_.inertias[payload_joint_];
    return true;
  }

  // RT safe (fixed-size Eigen, no allocation). kg <= 0 removes it.
  void set_payload(double kg, double x, double y, double z)
  {
    if (payload_joint_ == 0) {return;}
    model_.inertias[payload_joint_] = payload_base_;
    if (kg > 0) {
      const Eigen::Vector3d c = payload_placement_.act(Eigen::Vector3d(x, y, z));
      model_.inertias[payload_joint_] =
        payload_base_ + pinocchio::Inertia(kg, c, pinocchio::Symmetric3::Zero());
    }
  }

private:
  pinocchio::Model model_;
  std::unique_ptr<pinocchio::Data> data_;
  Eigen::VectorXd q_;
  std::vector<pinocchio::JointIndex> jid_;
  std::vector<int> vidx_;
  pinocchio::JointIndex payload_joint_ = 0;
  pinocchio::SE3 payload_placement_ = pinocchio::SE3::Identity();
  pinocchio::Inertia payload_base_ = pinocchio::Inertia::Zero();
};

}  // namespace wisco_ethercat_ros2_hw

#endif  // WISCO_ETHERCAT_ROS2_HW__GRAVITY_MODEL_HPP_
