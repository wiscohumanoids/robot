// C++/Python model parity: the ArmController's GravityModel must compute the
// same gravity torques and tool FK as arm_stack's ArmModel (sim, calibration)
// for the same URDF and calibration file. The golden file is written by
// tools/make_parity_golden.py; it holds random poses, a calibration with
// link inertials, link masses, friction and kinematic overlay entries, and the
// Python model's outputs.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "wisco_ethercat_ros2_hw/gravity_model.hpp"
#include "yaml-cpp/yaml.h"

namespace
{

using wisco_ethercat_ros2_hw::GravityModel;

constexpr double kTorqueTol = 1e-9;   // Nm
constexpr double kPoseTol = 1e-9;     // m, and rotation matrix entries

std::string source_path(const std::string & rel) {return std::string(WISCO_SOURCE_DIR) + "/" + rel;}

std::string read_file(const std::string & path)
{
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::vector<double> vec(const YAML::Node & n)
{
  std::vector<double> out;
  for (const auto & v : n) {out.push_back(v.as<double>());}
  return out;
}

class GravityParity : public ::testing::Test
{
protected:
  void SetUp() override
  {
    golden_ = YAML::LoadFile(source_path("test/data/arm01_parity.json"));
    ASSERT_TRUE(golden_["cases"] && golden_["cases"].size() > 0);
    urdf_ = read_file(source_path(golden_["urdf"].as<std::string>()));
    ASSERT_FALSE(urdf_.empty());
    for (const auto & j : golden_["joints"]) {joints_.push_back(j.as<std::string>());}
    // the calibration as the controller gets it: a YAML file on disk
    cal_path_ = ::testing::TempDir() + "arm01_parity_calibration.yaml";
    YAML::Emitter out;
    out << golden_["calibration"];
    std::ofstream(cal_path_) << out.c_str() << "\n";
    // link_masses_kg as the launch file passes it (link_mass_names / values)
    for (const auto & kv : golden_["link_masses_kg"]) {
      mass_links_.push_back(kv.first.as<std::string>());
      mass_values_.push_back(kv.second.IsNull() ? std::numeric_limits<double>::quiet_NaN() :
        kv.second.as<double>());
    }
  }

  bool build(GravityModel & m, wisco_ethercat_ros2_hw::arm::LawConfig & cfg, bool calibrated)
  {
    cfg.coulomb.assign(joints_.size(), 0.0);
    cfg.viscous.assign(joints_.size(), 0.0);
    std::string error;
    const bool ok = calibrated ?
      m.build(urdf_, joints_, cal_path_, mass_links_, mass_values_, cfg, error) :
      m.build(urdf_, joints_, "", {}, {}, cfg, error);
    EXPECT_TRUE(ok) << error;
    return ok;
  }

  YAML::Node golden_;
  std::string urdf_, cal_path_;
  std::vector<std::string> joints_, mass_links_;
  std::vector<double> mass_values_;
};

TEST_F(GravityParity, CalibratedGravityMatchesPython)
{
  GravityModel m;
  wisco_ethercat_ros2_hw::arm::LawConfig cfg;
  ASSERT_TRUE(build(m, cfg, true));
  std::vector<double> g(joints_.size());
  double worst = 0.0;
  for (const auto & c : golden_["cases"]) {
    const auto q = vec(c["q"]);
    const auto expect = vec(c["gravity"]);
    m.gravity(q, g);
    for (size_t i = 0; i < joints_.size(); ++i) {
      EXPECT_NEAR(g[i], expect[i], kTorqueTol) << joints_[i];
      worst = std::max(worst, std::fabs(g[i] - expect[i]));
    }
  }
  std::ostringstream s;
  s << std::scientific << worst;
  RecordProperty("worst_gravity_error_nm", s.str());
  std::printf("worst |C++ - Python| gravity error: %.3e Nm\n", worst);
}

TEST_F(GravityParity, NominalGravityMatchesPython)
{
  GravityModel m;
  wisco_ethercat_ros2_hw::arm::LawConfig cfg;
  ASSERT_TRUE(build(m, cfg, false));
  std::vector<double> g(joints_.size());
  for (const auto & c : golden_["cases"]) {
    const auto expect = vec(c["gravity_nominal"]);
    m.gravity(vec(c["q"]), g);
    for (size_t i = 0; i < joints_.size(); ++i) {EXPECT_NEAR(g[i], expect[i], kTorqueTol) << joints_[i];}
  }
}

TEST_F(GravityParity, CalibrationChangesGravity)
{
  // guards the golden file itself: the calibration must matter, or the
  // calibrated test above proves nothing about the calibration layers
  double diff = 0.0;
  for (const auto & c : golden_["cases"]) {
    const auto a = vec(c["gravity"]), b = vec(c["gravity_nominal"]);
    for (size_t i = 0; i < a.size(); ++i) {diff = std::max(diff, std::fabs(a[i] - b[i]));}
  }
  EXPECT_GT(diff, 0.05);
}

TEST_F(GravityParity, FrictionMatchesPython)
{
  GravityModel m;
  wisco_ethercat_ros2_hw::arm::LawConfig cfg;
  ASSERT_TRUE(build(m, cfg, true));
  const auto coulomb = vec(golden_["coulomb"]), viscous = vec(golden_["viscous"]);
  for (size_t i = 0; i < joints_.size(); ++i) {
    EXPECT_DOUBLE_EQ(cfg.coulomb[i], coulomb[i]) << joints_[i];
    EXPECT_DOUBLE_EQ(cfg.viscous[i], viscous[i]) << joints_[i];
  }
}

TEST_F(GravityParity, ToolFkMatchesPython)
{
  GravityModel m;
  wisco_ethercat_ros2_hw::arm::LawConfig cfg;
  ASSERT_TRUE(build(m, cfg, true));
  for (const auto & c : golden_["cases"]) {
    const auto q = vec(c["q"]);
    for (const auto & kv : c["fk"]) {
      const auto frame = kv.first.as<std::string>();
      pinocchio::SE3 M;
      ASSERT_TRUE(m.frame_placement(q, frame, M)) << frame;
      const auto t = vec(kv.second["translation"]);
      for (int a = 0; a < 3; ++a) {
        EXPECT_NEAR(M.translation()[a], t[a], kPoseTol) << frame;
        const auto row = vec(kv.second["rotation"][a]);
        for (int b = 0; b < 3; ++b) {EXPECT_NEAR(M.rotation()(a, b), row[b], kPoseTol) << frame;}
      }
    }
  }
}

TEST_F(GravityParity, UnknownFrameIsRejected)
{
  GravityModel m;
  wisco_ethercat_ros2_hw::arm::LawConfig cfg;
  ASSERT_TRUE(build(m, cfg, true));
  pinocchio::SE3 M;
  EXPECT_FALSE(m.frame_placement(std::vector<double>(joints_.size(), 0.0), "no_such_frame", M));
}

}  // namespace
