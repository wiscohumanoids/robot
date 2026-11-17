// Load a parallel-joint geometry (parallel_linkage.hpp) from robots/<robot>/linkage.json, the rod geometry measured
// from the CAD. yaml-cpp reads JSON. Not real-time: call at configure time.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "wisco_ethercat_ros2_hw/parallel_linkage.hpp"

namespace wisco_ethercat_ros2_hw::linkage
{

inline Vec3 vec3_of(const YAML::Node & n, const std::string & what)
{
  if (!n || !n.IsSequence() || n.size() != 3) {throw std::runtime_error(what + ": expected 3 numbers");}
  return {n[0].as<double>(), n[1].as<double>(), n[2].as<double>()};
}

struct LoadedMechanism
{
  Geometry geometry;
  std::vector<std::string> joints;   // the two joint names, in order (first axis, second axis)
};

// Mechanism `name` (e.g. "left_ankle") from a linkage.json file. Throws std::runtime_error with the reason.
inline LoadedMechanism load_mechanism(const std::string & path, const std::string & name)
{
  const YAML::Node root = YAML::LoadFile(path);
  const YAML::Node m = root[name];
  if (!m) {throw std::runtime_error(path + ": no mechanism '" + name + "'");}
  LoadedMechanism out;
  for (const auto & j : m["joints"]) {out.joints.push_back(j.as<std::string>());}
  if (out.joints.size() != 2) {throw std::runtime_error(name + ": expected 2 joints");}
  Geometry & g = out.geometry;
  g.axis1_pt = vec3_of(m["axis1_pt"], name + ".axis1_pt");
  g.axis1_dir = vec3_of(m["axis1_dir"], name + ".axis1_dir");
  g.axis2_pt = vec3_of(m["axis2_pt"], name + ".axis2_pt");
  g.axis2_dir = vec3_of(m["axis2_dir"], name + ".axis2_dir");
  const std::string on = m["crank_on"].as<std::string>();
  if (on != "fixed" && on != "moving") {throw std::runtime_error(name + ".crank_on: fixed or moving");}
  g.crank_on_fixed = on == "fixed";
  const YAML::Node cranks = m["cranks"];
  if (!cranks || cranks.size() != 2) {throw std::runtime_error(name + ": expected 2 cranks");}
  for (std::size_t i = 0; i < 2; ++i) {
    const std::string w = name + ".cranks[" + std::to_string(i) + "]";
    g.cranks[i].axis_pt = vec3_of(cranks[i]["axis_pt"], w + ".axis_pt");
    g.cranks[i].axis_dir = vec3_of(cranks[i]["axis_dir"], w + ".axis_dir");
    g.cranks[i].crank_ball = vec3_of(cranks[i]["crank_ball"], w + ".crank_ball");
    g.cranks[i].far_ball = vec3_of(cranks[i]["far_ball"], w + ".far_ball");
    g.cranks[i].rod_len = cranks[i]["rod_len"].as<double>();
  }
  return out;
}

}  // namespace wisco_ethercat_ros2_hw::linkage
