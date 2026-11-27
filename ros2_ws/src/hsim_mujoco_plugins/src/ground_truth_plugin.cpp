// GroundTruthPlugin: the simulated world's state, published from inside mujoco_ros2_control.
//
// On the sim stack's ros2_control backend MuJoCo runs inside the controller manager's process (the upstream
// mujoco_ros2_control hardware plugin), not in the hsim_sim MuJoCo node. Two things the rest of the simulation needs
// came from that node, and this plugin provides them in the same formats (ground_truth_stream.hpp):
// /sim/ground_truth/state for the simulated head camera and graders, and the UDP stream for tools/sim/viewer.sh. The
// world file the backend runs and the env's world.mjb have the same qpos layout (env_build.py writes both).
//
// Parameters (under mujoco_plugins.<name>.): type, state_topic, state_rate, viewer_host, viewer_port, viewer_rate.
// Runs in the control thread's read step (update_read), on the data the hardware interface just copied out.
#include <memory>
#include <string>

#include "hsim_mujoco_plugins/ground_truth_stream.hpp"
#include "mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"

namespace hsim_mujoco_plugins
{

class GroundTruthPlugin : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase
{
public:
  bool init(rclcpp::Node::SharedPtr node, const mjModel * model, mjData *) override
  {
    node_ = node;
    // parameters are node-level: find this instance's prefix (mujoco_plugins.<name>.) from the sub-node namespace
    const std::string ns = node->get_sub_namespace();
    prefix_ = "mujoco_plugins." + (ns.empty() ? std::string("ground_truth") : ns) + ".";
    GroundTruthSettings s;
    s.state_topic = param<std::string>("state_topic", s.state_topic);
    s.state_rate = param<double>("state_rate", s.state_rate);
    s.viewer_host = param<std::string>("viewer_host", s.viewer_host);
    s.viewer_port = static_cast<int>(param<int64_t>("viewer_port", s.viewer_port));
    s.viewer_rate = param<double>("viewer_rate", s.viewer_rate);
    stream_.configure(node, s, static_cast<std::size_t>(model->nq));
    return true;
  }

  void update_read(const mjModel *, mjData * data) override
  {
    stream_.update(data->time, data->qpos);
  }

  void cleanup() override {stream_.reset();}

private:
  template<typename T>
  T param(const std::string & key, const T & fallback)
  {
    const std::string name = prefix_ + key;
    if (!node_->has_parameter(name)) {node_->declare_parameter<T>(name, fallback);}
    return node_->get_parameter(name).get_value<T>();
  }

  rclcpp::Node::SharedPtr node_;
  std::string prefix_;
  GroundTruthStream stream_;
};

}  // namespace hsim_mujoco_plugins

PLUGINLIB_EXPORT_CLASS(hsim_mujoco_plugins::GroundTruthPlugin, mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
