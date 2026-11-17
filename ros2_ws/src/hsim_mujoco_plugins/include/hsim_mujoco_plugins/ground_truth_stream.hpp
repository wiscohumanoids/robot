// GroundTruthStream: the simulated world's state, in the formats the hsim_sim MuJoCo node uses, for the
// ros2_control backends (GroundTruthPlugin under mujoco_ros2_control; VirtualRobotSystem):
//
//   /sim/ground_truth/state   std_msgs/Float64MultiArray [time, qpos...] at `state_rate` Hz: the simulated head
//                             camera renders the world from it (and graders read it)
//   UDP to viewer_host:viewer_port at `viewer_rate` Hz: b"HSIM", uint32 nq, float64 time, float64[nq] qpos,
//                             for tools/viewer.sh's native MuJoCo window (0 = off)
//
// update() is called from the control thread with the latest simulated time and qpos; it keeps its own schedule
// (simulated time) and starts it again when time goes backwards (a world reset).
#ifndef HSIM_MUJOCO_PLUGINS__GROUND_TRUTH_STREAM_HPP_
#define HSIM_MUJOCO_PLUGINS__GROUND_TRUTH_STREAM_HPP_

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace hsim_mujoco_plugins
{

struct GroundTruthSettings
{
  std::string state_topic = "/sim/ground_truth/state";
  double state_rate = 20.0;
  std::string viewer_host = "host.docker.internal";
  int viewer_port = 9870;
  double viewer_rate = 0.0;
};

class GroundTruthStream
{
public:
  GroundTruthStream() = default;
  GroundTruthStream(const GroundTruthStream &) = delete;
  GroundTruthStream & operator=(const GroundTruthStream &) = delete;
  ~GroundTruthStream() {close_socket();}

  // Create the publisher and the viewer socket for a world with `nq` positions. Logs what it streams.
  void configure(const rclcpp::Node::SharedPtr & node, const GroundTruthSettings & s, std::size_t nq)
  {
    state_period_ = s.state_rate > 0.0 ? 1.0 / s.state_rate : 0.0;
    viewer_period_ = s.viewer_rate > 0.0 ? 1.0 / s.viewer_rate : 0.0;
    pub_ = node->create_publisher<std_msgs::msg::Float64MultiArray>(s.state_topic, 5);
    msg_.data.resize(1 + nq);
    packet_.resize(4 + 4 + 8 + 8 * nq);
    nq_ = nq;
    close_socket();
    if (viewer_period_ > 0.0) {
      addrinfo hints{}, * res = nullptr;
      hints.ai_family = AF_INET;
      hints.ai_socktype = SOCK_DGRAM;
      if (getaddrinfo(s.viewer_host.c_str(), std::to_string(s.viewer_port).c_str(), &hints, &res) == 0 && res) {
        std::memcpy(&viewer_addr_, res->ai_addr, sizeof(sockaddr_in));
        freeaddrinfo(res);
        sock_ = socket(AF_INET, SOCK_DGRAM, 0);
      } else {
        RCLCPP_WARN(node->get_logger(), "viewer stream off: cannot resolve %s", s.viewer_host.c_str());
        viewer_period_ = 0.0;
      }
    }
    RCLCPP_INFO(node->get_logger(), "ground truth: %s at %.0f Hz (nq %zu), viewer stream %s",
      s.state_topic.c_str(), s.state_rate, nq, viewer_period_ > 0.0 ? "on" : "off");
  }

  // Publish / send when due.
  void update(double t, const double * qpos)
  {
    if (!pub_) {return;}
    if (t < last_time_) {next_state_ = next_viewer_ = 0.0;}   // the world was reset: start the schedules again
    last_time_ = t;
    if (state_period_ > 0.0 && t >= next_state_) {
      next_state_ = t + state_period_;
      msg_.data[0] = t;
      std::memcpy(&msg_.data[1], qpos, sizeof(double) * nq_);
      pub_->publish(msg_);
    }
    if (viewer_period_ > 0.0 && sock_ >= 0 && t >= next_viewer_) {
      next_viewer_ = t + viewer_period_;
      const uint32_t nq = static_cast<uint32_t>(nq_);
      std::memcpy(packet_.data(), "HSIM", 4);
      std::memcpy(packet_.data() + 4, &nq, 4);                     // little-endian on every target we build for
      std::memcpy(packet_.data() + 8, &t, 8);
      std::memcpy(packet_.data() + 16, qpos, 8 * nq_);
      (void)sendto(sock_, packet_.data(), packet_.size(), 0,
        reinterpret_cast<const sockaddr *>(&viewer_addr_), sizeof(viewer_addr_));
    }
  }

  void reset()
  {
    close_socket();
    pub_.reset();
  }

private:
  void close_socket()
  {
    if (sock_ >= 0) {
      close(sock_);
      sock_ = -1;
    }
  }

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_;
  std_msgs::msg::Float64MultiArray msg_;
  std::vector<char> packet_;
  std::size_t nq_ = 0;
  double state_period_ = 0.05, viewer_period_ = 0.0, next_state_ = 0.0, next_viewer_ = 0.0, last_time_ = 0.0;
  int sock_ = -1;
  sockaddr_in viewer_addr_{};
};

}  // namespace hsim_mujoco_plugins

#endif  // HSIM_MUJOCO_PLUGINS__GROUND_TRUTH_STREAM_HPP_
