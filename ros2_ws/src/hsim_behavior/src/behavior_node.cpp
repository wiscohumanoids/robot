// Task behavior tree (node name: `behavior`). BehaviorTree.CPP v4.
//
// Action server /behavior/execute_plan (hsim_interfaces/ExecutePlan): receives a skill plan from
// the planner, turns it into a tree, ticks it at 20 Hz, reports progress, and returns a plain-words
// reason on failure (which goes back to the planner for replanning).
//
//   ReactiveSequence                 <- re-checks IsSafe on every tick; halts the running skill if unsafe
//   |- IsSafe                        (/safety/status not ERROR, and fresh)
//   '- Sequence
//      |- NavigateTo target=...      object memory -> pick a side it can stand at (own map) -> Nav2 to a
//      |                             staging spot -> final approach to arm's reach (own lidar + pose)
//      |- FindObject target=...      in memory? else look around once and check again
//      |- SearchFor target=... hints=a;b   visit hinted places and look around until it shows up
//      |- LookAround                 full turn (Nav2 spin)
//      |- Say text=...               /robot/say
//      '- Pick / Place               not available until Phase 5: fail with that reason
//
// Geometry never comes from the planner: targets are names, resolved against /world/objects.

#include <algorithm>
#include <chrono>
#include <limits>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "behaviortree_cpp/bt_factory.h"
#include "behaviortree_cpp/loggers/groot2_publisher.h"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "hsim_interfaces/action/execute_plan.hpp"
#include "hsim_interfaces/action/manipulate.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "nav2_msgs/action/spin.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "vision_msgs/msg/detection3_d_array.hpp"

using namespace std::chrono_literals;
using ExecutePlan = hsim_interfaces::action::ExecutePlan;
using NavigateToPose = nav2_msgs::action::NavigateToPose;
using Spin = nav2_msgs::action::Spin;
using ManipulateAction = hsim_interfaces::action::Manipulate;

struct KnownObject {
  std::string id, label;
  double x, y, z, sx, sy, sz, confidence;
};

// Everything the tree nodes share: ROS handles, latest world/safety state, progress, failure reason.
struct Context {
  bool off_map_blocked = true;   // param: off the map = wall (prebuilt map) vs unknown (SLAM: the map grows)
  rclcpp::Node::SharedPtr node;
  rclcpp_action::Client<NavigateToPose>::SharedPtr nav;
  rclcpp_action::Client<Spin>::SharedPtr spin;
  rclcpp_action::Client<ManipulateAction>::SharedPtr manip;   // /skills/manipulate (hsim_manipulation, VLA)
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr say;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr approach_cmd;   // -> twist_mux "approach"
  std::shared_ptr<tf2_ros::Buffer> tf;
  std::mutex mtx;
  std::vector<KnownObject> objects;
  nav_msgs::msg::OccupancyGrid::SharedPtr map;       // the robot's own map (known map or SLAM)
  double map_axis = 0.0;   // the room's dominant wall direction in the map frame, rad in [0, pi/2) (Manhattan world):
                           // a prebuilt map is drawn along the walls (0); a SLAM map starts at the robot's start
                           // heading, so walls and furniture can sit at any angle to its axes
  sensor_msgs::msg::LaserScan::SharedPtr scan;       // latest /scan (lidar, or from stereo depth)
  rclcpp::Time scan_stamp{0, 0, RCL_ROS_TIME};
  double scan_offset = 0.0;                          // scan origin, meters ahead of base_footprint
  // Where the robot last stepped up close to a surface (closer than Nav2's body radius allows):
  // the next motion skill backs off first. Cleared once it has, or if the robot was moved away.
  std::optional<std::array<double, 3>> close_up;
  int safety_level = -1;
  std::string safety_message = "no safety status yet";
  rclcpp::Time safety_stamp{0, 0, RCL_ROS_TIME};
  int current_index = -1;
  std::string current_skill, status, failure;

  static std::string lower(std::string s) {
    for (auto& c : s) c = std::tolower(c);
    return s;
  }
  // Best match by label (exact, then contains), highest confidence first.
  std::optional<KnownObject> resolve(const std::string& target) {
    std::lock_guard<std::mutex> lk(mtx);
    const std::string t = lower(target);
    std::optional<KnownObject> best;
    int best_rank = 0;
    for (const auto& o : objects) {
      const std::string l = lower(o.label), id = lower(o.id);
      int rank = (l == t || id == t) ? 3 : (l.find(t) != std::string::npos || t.find(l) != std::string::npos) ? 1 : 0;
      if (rank == 0) continue;
      if (!best || rank > best_rank || (rank == best_rank && o.confidence > best->confidence)) {
        best = o;
        best_rank = rank;
      }
    }
    return best;
  }
  std::optional<std::array<double, 3>> robot_pose(const char* frame = "map") {
    try {
      auto t = tf->lookupTransform(frame, "base_footprint", tf2::TimePointZero);
      const auto& q = t.transform.rotation;
      double yaw = std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
      return std::array<double, 3>{t.transform.translation.x, t.transform.translation.y, yaw};
    } catch (...) {
      return std::nullopt;
    }
  }
  void fail(const std::string& why) { failure = why; }

  // The furniture a small object rests on (the red block's table), if it is in memory.
  std::optional<KnownObject> support_of(const KnownObject& o) {
    std::lock_guard<std::mutex> lk(mtx);
    std::optional<KnownObject> best;
    for (const auto& b : objects) {
      if (b.id == o.id || std::max(b.sx, b.sy) < 0.35) continue;
      bool under = std::abs(o.x - b.x) <= b.sx / 2 + 0.05 && std::abs(o.y - b.y) <= b.sy / 2 + 0.05;
      double top = b.z + b.sz / 2;
      if (under && top <= o.z + 0.05 && top >= o.z - o.sz / 2 - 0.15 && (!best || b.sx * b.sy < best->sx * best->sy))
        best = b;
    }
    return best;
  }
  // Anything solid within r of (x, y) on the robot's own map? Unknown cells inside the map count as free (SLAM);
  // cells OFF the map count as blocked: the robot can't stand somewhere it has no map for (a counter's back side
  // against the wall would otherwise put the staging spot outside the room, which Nav2 rejects).
  bool blocked(double x, double y, double r) {
    std::lock_guard<std::mutex> lk(mtx);
    if (!map) return false;
    const auto& in = map->info;
    const double res = in.resolution;
    const int ci = static_cast<int>(std::floor((x - in.origin.position.x) / res));
    const int cj = static_cast<int>(std::floor((y - in.origin.position.y) / res));
    const int n = static_cast<int>(std::ceil(r / res));
    for (int dj = -n; dj <= n; ++dj)
      for (int di = -n; di <= n; ++di) {
        if (std::hypot(di, dj) * res > r) continue;
        const int i = ci + di, j = cj + dj;
        if (i < 0 || j < 0 || i >= static_cast<int>(in.width) || j >= static_cast<int>(in.height)) {
          if (off_map_blocked) return true;
          continue;
        }
        if (map->data[j * in.width + i] >= 65) return true;
      }
    return false;
  }
  // Walk from (x, y) along (dx, dy) on the robot's own map; distance to the first solid cell.
  // nullopt: nothing solid within max_d, or no map.
  std::optional<double> first_solid(double x, double y, double dx, double dy, double max_d) {
    std::lock_guard<std::mutex> lk(mtx);
    if (!map) return std::nullopt;
    const auto& in = map->info;
    for (double t = 0.0; t <= max_d; t += in.resolution / 2) {
      const int i = static_cast<int>(std::floor((x + dx * t - in.origin.position.x) / in.resolution));
      const int j = static_cast<int>(std::floor((y + dy * t - in.origin.position.y) / in.resolution));
      if (i < 0 || j < 0 || i >= static_cast<int>(in.width) || j >= static_cast<int>(in.height)) continue;
      if (map->data[j * in.width + i] >= 65) return t;
    }
    return std::nullopt;
  }
  bool has_map() { std::lock_guard<std::mutex> lk(mtx); return static_cast<bool>(map); }
  // What the robot's map says at (x, y): -1 unknown / off the map, 0 free, 1 occupied.
  int cell(double x, double y) {
    std::lock_guard<std::mutex> lk(mtx);
    if (!map) return -1;
    const auto& in = map->info;
    const int i = static_cast<int>(std::floor((x - in.origin.position.x) / in.resolution));
    const int j = static_cast<int>(std::floor((y - in.origin.position.y) / in.resolution));
    if (i < 0 || j < 0 || i >= static_cast<int>(in.width) || j >= static_cast<int>(in.height)) return -1;
    const int v = map->data[j * in.width + i];
    return v < 0 ? -1 : (v >= 65 ? 1 : 0);
  }
  // Nearest scan return toward body angle `dir` (0 = ahead, pi = behind, within +-half_angle),
  // as a distance from base_footprint along that direction.
  // nullopt: no fresh scan or the scan doesn't cover that direction. +inf: nothing within range.
  std::optional<double> range_toward(double dir, double half_angle) {
    std::lock_guard<std::mutex> lk(mtx);
    if (!scan || (node->now() - scan_stamp).seconds() > 0.5) return std::nullopt;
    double best = std::numeric_limits<double>::infinity();
    bool covered = false;
    for (size_t k = 0; k < scan->ranges.size(); ++k) {
      const double a = scan->angle_min + k * scan->angle_increment;
      const double off = std::atan2(std::sin(a - dir), std::cos(a - dir));
      if (std::abs(off) > half_angle) continue;
      covered = true;
      const double r = scan->ranges[k];
      if (std::isfinite(r) && r >= scan->range_min && r <= scan->range_max)
        best = std::min(best, r * std::cos(off) + scan_offset * std::cos(dir));
    }
    if (!covered) return std::nullopt;
    return best;
  }
  std::optional<double> front_range(double half_angle) { return range_toward(0.0, half_angle); }
  // Heading error to the flat surface straight ahead, from the scan alone (fits a line to the returns
  // within +-35 deg and 1 m). Independent of the pose estimate. nullopt: no clean flat surface seen.
  // capture time of the latest scan (header stamp), for "measured after I stopped" checks
  std::optional<rclcpp::Time> scan_capture_time() {
    std::lock_guard<std::mutex> lk(mtx);
    if (!scan) return std::nullopt;
    return rclcpp::Time(scan->header.stamp, RCL_ROS_TIME);
  }
  std::optional<double> surface_heading_error() {
    std::lock_guard<std::mutex> lk(mtx);
    if (!scan || (node->now() - scan_stamp).seconds() > 0.5) return std::nullopt;
    std::vector<std::pair<double, double>> pts;
    for (size_t k = 0; k < scan->ranges.size(); ++k) {
      const double a = scan->angle_min + k * scan->angle_increment, r = scan->ranges[k];
      if (std::abs(a) > 0.61 || !std::isfinite(r) || r < scan->range_min || r > 1.0) continue;
      pts.push_back({scan_offset + r * std::cos(a), r * std::sin(a)});
    }
    if (pts.size() < 10) return std::nullopt;
    double mx = 0, my = 0;
    for (auto& p : pts) { mx += p.first; my += p.second; }
    mx /= pts.size(); my /= pts.size();
    double sxx = 0, syy = 0, sxy = 0;
    for (auto& p : pts) { const double dx = p.first - mx, dy = p.second - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
    const double th = 0.5 * std::atan2(2 * sxy, sxx - syy);          // line direction (principal axis)
    const double ux = std::cos(th), uy = std::sin(th);
    double rms = 0, lo = 1e9, hi = -1e9;
    for (auto& p : pts) {
      const double dx = p.first - mx, dy = p.second - my;
      const double off = -uy * dx + ux * dy, along = ux * dx + uy * dy;
      rms += off * off; lo = std::min(lo, along); hi = std::max(hi, along);
    }
    rms = std::sqrt(rms / pts.size());
    if (rms > 0.03 || hi - lo < 0.3) return std::nullopt;            // not one flat surface
    double nx = uy, ny = -ux;                                          // normal, pointing at the surface
    if (nx * mx + ny * my < 0) { nx = -nx; ny = -ny; }
    return std::atan2(ny, nx);
  }
  // Nearest scan return in any direction: {distance from base_footprint, body angle}. nullopt: no fresh scan.
  std::optional<std::pair<double, double>> nearest_return() {
    std::lock_guard<std::mutex> lk(mtx);
    if (!scan || (node->now() - scan_stamp).seconds() > 0.5) return std::nullopt;
    std::pair<double, double> best{std::numeric_limits<double>::infinity(), 0.0};
    for (size_t k = 0; k < scan->ranges.size(); ++k) {
      const double a = scan->angle_min + k * scan->angle_increment, r = scan->ranges[k];
      if (!std::isfinite(r) || r < scan->range_min || r > scan->range_max) continue;
      const double px = scan_offset + r * std::cos(a), py = r * std::sin(a);
      const double d = std::hypot(px, py);
      if (d < best.first) best = {d, std::atan2(py, px)};
    }
    return best;
  }
  void drive(double vx, double wz, double vy = 0.0) {
    geometry_msgs::msg::TwistStamped t;
    t.header.stamp = node->now();
    t.header.frame_id = "base_footprint";
    t.twist.linear.x = vx;
    t.twist.linear.y = vy;
    t.twist.angular.z = wz;
    approach_cmd->publish(t);
  }
};
using Ctx = std::shared_ptr<Context>;

// ---------------------------------------------------------------- helpers: Nav2 goals as BT steps
// One in-flight Nav2 action (navigate or spin) with a result flag the tree can poll.
template <typename ActionT>
struct PendingGoal {
  typename rclcpp_action::ClientGoalHandle<ActionT>::SharedPtr handle;
  std::shared_ptr<std::optional<rclcpp_action::ResultCode>> result = std::make_shared<std::optional<rclcpp_action::ResultCode>>();
  bool rejected = false;
  std::shared_ptr<bool> rejected_flag = std::make_shared<bool>(false);
  std::shared_ptr<typename rclcpp_action::ClientGoalHandle<ActionT>::SharedPtr> handle_ptr =
      std::make_shared<typename rclcpp_action::ClientGoalHandle<ActionT>::SharedPtr>();

  void send(typename rclcpp_action::Client<ActionT>::SharedPtr client, const typename ActionT::Goal& goal) {
    // Fresh slots for every goal: when a new goal preempts the old one, the old goal's late
    // "aborted/canceled" result lands in the old slots and can't end the new goal.
    result = std::make_shared<std::optional<rclcpp_action::ResultCode>>();
    rejected_flag = std::make_shared<bool>(false);
    handle_ptr = std::make_shared<typename rclcpp_action::ClientGoalHandle<ActionT>::SharedPtr>();
    typename rclcpp_action::Client<ActionT>::SendGoalOptions opts;
    auto res = result;
    auto rej = rejected_flag;
    auto hp = handle_ptr;
    opts.goal_response_callback = [rej, hp](typename rclcpp_action::ClientGoalHandle<ActionT>::SharedPtr gh) {
      if (!gh) *rej = true; else *hp = gh;
    };
    opts.result_callback = [res](const typename rclcpp_action::ClientGoalHandle<ActionT>::WrappedResult& r) {
      *res = r.code;
    };
    client->async_send_goal(goal, opts);
  }
  void cancel(typename rclcpp_action::Client<ActionT>::SharedPtr client) {
    if (*handle_ptr && !result->has_value()) client->async_cancel_goal(*handle_ptr);
  }
  bool done() const { return result->has_value() || *rejected_flag; }
  bool was_rejected() const { return *rejected_flag; }
  bool succeeded() const { return result->has_value() && **result == rclcpp_action::ResultCode::SUCCEEDED; }
};

double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

geometry_msgs::msg::PoseStamped pose_at(double x, double y, double yaw) {
  geometry_msgs::msg::PoseStamped p;
  p.header.frame_id = "map";
  p.pose.position.x = x;
  p.pose.position.y = y;
  p.pose.orientation.z = std::sin(yaw / 2);
  p.pose.orientation.w = std::cos(yaw / 2);
  return p;
}

// Fallback standing spot: the robot's side of the object, `clearance` beyond its edge, facing it.
geometry_msgs::msg::PoseStamped approach_pose(const KnownObject& o, const std::array<double, 3>& robot, double clearance) {
  double dx = robot[0] - o.x, dy = robot[1] - o.y, d = std::hypot(dx, dy);
  if (d < 1e-3) { dx = 1; dy = 0; d = 1; }
  double ux = dx / d, uy = dy / d;
  double hx = std::max(o.sx / 2, 0.02), hy = std::max(o.sy / 2, 0.02);
  double edge = std::min(std::abs(ux) > 1e-6 ? hx / std::abs(ux) : 1e9, std::abs(uy) > 1e-6 ? hy / std::abs(uy) : 1e9);
  double gx = o.x + ux * (edge + clearance), gy = o.y + uy * (edge + clearance);
  return pose_at(gx, gy, std::atan2(o.y - gy, o.x - gx));
}

// Dominant wall direction of an occupancy grid (Manhattan world: kitchens are mostly right angles). Histogram of the
// occupancy-edge gradient directions, folded into [0, 90 deg), weighted by gradient strength; returns the peak.
double estimate_map_axis(const nav_msgs::msg::OccupancyGrid& m) {
  const int W = static_cast<int>(m.info.width), H = static_cast<int>(m.info.height);
  if (W < 5 || H < 5) return 0.0;
  auto occ = [&](int i, int j) { const int v = m.data[j * W + i]; return v >= 65 ? 1.0 : 0.0; };
  std::array<double, 90> hist{};
  for (int j = 1; j < H - 1; ++j)
    for (int i = 1; i < W - 1; ++i) {
      const double gx = (occ(i + 1, j - 1) + 2 * occ(i + 1, j) + occ(i + 1, j + 1)) - (occ(i - 1, j - 1) + 2 * occ(i - 1, j) + occ(i - 1, j + 1));
      const double gy = (occ(i - 1, j + 1) + 2 * occ(i, j + 1) + occ(i + 1, j + 1)) - (occ(i - 1, j - 1) + 2 * occ(i, j - 1) + occ(i + 1, j - 1));
      const double g = std::hypot(gx, gy);
      if (g < 1.0) continue;
      double a = std::atan2(gy, gx);
      a = std::fmod(a + 4 * M_PI, M_PI / 2);                        // fold to [0, 90 deg)
      hist[static_cast<int>(a * 180 / M_PI) % 90] += g;
    }
  int best = 0; double best_v = -1;
  for (int k = 0; k < 90; ++k) {                                    // smooth over +-2 deg (circular)
    double v = 0;
    for (int d = -2; d <= 2; ++d) v += hist[(k + d + 90) % 90];
    if (v > best_v) { best_v = v; best = k; }
  }
  return (best + 0.5) * M_PI / 180;
}

// ---------------------------------------------------------------- reaching a target
// "Go to the fridge" means: stand within arm's reach of it, facing it, on a side you can actually
// stand at. Distances are from the body center (base_footprint) to the surface being faced.
constexpr double REACH_STANDOFF = 0.27;    // body center -> surface. Torso ~0.12 m deep, arms reach ~0.5 m. 0.27 (was 0.30): pose
                                           // estimates run ~5-10 cm long in kitchens, so the true gap lands 0.25-0.38 m
constexpr double STAGING_STANDOFF = 0.65;  // Nav2 stops here first (it keeps its 0.3 m body radius off obstacles)
constexpr double STAGING_MIN = 0.45;       // narrow aisle: pull the staging spot in to here before giving up on a side
constexpr double NAV_CLEAR = 0.32;         // the staging spot needs this much free map around it (Nav2 body radius 0.30)
constexpr double ARM_REACH = 0.65;         // horizontal reach from the body center (shoulder + arm)
// What "within reach" means depends on the thing. Appliances with doors that swing toward the robot (fridge,
// dishwasher, oven) need room for the door; counters, sinks and tables are worked at close. Distances are body
// center -> the surface faced. Appliances are approached at the middle of their front (their door/handle/controls);
// long surfaces (counter runs) at the part nearest the robot.
double standoff_for(const std::string& label) {
  const std::string l = Context::lower(label);
  if (l.find("fridge") != std::string::npos || l.find("refrigerator") != std::string::npos) return 0.45;
  if (l.find("dishwasher") != std::string::npos || l.find("oven") != std::string::npos) return 0.42;
  if (l.find("cabinet") != std::string::npos || l.find("drawer") != std::string::npos) return 0.35;
  return REACH_STANDOFF;
}
bool center_on_front(const std::string& label) {
  const std::string l = Context::lower(label);
  for (const char* k : {"fridge", "refrigerator", "dishwasher", "oven", "stove", "sink", "microwave", "cabinet", "drawer",
                        "coffee", "toaster"})
    if (l.find(k) != std::string::npos) return true;
  return false;
}
constexpr double SQUARE_TOL = 0.10;        // rad (~6 deg): square-on target at the surface
constexpr double SQUARE_ACCEPT = 0.17;     // rad (~10 deg): the most it may be off and still count as there

struct Approach {
  double px, py;      // point on the surface to face
  double nx, ny;      // that surface's outward normal (points at where the robot stands)
  std::string surface;   // what it stands at ("table" when the target is a block on it)
  double reach_dist;     // body center (at the reach spot) -> target, horizontally
  double staging = STAGING_STANDOFF;   // where Nav2 stops: pulled in (>= STAGING_MIN) when a wall is close behind
  double reach = REACH_STANDOFF;       // where the approach stops (per object kind: standoff_for)
  double x(double standoff) const { return px + nx * standoff; }
  double y(double standoff) const { return py + ny * standoff; }
  double yaw() const { return std::atan2(-ny, -nx); }
  geometry_msgs::msg::PoseStamped pose(double standoff) const { return pose_at(x(standoff), y(standoff), yaw()); }
};

// Try the four sides of the object (or of the furniture it sits on) and keep the best side the robot
// can stand at according to its own map: a fridge against a wall with counters beside it has one.
std::optional<Approach> plan_approach(Context& c, const KnownObject& o, const std::array<double, 3>& robot,
                                     const std::optional<std::array<double, 2>>& prefer_normal = std::nullopt) {
  const bool small = std::max(o.sx, o.sy) < 0.35;
  const auto support = small ? c.support_of(o) : std::nullopt;
  const KnownObject& b = support ? *support : o;
  const double hx = std::max(b.sx / 2, 0.02), hy = std::max(b.sy / 2, 0.02);
  struct Side { double nx, ny, half_len, depth; };
  // sides along the room's wall directions (SLAM maps aren't drawn along the walls). The perceived box is
  // axis-aligned in the map frame; project its extents onto the room axes (the step-in below finds the real face).
  const double ca = std::cos(c.map_axis), sa = std::sin(c.map_axis);
  const double dx = std::abs(ca) * hx + std::abs(sa) * hy, dy = std::abs(sa) * hx + std::abs(ca) * hy;
  const Side sides[4] = {{ca, sa, dy, dx}, {-ca, -sa, dy, dx}, {-sa, ca, dx, dy}, {sa, -ca, dx, dy}};
  std::optional<Approach> best;
  double best_score = 1e18;
  for (const auto& sd : sides) {
    const double fcx = b.x + sd.nx * sd.depth, fcy = b.y + sd.ny * sd.depth;   // middle of that side
    const double tx = -sd.ny, ty = sd.nx;                                       // along that side
    // where along the side: the target's own position for small objects and for appliances (middle of the fridge
    // door, the sink, the stove), else (a counter run) the part nearest the robot
    const bool centered = small || center_on_front(b.label);
    const double rx = centered ? o.x : robot[0], ry = centered ? o.y : robot[1];
    // a side up to 1.2 m long: its middle; longer (a counter run): the part nearest the robot
    const double lim = sd.half_len > 0.6 ? sd.half_len - (centered ? 0.15 : 0.3) : 0.0;
    const double s = std::clamp((rx - fcx) * tx + (ry - fcy) * ty, -lim, lim);
    Approach a{fcx + tx * s, fcy + ty * s, sd.nx, sd.ny, b.label, 0.0};
    a.reach = small ? REACH_STANDOFF : standoff_for(b.label);
    // Perceived boxes are loose (a fridge's box sticks out past its doors): step inward from the
    // box side on the robot's map to the real surface. If the map shows only free floor there, it's
    // not a real side. If that part isn't mapped yet (SLAM), keep the side but prefer mapped ones;
    // the plan is redone every second as the map grows.
    double unverified = 0.0;
    if (c.has_map()) {
      const double back = 0.15;    // start a little outside, in case the box is too tight
      auto hit = c.first_solid(a.px + sd.nx * back, a.py + sd.ny * back, -sd.nx, -sd.ny, back + 2 * sd.depth);
      if (hit && *hit < 0.02) {
        // The search started inside something solid: the perceived box is too big or shifted (stereo boxes drift as
        // views merge), so its side lies inside the real object. Walk OUTWARD to the first free cell = the real
        // surface. (Stepping in from there would put the reach spot inside the table and reject the true side.)
        double t = 0.0;
        for (; t <= 0.6; t += 0.025)
          if (c.cell(a.px + sd.nx * (back + t), a.py + sd.ny * (back + t)) == 0) break;
        if (t > 0.6) continue;                                  // solid all the way out: not a side to stand at
        a.px += sd.nx * (back + t); a.py += sd.ny * (back + t);
      } else if (hit) {
        a.px += sd.nx * (back - *hit); a.py += sd.ny * (back - *hit);
      } else if (!small) {
        int known = 0, n = 0;
        for (double t = 0.0; t <= back + 2 * sd.depth; t += 0.05, ++n)
          if (c.cell(a.px + sd.nx * (back - t), a.py + sd.ny * (back - t)) >= 0) ++known;
        if (known * 2 > n) continue;          // mapped, and nothing solid there
        unverified = 3.0;
      }
    }
    a.reach_dist = std::hypot(o.x - a.x(a.reach), o.y - a.y(a.reach));
    // Nav2 needs its body radius free around the staging spot; in a narrow aisle (wall close behind) pull the
    // staging spot in toward the surface rather than send Nav2 a goal it can't plan to.
    bool staging_ok = false;
    for (double st = std::max(STAGING_STANDOFF, a.reach + 0.25); st >= std::max(STAGING_MIN, a.reach + 0.15) - 1e-6; st -= 0.05)
      if (!c.blocked(a.x(st), a.y(st), NAV_CLEAR)) { a.staging = st; staging_ok = true; break; }
    const double mid = (a.reach + a.staging) / 2;
    if (!staging_ok || c.blocked(a.x(a.reach), a.y(a.reach), 0.18) || c.blocked(a.x(mid), a.y(mid), 0.2))
      continue;                                          // wall, counter or clutter there: can't stand on this side
    double score = small ? a.reach_dist : std::hypot(robot[0] - a.x(a.staging), robot[1] - a.y(a.staging));
    // keep the side already chosen unless it became impossible: a slightly better target estimate mid-walk
    // must not send the robot round to the other side of a counter
    if (prefer_normal && sd.nx * (*prefer_normal)[0] + sd.ny * (*prefer_normal)[1] > 0.95)
      score -= 5.0;
    if (small && a.reach_dist > ARM_REACH) score += 10.0;   // prefer any side that puts it in reach
    if (!c.off_map_blocked && c.cell(a.x(a.staging), a.y(a.staging)) != 0) score += 3.0;   // SLAM: prefer seen floor
    score += unverified;
    if (score < best_score) { best = a; best_score = score; }
  }
  return best;
}

// ---------------------------------------------------------------- tree nodes
class SkillBase : public BT::StatefulActionNode {
 public:
  SkillBase(const std::string& name, const BT::NodeConfig& cfg, Ctx ctx) : BT::StatefulActionNode(name, cfg), ctx_(ctx) {}
 protected:
  void mark(const std::string& what) {
    auto idx = getInput<int>("index");
    ctx_->current_index = idx ? idx.value() : -1;
    ctx_->current_skill = what;
    ctx_->status = "running " + what;
  }
  std::string in(const char* key) {
    auto v = getInput<std::string>(key);
    return v ? v.value() : "";
  }
  Ctx ctx_;
};

class NavigateTo : public SkillBase {
 public:
  using SkillBase::SkillBase;
  static BT::PortsList providedPorts() { return {BT::InputPort<std::string>("target"), BT::InputPort<int>("index")}; }
  BT::NodeStatus onStart() override {
    target_ = in("target");
    mark("navigate_to(" + target_ + ")");
    skill_start_ = ctx_->node->now();
    auto obj = ctx_->resolve(target_);
    if (!obj) { ctx_->fail("I haven't seen a " + target_ + " yet, so I don't know where to go"); return BT::NodeStatus::FAILURE; }
    auto robot = ctx_->robot_pose();
    if (!robot) { ctx_->fail("I don't know where I am (no localization)"); return BT::NodeStatus::FAILURE; }
    if (!ctx_->nav->wait_for_action_server(2s)) { ctx_->fail("navigation is not running"); return BT::NodeStatus::FAILURE; }
    plan_ = plan_approach(*ctx_, *obj, *robot);
    auto staging = plan_ ? plan_->pose(plan_->staging) : approach_pose(*obj, *robot, 0.55);
    if (plan_)
      RCLCPP_INFO(ctx_->node->get_logger(), "NavigateTo %s (%s at %.2f, %.2f): facing the %s side at (%.2f, %.2f), "
                  "staging (%.2f, %.2f), reach spot (%.2f, %.2f)", target_.c_str(), obj->id.c_str(), obj->x, obj->y,
                  plan_->surface.c_str(), plan_->px, plan_->py, staging.pose.position.x, staging.pose.position.y,
                  plan_->x(plan_->reach), plan_->y(plan_->reach));
    else
      RCLCPP_WARN(ctx_->node->get_logger(), "NavigateTo %s: no free side on my map, going to the near side without "
                  "the final approach (%.2f, %.2f)", target_.c_str(), staging.pose.position.x, staging.pose.position.y);
    send(staging);
    phase_ = Phase::Travel;
    return BT::NodeStatus::RUNNING;
  }
  BT::NodeStatus onRunning() override {
    if (phase_ != Phase::Travel) return approach();
    // The view of the target sharpens as we get closer: re-plan every second and move the staging
    // spot if it changed by more than 0.4 m.
    if (!goal_.done() && (ctx_->node->now() - last_check_).seconds() > 1.0) {
      last_check_ = ctx_->node->now();
      auto obj = ctx_->resolve(target_);
      auto robot = ctx_->robot_pose();
      if (obj && robot) {
        std::optional<std::array<double, 2>> keep;
        if (plan_) keep = std::array<double, 2>{plan_->nx, plan_->ny};
        auto p = plan_approach(*ctx_, *obj, *robot, keep);
        // No valid side right now (e.g. the perceived box is growing while it is merged): keep the side already
        // chosen. Only fall back to "0.55 m from the centre, toward me" when there never was one - mid-kitchen
        // that spot lands inside the counter run and Nav2 reports the way blocked.
        if (!p && plan_) p = plan_;
        auto updated = p ? p->pose(p->staging) : approach_pose(*obj, *robot, 0.55);
        if (p) plan_ = p;          // small corrections still refine the final approach
        double moved = std::hypot(updated.pose.position.x - current_.pose.position.x,
                                  updated.pose.position.y - current_.pose.position.y);
        // Move the goal only for a real change: > 0.6 m on two checks in a row (a young SLAM map and merging
        // detections nudge the estimate; chasing every nudge made the robot zig-zag).
        move_votes_ = moved > 0.6 ? move_votes_ + 1 : 0;
        if (move_votes_ >= 2) {
          move_votes_ = 0;
          RCLCPP_INFO(ctx_->node->get_logger(), "NavigateTo %s: target estimate improved, staging spot moved %.2f m "
                      "to (%.2f, %.2f)%s", target_.c_str(), moved, updated.pose.position.x, updated.pose.position.y,
                      p ? (" side normal (" + std::to_string(p->nx).substr(0, 4) + ", " + std::to_string(p->ny).substr(0, 4) +
                           ") target " + obj->label + " at (" + std::to_string(obj->x).substr(0, 5) + ", " +
                           std::to_string(obj->y).substr(0, 5) + ") size " + std::to_string(obj->sx).substr(0, 4) + " x " +
                           std::to_string(obj->sy).substr(0, 4)).c_str() : " (no side: fallback)");
          send(updated);           // Nav2 preempts the old goal with the new one
        }
      }
    }
    if (!goal_.done()) return BT::NodeStatus::RUNNING;
    if (!goal_.succeeded()) {
      // Nav2 still starting up (a stereo robot's stack takes longer: the camera renders the world) rejects or
      // instantly aborts goals; that's "not ready", not "blocked". Retry for up to 15 s from the skill's start.
      const double since_start = (ctx_->node->now() - skill_start_).seconds();
      const double since_send = (ctx_->node->now() - last_check_).seconds();
      if (since_start < 15.0 && (goal_.was_rejected() || since_send < 1.5)) {
        if (since_send < 1.0) return BT::NodeStatus::RUNNING;      // wait a moment, then send again
        RCLCPP_INFO(ctx_->node->get_logger(), "NavigateTo %s: Nav2 not ready (goal %s), retrying", target_.c_str(),
                    goal_.was_rejected() ? "rejected" : "aborted at once");
        send(current_);
        return BT::NodeStatus::RUNNING;
      }
      ctx_->fail("I couldn't reach the " + target_ + " (navigation failed or the way is blocked)");
      return BT::NodeStatus::FAILURE;
    }
    if (!plan_) return BT::NodeStatus::SUCCESS;
    phase_ = Phase::Turn;
    approach_start_ = ctx_->node->now();
    seen_gap_.reset();
    yaw_goal_.reset();
    pulse_until_.reset();
    settle_since_.reset();
    ctx_->status = "stepping up to the " + plan_->surface;
    return BT::NodeStatus::RUNNING;
  }
  void onHalted() override {
    goal_.cancel(ctx_->nav);
    if (phase_ != Phase::Travel) ctx_->drive(0.0, 0.0);
  }

 private:
  enum class Phase { Travel, Turn, Drive, Square };
  void send(const geometry_msgs::msg::PoseStamped& pose) {
    NavigateToPose::Goal g;
    g.pose = pose;
    g.pose.header.stamp = ctx_->node->now();
    goal_.send(ctx_->nav, g);
    current_ = pose;
    last_check_ = ctx_->node->now();
  }

  // Final approach, closed-loop on the robot's own pose estimate and lidar: square up to the
  // surface, then walk straight in until it is REACH_STANDOFF away. Commands go to twist_mux's
  // "approach" input (above Nav2, below teleop; the e-stop lock still blocks it).
  // Final approach (after Nav2 stops at the staging spot): square up to the surface while my own sensor can still
  // see it, then walk straight in holding that heading, and stop REACH_STANDOFF from it.
  // Headings and the last few cm are tracked in the ODOM frame (leg odometry + IMU): smooth over half a metre, and
  // unaffected by localization corrections. The map-frame pose (AMCL) can be 10-20 deg off in heading with a
  // forward-only depth camera, and it jumps when AMCL corrects - squaring up to it left the robot visibly crooked.
  // Distances: the sensor's own range to the surface while it sees it; then that range carried forward with odom
  // (a depth camera can't range closer than ~0.3 m). Commands go to twist_mux's "approach" input (above Nav2,
  // below teleop; the e-stop lock still blocks it).
  BT::NodeStatus approach() {
    auto r = ctx_->robot_pose();
    auto od = ctx_->robot_pose("odom");
    if (!r || !od) { ctx_->drive(0, 0); ctx_->fail("I lost track of where I am while stepping up to the " + target_); return BT::NodeStatus::FAILURE; }
    const auto& a = *plan_;
    const double x = (*r)[0], y = (*r)[1], yaw = (*r)[2];
    const double ox = (*od)[0], oy = (*od)[1], oyaw = (*od)[2];
    const double map_err = wrap(a.yaw() - yaw);                    // heading error by the map pose
    auto seen = ctx_->surface_heading_error();                     // by what the sensor sees (flat surface ahead)
    if (seen && std::abs(*seen - map_err) > 0.6) seen.reset();     // that's some other surface
    // heading error to hold: the sensor's when it sees the surface, else the odom heading fixed at the squaring-up
    double err = seen ? *seen : (yaw_goal_ ? wrap(*yaw_goal_ - oyaw) : map_err);
    if (seen) yaw_goal_ = oyaw + *seen;                            // keep the odom target in step with what is seen

    double gap = (x - a.px) * a.nx + (y - a.py) * a.ny;           // body center -> surface, by the map pose
    auto ahead = ctx_->front_range(0.2);
    // Without a flat surface in view, the sensor's range is only trusted when SHORTER than the map's: a level head
    // camera loses a low surface (a 0.75 m table from 1.3 m up) inside ~0.8 m and then ranges something further
    // back, so it can read 0.8 m while the robot is at the table. Shorter readings always win.
    // When the sensor sees a flat surface straight ahead ("seen"), its range IS that surface: trust it either way
    // (localization can be 20-30 cm off along the approach with a forward-only camera).
    // Also trust it when it is within 0.45 m of the map's: localization alone can be ~30 cm off at a fridge (its door
    // doesn't fit as one flat surface, so "seen" fails there); a level camera's blind-spot reading is 0.8 m+ off.
    if (ahead && std::isfinite(*ahead) && std::abs(err) < 0.35 &&
        (*ahead < gap || (seen && *ahead < 1.0) || std::abs(*ahead - gap) < 0.45)) {
      gap = *ahead;
      seen_gap_ = *ahead; seen_x_ = ox; seen_y_ = oy;
    } else if (seen_gap_) {
      const double h = yaw_goal_ ? *yaw_goal_ : oyaw;              // walked toward the surface along h since then
      gap = std::min(gap, *seen_gap_ - ((ox - seen_x_) * std::cos(h) + (oy - seen_y_) * std::sin(h)));
    }
    const double remaining = gap - plan_->reach;
    const double elapsed = (ctx_->node->now() - approach_start_).seconds();
    RCLCPP_INFO_THROTTLE(ctx_->node->get_logger(), *ctx_->node->get_clock(), 1000,
                         "approach %s: phase %d, err %+.0f deg (%s), gap %.2f m (%s), map gap %.2f, raw ahead %.2f",
                         target_.c_str(), static_cast<int>(phase_), err * 180 / M_PI, seen ? "seen" : (yaw_goal_ ? "odom" : "map"),
                         gap, (ahead && std::isfinite(*ahead)) ? "measured" : (seen_gap_ ? "carried" : "map"),
                         (x - a.px) * a.nx + (y - a.py) * a.ny, (ahead && std::isfinite(*ahead)) ? *ahead : -1.0);

    // walking slowly the policy takes ~0.15 s to stop: stop slightly early
    if (phase_ == Phase::Square || (phase_ == Phase::Drive && (remaining <= 0.04 || (ahead && *ahead < 0.22)))) {
      if (phase_ != Phase::Square) {
        phase_ = Phase::Square; square_start_ = ctx_->node->now(); settle_since_.reset(); pulse_until_.reset();
      }
      // "Got there" = within arm's reach AND square-on: aim for 6 deg; accept up to 10 deg after a few seconds of
      // trying (the walking policy can't turn by less than a few degrees); beyond that it isn't there yet.
      auto e = settle_and_measure(map_err, oyaw);
      if (!e) return BT::NodeStatus::RUNNING;
      const double sq_t = (ctx_->node->now() - square_start_).seconds();
      if (std::abs(*e) < SQUARE_TOL || (sq_t > 4.0 && std::abs(*e) < SQUARE_ACCEPT)) return arrived(gap, *e);
      if (sq_t > 14.0) {
        ctx_->drive(0, 0);
        ctx_->fail("I got to the " + target_ + " but couldn't square up to it (" +
                   std::to_string(static_cast<int>(std::abs(*e) * 180 / M_PI)) + " deg off)");
        return BT::NodeStatus::FAILURE;
      }
      pulse(*e);
      return BT::NodeStatus::RUNNING;
    }
    if (elapsed > 20.0) {
      if (remaining < 0.15 && std::abs(err) < SQUARE_ACCEPT) return arrived(gap, err);
      ctx_->drive(0, 0);
      ctx_->fail("I got to the " + target_ + " but couldn't step close enough to reach it (stopped " +
                 std::to_string(static_cast<int>(gap * 100)) + " cm away)");
      return BT::NodeStatus::FAILURE;
    }
    if (phase_ == Phase::Drive && std::abs(err) > 0.5) phase_ = Phase::Turn;   // drifted badly: square up again
    if (phase_ == Phase::Turn) {
      // Square up BEFORE stepping in, while the sensor still sees the surface (a depth camera loses it inside 0.3 m,
      // a level head camera loses a low surface sooner). Turn in short pulses and judge the angle only from a scan
      // captured after stopping: the 5 Hz depth scan lags a few hundred ms, i.e. 8-16 deg at 0.7 rad/s.
      // Start walking within 7 deg, or within 14 deg after 6 s (walking forward corrects small errors, and the
      // surface is squared up again on arrival).
      auto e = settle_and_measure(map_err, oyaw);
      if (!e) return BT::NodeStatus::RUNNING;
      const double turn_t = (ctx_->node->now() - approach_start_).seconds();
      if (std::abs(*e) < 0.12 || (turn_t > 6.0 && std::abs(*e) < 0.25)) {
        phase_ = Phase::Drive;
        yaw_goal_ = oyaw + *e;
        err = *e;
      } else {
        pulse(*e);
        return BT::NodeStatus::RUNNING;
      }
    }
    // walk straight in holding the square-on heading; slow down for the last 10 cm
    ctx_->drive(remaining > 0.10 ? 0.3 : 0.2, std::clamp(1.5 * err, -0.4, 0.4));
    return BT::NodeStatus::RUNNING;
  }
  BT::NodeStatus arrived(double gap, double err) {
    ctx_->drive(0.0, 0.0);
    auto r = ctx_->robot_pose();
    if (r) ctx_->close_up = r;
    RCLCPP_INFO(ctx_->node->get_logger(), "NavigateTo %s: at the %s, %.2f m from it (target %.2f m), %+.0f deg off square, "
                "target %.2f m from my center", target_.c_str(), plan_->surface.c_str(), gap, plan_->reach,
                err * 180.0 / M_PI, plan_->reach_dist);
    return BT::NodeStatus::SUCCESS;
  }

  std::string target_;
  std::optional<Approach> plan_;
  Phase phase_ = Phase::Travel;
  PendingGoal<NavigateToPose> goal_;
  geometry_msgs::msg::PoseStamped current_;
  rclcpp::Time last_check_{0, 0, RCL_ROS_TIME};
  rclcpp::Time approach_start_{0, 0, RCL_ROS_TIME};
  rclcpp::Time skill_start_{0, 0, RCL_ROS_TIME};
  int move_votes_ = 0;
  rclcpp::Time square_start_{0, 0, RCL_ROS_TIME};
  // Pulse-and-settle turning: nullopt while still turning or settling; else the heading error measured AFTER the
  // robot stopped (the surface seen in a scan captured >= 0.15 s after stopping; or, if no flat surface is seen within
  // 1 s, the odom heading target / the map pose).
  std::optional<double> settle_and_measure(double map_err, double oyaw) {
    const auto now = ctx_->node->now();
    if (pulse_until_) {
      if (now < *pulse_until_) { ctx_->drive(0.0, pulse_dir_ * 0.7); return std::nullopt; }
      pulse_until_.reset(); settle_since_ = now;
    }
    if (!settle_since_) settle_since_ = now;
    ctx_->drive(0.0, 0.0);
    auto cap = ctx_->scan_capture_time();
    if (cap && *cap > *settle_since_ + rclcpp::Duration::from_seconds(0.15)) {
      auto seen = ctx_->surface_heading_error();
      if (seen && std::abs(*seen - map_err) < 0.6) { yaw_goal_ = oyaw + *seen; return *seen; }
    }
    if ((now - *settle_since_).seconds() > 1.0) return yaw_goal_ ? wrap(*yaw_goal_ - oyaw) : map_err;
    return std::nullopt;
  }
  // one turn pulse sized to the error (the policy turns ~0.6-0.8 rad/s when asked for 0.7)
  void pulse(double e) {
    pulse_dir_ = e > 0 ? 1.0 : -1.0;
    pulse_until_ = ctx_->node->now() + rclcpp::Duration::from_seconds(std::clamp(std::abs(e) / 0.7 * 0.9, 0.25, 1.2));
    settle_since_.reset();
    ctx_->drive(0.0, pulse_dir_ * 0.7);
  }
  std::optional<rclcpp::Time> pulse_until_, settle_since_;
  double pulse_dir_ = 0.0;
  std::optional<double> yaw_goal_;   // square-on heading in the odom frame, fixed while squaring up
  std::optional<double> seen_gap_;   // last surface distance the depth/lidar measured, and where I was then (odom)
  double seen_x_ = 0.0, seen_y_ = 0.0;
};

// Inserted before every motion skill: make room for Nav2, which keeps a 0.3 m body radius off
// obstacles and refuses to turn or plan when the robot is closer than that. No-op when there is room.
//  1. still standing where navigate_to stepped up to a surface: walk straight back 0.3 m
//     (Nav2 can't plan from there: it thinks the robot is touching the surface)
//  2. before turning in place (turn="true"): a humanoid drifts ~0.15 m while turning, so if anything is
//     within 0.6 m, first walk (Nav2) to the nearest open floor with 0.6 m all round, on the robot's own map
class StepBack : public BT::StatefulActionNode {
 public:
  StepBack(const std::string& name, const BT::NodeConfig& cfg, Ctx ctx) : BT::StatefulActionNode(name, cfg), ctx_(ctx) {}
  static BT::PortsList providedPorts() { return {BT::InputPort<bool>("turn", false, "a turn in place comes next")}; }
  BT::NodeStatus onStart() override {
    auto t = getInput<bool>("turn");
    turn_ = t && t.value();
    auto r = ctx_->robot_pose();
    phase_ = Phase::Open;
    if (!r) return escape();                    // not localized yet: stepping away needs only the lidar
    if (ctx_->close_up) {
      const auto c = *ctx_->close_up;
      ctx_->close_up.reset();
      if (std::hypot((*r)[0] - c[0], (*r)[1] - c[1]) <= 0.2) {        // moved since (e.g. teleop): nothing to undo
        phase_ = Phase::Back;
        start_pose_ = *r;
        start_ = ctx_->node->now();
        ctx_->status = "stepping back from the surface";
        return BT::NodeStatus::RUNNING;
      }
    }
    return escape();
  }
  BT::NodeStatus onRunning() override {
    if (phase_ == Phase::Back) {
      auto r = ctx_->robot_pose();
      const auto& c = start_pose_;
      const double back = r ? -(((*r)[0] - c[0]) * std::cos(c[2]) + ((*r)[1] - c[1]) * std::sin(c[2])) : 0.0;
      auto behind = ctx_->range_toward(M_PI, 0.5);
      if (back >= 0.27 || (ctx_->node->now() - start_).seconds() > 4.0 || (behind && *behind < 0.4)) {
        ctx_->drive(0.0, 0.0);
        return escape();
      }
      ctx_->drive(-0.3, 0.0);
      return BT::NodeStatus::RUNNING;
    }
    if (phase_ == Phase::Escape) {
      auto near = ctx_->nearest_return();
      if (!near || near->first >= ESCAPE_CLEAR + 0.03 || (ctx_->node->now() - start_).seconds() > 3.0) {
        ctx_->drive(0.0, 0.0);
        return open_floor();
      }
      // straight away from the nearest thing: walk or sidestep, whichever is closer to that direction
      const double away = wrap(near->second + M_PI);
      if (std::abs(std::cos(away)) >= std::abs(std::sin(away))) ctx_->drive(std::cos(away) > 0 ? 0.3 : -0.3, 0.0, 0.0);
      else ctx_->drive(0.0, 0.0, std::sin(away) > 0 ? 0.4 : -0.4);
      return BT::NodeStatus::RUNNING;
    }
    if (!nav_.done()) return BT::NodeStatus::RUNNING;
    return BT::NodeStatus::SUCCESS;              // reached open floor, or couldn't: let the next skill try anyway
  }
  void onHalted() override {
    if (phase_ != Phase::Open) ctx_->drive(0.0, 0.0);
    else nav_.cancel(ctx_->nav);
  }

 private:
  enum class Phase { Back, Escape, Open };
  static constexpr double CLEAR = 0.6;          // room needed to turn in place
  static constexpr double ESCAPE_CLEAR = 0.42;  // closer than this, Nav2 may think the robot is touching it
  BT::NodeStatus escape() {
    auto near = ctx_->nearest_return();
    if (!near || near->first >= ESCAPE_CLEAR) return open_floor();
    phase_ = Phase::Escape;
    start_ = ctx_->node->now();
    ctx_->status = "stepping away from an obstacle";
    RCLCPP_INFO(ctx_->node->get_logger(), "StepBack: %.2f m from an obstacle (at %+.0f deg), stepping away",
                near->first, near->second * 180 / M_PI);
    return BT::NodeStatus::RUNNING;
  }
  BT::NodeStatus open_floor() {
    phase_ = Phase::Open;
    if (!turn_) return BT::NodeStatus::SUCCESS;
    auto near = ctx_->nearest_return();
    auto r = ctx_->robot_pose();
    if (!r || !near || near->first >= CLEAR || !ctx_->has_map()) return BT::NodeStatus::SUCCESS;
    for (double rad = 0.4; rad <= 1.4; rad += 0.1)          // nearest open spot beyond Nav2's goal tolerance
      for (int k = 0; k < 16; ++k) {
        const double a = (*r)[2] + k * M_PI / 8;
        const double x = (*r)[0] + rad * std::cos(a), y = (*r)[1] + rad * std::sin(a);
        if (ctx_->blocked(x, y, CLEAR)) continue;
        if (!ctx_->nav->wait_for_action_server(1s)) return BT::NodeStatus::SUCCESS;
        NavigateToPose::Goal g;
        g.pose = pose_at(x, y, (*r)[2]);
        g.pose.header.stamp = ctx_->node->now();
        nav_.send(ctx_->nav, g);
        ctx_->status = "moving to open floor before turning";
        RCLCPP_INFO(ctx_->node->get_logger(), "StepBack: %.2f m from an obstacle, walking %.1f m to open floor (%.2f, %.2f) "
                    "before turning", near->first, rad, x, y);
        return BT::NodeStatus::RUNNING;
      }
    return BT::NodeStatus::SUCCESS;              // nowhere better nearby: try from here
  }
  Ctx ctx_;
  bool turn_ = false;
  Phase phase_ = Phase::Open;
  PendingGoal<NavigateToPose> nav_;
  std::array<double, 3> start_pose_{};
  rclcpp::Time start_{0, 0, RCL_ROS_TIME};
};

class LookAround : public SkillBase {
 public:
  using SkillBase::SkillBase;
  static BT::PortsList providedPorts() { return {BT::InputPort<int>("index")}; }
  BT::NodeStatus onStart() override {
    mark("look_around");
    manual_ = false;
    if (!ctx_->spin->wait_for_action_server(2s)) { ctx_->fail("the spin behavior is not running"); return BT::NodeStatus::FAILURE; }
    Spin::Goal g;
    g.target_yaw = 6.2;
    g.time_allowance.sec = 20;
    goal_.send(ctx_->spin, g);
    return BT::NodeStatus::RUNNING;
  }
  BT::NodeStatus onRunning() override {
    if (manual_) return turn_myself();
    if (!goal_.done()) return BT::NodeStatus::RUNNING;
    if (goal_.succeeded()) return BT::NodeStatus::SUCCESS;
    // Nav2's spin refuses whenever its costmap puts anything within the body radius (a counter 0.4-0.5 m away
    // plus the walk's sway is enough). A humanoid turning in place sweeps nothing, so turn myself instead and
    // stop only if something is really touching close (< TOUCH m on the scan).
    auto r = ctx_->robot_pose();
    if (!r) { ctx_->fail("I couldn't turn around to look (I don't know where I am)"); return BT::NodeStatus::FAILURE; }
    RCLCPP_INFO(ctx_->node->get_logger(), "LookAround: Nav2's spin refused (costmap too close); turning in place myself");
    manual_ = true; turned_ = 0.0; last_yaw_ = (*r)[2]; start_ = ctx_->node->now();
    return turn_myself();
  }
  void onHalted() override { if (manual_) ctx_->drive(0, 0); else goal_.cancel(ctx_->spin); manual_ = false; }
 private:
  static constexpr double TOUCH = 0.30;
  BT::NodeStatus turn_myself() {
    auto r = ctx_->robot_pose();
    auto near = ctx_->nearest_return();
    if (near && near->first < TOUCH) {
      ctx_->drive(0, 0); manual_ = false;
      ctx_->fail("I couldn't turn around to look (something is touching close, " +
                 std::to_string(static_cast<int>(near->first * 100)) + " cm away)");
      return BT::NodeStatus::FAILURE;
    }
    if (r) { turned_ += std::abs(wrap((*r)[2] - last_yaw_)); last_yaw_ = (*r)[2]; }
    if (turned_ >= 6.2 || (ctx_->node->now() - start_).seconds() > 20.0) {
      ctx_->drive(0, 0); manual_ = false;
      return turned_ >= 5.5 ? BT::NodeStatus::SUCCESS
                            : (ctx_->fail("I couldn't turn all the way around to look"), BT::NodeStatus::FAILURE);
    }
    ctx_->drive(0.0, 0.8);            // above the policy's turn deadband
    return BT::NodeStatus::RUNNING;
  }
  PendingGoal<Spin> goal_;
  bool manual_ = false;
  double turned_ = 0.0, last_yaw_ = 0.0;
  rclcpp::Time start_{0, 0, RCL_ROS_TIME};
};

class FindObject : public SkillBase {
 public:
  using SkillBase::SkillBase;
  static BT::PortsList providedPorts() { return {BT::InputPort<std::string>("target"), BT::InputPort<int>("index")}; }
  BT::NodeStatus onStart() override {
    target_ = in("target");
    mark("find_object(" + target_ + ")");
    if (ctx_->resolve(target_)) return BT::NodeStatus::SUCCESS;
    if (!ctx_->spin->wait_for_action_server(2s)) { ctx_->fail("I can't see a " + target_); return BT::NodeStatus::FAILURE; }
    Spin::Goal g;                                  // not in memory yet: look around once
    g.target_yaw = 6.2;
    g.time_allowance.sec = 20;
    goal_.send(ctx_->spin, g);
    return BT::NodeStatus::RUNNING;
  }
  BT::NodeStatus onRunning() override {
    if (ctx_->resolve(target_)) { goal_.cancel(ctx_->spin); return BT::NodeStatus::SUCCESS; }
    if (!goal_.done()) return BT::NodeStatus::RUNNING;
    ctx_->fail("I looked around but I can't see a " + target_ + " from here");
    return BT::NodeStatus::FAILURE;
  }
  void onHalted() override { goal_.cancel(ctx_->spin); }
 private:
  std::string target_;
  PendingGoal<Spin> goal_;
};

class SearchFor : public SkillBase {
 public:
  using SkillBase::SkillBase;
  static BT::PortsList providedPorts() {
    return {BT::InputPort<std::string>("target"), BT::InputPort<std::string>("hints"), BT::InputPort<int>("index")};
  }
  BT::NodeStatus onStart() override {
    target_ = in("target");
    mark("search_for(" + target_ + ")");
    hints_.clear();
    std::stringstream ss(in("hints"));
    for (std::string h; std::getline(ss, h, ';');) if (!h.empty()) hints_.push_back(h);
    step_ = 0;
    phase_ = Phase::Next;
    visited_.clear();
    return ctx_->resolve(target_) ? BT::NodeStatus::SUCCESS : BT::NodeStatus::RUNNING;
  }
  BT::NodeStatus onRunning() override {
    if (ctx_->resolve(target_)) { nav_.cancel(ctx_->nav); spin_.cancel(ctx_->spin); return BT::NodeStatus::SUCCESS; }
    switch (phase_) {
      case Phase::Next: {
        if (step_ == 0) {                            // first: look around right here
          step_ = 1;
          return start_spin();
        }
        while (!hints_.empty()) {                    // then: each hinted place we know, in order
          std::string h = hints_.front();
          hints_.erase(hints_.begin());
          auto obj = ctx_->resolve(h);
          auto robot = ctx_->robot_pose();
          if (!obj || !robot) continue;
          visited_.push_back(h);
          NavigateToPose::Goal g;
          auto side = plan_approach(*ctx_, *obj, *robot);
          g.pose = side ? side->pose(side->staging) : approach_pose(*obj, *robot, 0.6);
          g.pose.header.stamp = ctx_->node->now();
          ctx_->status = "searching for " + target_ + ": going to the " + h;
          nav_.send(ctx_->nav, g);
          phase_ = Phase::Walking;
          return BT::NodeStatus::RUNNING;
        }
        std::string where = visited_.empty() ? "around me" : "around me and near the ";
        for (size_t i = 0; i < visited_.size(); ++i) where += (i ? ", " : "") + visited_[i];
        ctx_->fail("I searched " + where + " but didn't find a " + target_);
        return BT::NodeStatus::FAILURE;
      }
      case Phase::Walking:
        if (!nav_.done()) return BT::NodeStatus::RUNNING;
        return start_spin();
      case Phase::Looking:
        if (!spin_.done()) return BT::NodeStatus::RUNNING;
        phase_ = Phase::Next;
        return BT::NodeStatus::RUNNING;
    }
    return BT::NodeStatus::FAILURE;
  }
  void onHalted() override { nav_.cancel(ctx_->nav); spin_.cancel(ctx_->spin); }
 private:
  enum class Phase { Next, Walking, Looking };
  BT::NodeStatus start_spin() {
    Spin::Goal g;
    g.target_yaw = 6.2;
    g.time_allowance.sec = 20;
    spin_.send(ctx_->spin, g);
    phase_ = Phase::Looking;
    return BT::NodeStatus::RUNNING;
  }
  std::string target_;
  std::vector<std::string> hints_, visited_;
  int step_ = 0;
  Phase phase_ = Phase::Next;
  PendingGoal<NavigateToPose> nav_;
  PendingGoal<Spin> spin_;
};

// Hand skill: one VLA goal on /skills/manipulate (hsim_manipulation). The robot must already stand at the surface
// (navigate_to brings it to arm's reach, facing it). pick(X) followed by place(Y) in a plan becomes ONE goal
// "pick up the X and put it on the Y" (to_xml merges them): the v1 policy is trained on whole pick-and-place.
class Manipulate : public SkillBase {
 public:
  using SkillBase::SkillBase;
  static BT::PortsList providedPorts() {
    return {BT::InputPort<std::string>("instruction"), BT::InputPort<std::string>("object"),
            BT::InputPort<std::string>("target"), BT::InputPort<int>("index")};
  }
  BT::NodeStatus onStart() override {
    const std::string instr = in("instruction");
    mark("manipulate(" + instr + ")");
    if (!ctx_->manip->wait_for_action_server(1s)) {
      ctx_->fail("my manipulation skill isn't running (start hsim_manipulation and the VLA server)");
      return BT::NodeStatus::FAILURE;
    }
    ManipulateAction::Goal g;
    g.instruction = instr;
    g.object = in("object");
    g.target = in("target");
    done_ = std::make_shared<std::optional<std::pair<bool, std::string>>>();
    handle_ = std::make_shared<rclcpp_action::ClientGoalHandle<ManipulateAction>::SharedPtr>();
    rclcpp_action::Client<ManipulateAction>::SendGoalOptions opts;
    auto done = done_;
    auto hp = handle_;
    opts.goal_response_callback = [done, hp](rclcpp_action::ClientGoalHandle<ManipulateAction>::SharedPtr gh) {
      if (!gh) *done = std::make_pair(false, std::string("my manipulation skill is busy"));
      else *hp = gh;
    };
    opts.result_callback = [done](const rclcpp_action::ClientGoalHandle<ManipulateAction>::WrappedResult& r) {
      const bool ok = r.code == rclcpp_action::ResultCode::SUCCEEDED && r.result && r.result->success;
      *done = std::make_pair(ok, r.result ? r.result->message : std::string("no result"));
    };
    ctx_->manip->async_send_goal(g, opts);
    return BT::NodeStatus::RUNNING;
  }
  BT::NodeStatus onRunning() override {
    if (!done_->has_value()) return BT::NodeStatus::RUNNING;
    if ((*done_)->first) return BT::NodeStatus::SUCCESS;
    ctx_->fail((*done_)->second.empty() ? "the hand skill failed" : (*done_)->second);
    return BT::NodeStatus::FAILURE;
  }
  void onHalted() override {
    if (*handle_ && !done_->has_value()) ctx_->manip->async_cancel_goal(*handle_);
  }
 private:
  std::shared_ptr<std::optional<std::pair<bool, std::string>>> done_ = std::make_shared<std::optional<std::pair<bool, std::string>>>();
  std::shared_ptr<rclcpp_action::ClientGoalHandle<ManipulateAction>::SharedPtr> handle_ =
      std::make_shared<rclcpp_action::ClientGoalHandle<ManipulateAction>::SharedPtr>();
};

class NotYetAvailable : public SkillBase {     // kept for skills that have no implementation
 public:
  using SkillBase::SkillBase;
  static BT::PortsList providedPorts() { return {BT::InputPort<std::string>("target"), BT::InputPort<int>("index")}; }
  BT::NodeStatus onStart() override {
    mark(name() + "(" + in("target") + ")");
    ctx_->fail("I can't " + Context::lower(name()) + " things yet: my manipulation skills aren't installed (Phase 5)");
    return BT::NodeStatus::FAILURE;
  }
  BT::NodeStatus onRunning() override { return BT::NodeStatus::FAILURE; }
  void onHalted() override {}
};

class Say : public BT::SyncActionNode {
 public:
  Say(const std::string& name, const BT::NodeConfig& cfg, Ctx ctx) : BT::SyncActionNode(name, cfg), ctx_(ctx) {}
  static BT::PortsList providedPorts() { return {BT::InputPort<std::string>("text"), BT::InputPort<int>("index")}; }
  BT::NodeStatus tick() override {
    auto t = getInput<std::string>("text");
    std_msgs::msg::String m;
    m.data = t ? t.value() : "";
    ctx_->say->publish(m);
    RCLCPP_INFO(ctx_->node->get_logger(), "SAY: %s", m.data.c_str());
    return BT::NodeStatus::SUCCESS;
  }
 private:
  Ctx ctx_;
};

class IsSafe : public BT::ConditionNode {
 public:
  IsSafe(const std::string& name, const BT::NodeConfig& cfg, Ctx ctx) : BT::ConditionNode(name, cfg), ctx_(ctx) {}
  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus tick() override {
    std::lock_guard<std::mutex> lk(ctx_->mtx);
    double age = (ctx_->node->now() - ctx_->safety_stamp).seconds();
    if (ctx_->safety_level < 0 || age > 2.0) { ctx_->failure = "stopped: no safety status from the safety monitor"; return BT::NodeStatus::FAILURE; }
    if (ctx_->safety_level >= diagnostic_msgs::msg::DiagnosticStatus::ERROR) {
      ctx_->failure = "stopped for safety: " + ctx_->safety_message;
      return BT::NodeStatus::FAILURE;
    }
    return BT::NodeStatus::SUCCESS;
  }
 private:
  Ctx ctx_;
};

// ---------------------------------------------------------------- the executor node
class BehaviorNode {
 public:
  explicit BehaviorNode(rclcpp::Node::SharedPtr node) : node_(node), ctx_(std::make_shared<Context>()) {
    ctx_->off_map_blocked = node_->declare_parameter<bool>("off_map_blocked", true);
    ctx_->node = node;
    ctx_->nav = rclcpp_action::create_client<NavigateToPose>(node, "/navigate_to_pose");
    ctx_->manip = rclcpp_action::create_client<ManipulateAction>(node, "/skills/manipulate");
    ctx_->spin = rclcpp_action::create_client<Spin>(node, "/spin");
    ctx_->say = node->create_publisher<std_msgs::msg::String>("/robot/say", 10);
    ctx_->approach_cmd = node->create_publisher<geometry_msgs::msg::TwistStamped>("/cmd_vel_approach", 10);
    ctx_->tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*ctx_->tf);

    objects_sub_ = node->create_subscription<vision_msgs::msg::Detection3DArray>(
        "/world/objects", 5, [this](vision_msgs::msg::Detection3DArray::SharedPtr m) {
          std::vector<KnownObject> objs;
          for (const auto& d : m->detections) {
            if (d.results.empty()) continue;
            const auto& c = d.bbox.center.position;
            objs.push_back({d.id, d.results[0].hypothesis.class_id, c.x, c.y, c.z, d.bbox.size.x, d.bbox.size.y,
                            d.bbox.size.z, d.results[0].hypothesis.score});
          }
          std::lock_guard<std::mutex> lk(ctx_->mtx);
          ctx_->objects = std::move(objs);
        });
    map_sub_ = node->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/map", rclcpp::QoS(1).transient_local().reliable(), [this](nav_msgs::msg::OccupancyGrid::SharedPtr m) {
          std::lock_guard<std::mutex> lk(ctx_->mtx);
          ctx_->map = m;
          ctx_->map_axis = ctx_->off_map_blocked ? 0.0 : estimate_map_axis(*m);   // prebuilt maps are wall-aligned
        });
    scan_sub_ = node->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::LaserScan::SharedPtr m) {
          double offset = 0.0;                       // where the scan origin sits on the body
          try {
            offset = ctx_->tf->lookupTransform("base_footprint", m->header.frame_id, tf2::TimePointZero).transform.translation.x;
          } catch (...) {}
          std::lock_guard<std::mutex> lk(ctx_->mtx);
          ctx_->scan = m;
          ctx_->scan_offset = offset;
          ctx_->scan_stamp = ctx_->node->now();
        });
    safety_sub_ = node->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
        "/safety/status", 5, [this](diagnostic_msgs::msg::DiagnosticArray::SharedPtr m) {
          if (m->status.empty()) return;
          std::lock_guard<std::mutex> lk(ctx_->mtx);
          ctx_->safety_level = m->status[0].level;
          ctx_->safety_message = m->status[0].message;
          ctx_->safety_stamp = ctx_->node->now();
        });

    factory_.registerNodeType<NavigateTo>("NavigateTo", ctx_);
    factory_.registerNodeType<LookAround>("LookAround", ctx_);
    factory_.registerNodeType<StepBack>("StepBack", ctx_);
    factory_.registerNodeType<FindObject>("FindObject", ctx_);
    factory_.registerNodeType<SearchFor>("SearchFor", ctx_);
    factory_.registerNodeType<Manipulate>("Manipulate", ctx_);
    factory_.registerNodeType<Say>("Say", ctx_);
    factory_.registerNodeType<IsSafe>("IsSafe", ctx_);

    server_ = rclcpp_action::create_server<ExecutePlan>(
        node, "/behavior/execute_plan",
        [](const rclcpp_action::GoalUUID&, std::shared_ptr<const ExecutePlan::Goal>) {
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [](std::shared_ptr<rclcpp_action::ServerGoalHandle<ExecutePlan>>) { return rclcpp_action::CancelResponse::ACCEPT; },
        [this](std::shared_ptr<rclcpp_action::ServerGoalHandle<ExecutePlan>> gh) { start(gh); });
    timer_ = node->create_wall_timer(50ms, [this] { tick(); });
    RCLCPP_INFO(node->get_logger(), "behavior tree executor ready on /behavior/execute_plan (Groot2 port 1667)");
  }

 private:
  static std::string escape(const std::string& s) {
    std::string o;
    for (char c : s) {
      switch (c) {
        case '&': o += "&amp;"; break;
        case '<': o += "&lt;"; break;
        case '>': o += "&gt;"; break;
        case '"': o += "&quot;"; break;
        default: o += c;
      }
    }
    return o;
  }

  std::string to_xml(const hsim_interfaces::msg::SkillPlan& plan, std::string& error) {
    std::ostringstream x;
    x << "<root BTCPP_format=\"4\"><BehaviorTree ID=\"Plan\"><ReactiveSequence><IsSafe/><Sequence>";
    int i = 0;
    for (size_t k = 0; k < plan.skills.size(); ++k) {
      const auto& s = plan.skills[k];
      const std::string idx = " index=\"" + std::to_string(i++) + "\"";
      const std::string tgt = " target=\"" + escape(s.target) + "\"";
      // motion skills first back off from a surface the robot stepped up to (no-op otherwise)
      if (s.name == "navigate_to") x << "<StepBack/>";
      else if (s.name == "find_object" || s.name == "search_for" || s.name == "look_around") x << "<StepBack turn=\"true\"/>";
      if (s.name == "navigate_to") x << "<NavigateTo" << tgt << idx << "/>";
      else if (s.name == "find_object") x << "<FindObject" << tgt << idx << "/>";
      else if (s.name == "search_for") {
        std::string hints;
        for (size_t k = 0; k < s.hints.size(); ++k) hints += (k ? ";" : "") + s.hints[k];
        x << "<SearchFor" << tgt << " hints=\"" << escape(hints) << "\"" << idx << "/>";
      } else if (s.name == "look_around") x << "<LookAround" << idx << "/>";
      else if (s.name == "pick" && k + 1 < plan.skills.size() && plan.skills[k + 1].name == "place") {
        // pick(X) + place(Y) -> one hand-skill goal (the policy is trained on whole pick-and-place)
        const auto& p = plan.skills[k + 1];
        x << "<Manipulate instruction=\"" << escape("pick up the " + s.target + " and put it on the " + p.target)
          << "\" object=\"" << escape(s.target) << "\" target=\"" << escape(p.target) << "\"" << idx << "/>";
        ++k;
        ++i;
      } else if (s.name == "pick") {
        x << "<Manipulate instruction=\"" << escape("pick up the " + s.target) << "\" object=\"" << escape(s.target) << "\"" << idx << "/>";
      } else if (s.name == "place") {
        x << "<Manipulate instruction=\"" << escape("put it on the " + s.target) << "\" target=\"" << escape(s.target) << "\"" << idx << "/>";
      }
      else if (s.name == "say") x << "<Say text=\"" << escape(s.text) << "\"" << idx << "/>";
      else { error = "unknown skill '" + s.name + "'"; return ""; }
    }
    x << "</Sequence></ReactiveSequence></BehaviorTree></root>";
    return x.str();
  }

  void finish(bool success, const std::string& message) {
    if (!active_) return;
    auto r = std::make_shared<ExecutePlan::Result>();
    r->success = success;
    r->failed_skill_index = success ? -1 : ctx_->current_index;
    r->message = message;
    if (active_->is_canceling()) active_->canceled(r);
    else if (success) active_->succeed(r);
    else active_->abort(r);
    RCLCPP_INFO(node_->get_logger(), "plan %s: %s", success ? "SUCCEEDED" : "FAILED", message.c_str());
    active_.reset();
    groot_.reset();
    tree_.reset();
  }

  void start(std::shared_ptr<rclcpp_action::ServerGoalHandle<ExecutePlan>> gh) {
    if (active_) {                                   // a new command preempts the running plan
      if (tree_) tree_->haltTree();
      finish(false, "replaced by a new command");
    }
    std::string err;
    const auto& plan = gh->get_goal()->plan;
    const std::string xml = to_xml(plan, err);
    active_ = gh;
    ctx_->failure.clear();
    ctx_->current_index = -1;
    if (xml.empty()) { finish(false, err); return; }
    try {
      tree_ = std::make_unique<BT::Tree>(factory_.createTreeFromText(xml));
      groot_ = std::make_unique<BT::Groot2Publisher>(*tree_, 1667);
    } catch (const std::exception& e) {
      finish(false, std::string("could not build the behavior tree: ") + e.what());
      return;
    }
    RCLCPP_INFO(node_->get_logger(), "running plan for \"%s\" (%zu skills)", plan.task.c_str(), plan.skills.size());
  }

  void tick() {
    if (!active_ || !tree_) return;
    if (active_->is_canceling()) {
      tree_->haltTree();
      finish(false, "cancelled");
      return;
    }
    BT::NodeStatus s = tree_->tickOnce();
    auto fb = std::make_shared<ExecutePlan::Feedback>();
    fb->current_skill_index = ctx_->current_index;
    fb->current_skill = ctx_->current_skill;
    fb->status = ctx_->status;
    active_->publish_feedback(fb);
    if (s == BT::NodeStatus::SUCCESS) finish(true, "done");
    else if (s == BT::NodeStatus::FAILURE) finish(false, ctx_->failure.empty() ? "a step failed" : ctx_->failure);
  }

  rclcpp::Node::SharedPtr node_;
  Ctx ctx_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr objects_sub_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr safety_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  BT::BehaviorTreeFactory factory_;
  std::unique_ptr<BT::Tree> tree_;
  std::unique_ptr<BT::Groot2Publisher> groot_;
  rclcpp_action::Server<ExecutePlan>::SharedPtr server_;
  std::shared_ptr<rclcpp_action::ServerGoalHandle<ExecutePlan>> active_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("behavior");
  BehaviorNode bt(node);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
