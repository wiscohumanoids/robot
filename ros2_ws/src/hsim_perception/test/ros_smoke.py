"""ROS smoke test for perception_client (runs inside the ROS container, no sim, no GPU).

  docker run --rm -v "$PWD:/src:ro" hsim/ros-jazzy:dev bash /src/ros2_ws/src/hsim_perception/test/run_ros_smoke.sh

Plays the camera (synthetic ZED X frames + camera_info + TF map->head_camera_optical), the planner
(/planner/status asking for a mug) and the perception server (a tiny HTTP server that checks the request
and answers with one "mug" detection). Checks:
  1. the node sends a valid request (images, intrinsics, camera pose, the mug in the prompts)
  2. /perception/detections carries the mug at the right place (map position via the published TF)
  3. world_model with multi-view confirmation (confirm_views 2): the mug (seen every frame) is committed after
     >= 2 views; a phantom "plate" (only in the first response) never reaches /world/summary and expires
  4. server stopped -> /perception/status mode "fallback";  server back -> mode "server" again
"""
import json
import math
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster
from vision_msgs.msg import Detection3DArray

PORT = 18100
W, H = 640, 320
FX = (W / 2) / math.tan(math.radians(110) / 2)
CAM_T = [1.0, 2.0, 1.25]                                   # camera in the map; looks along map +x
R_MC = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
MUG_CAM = [0.1, 0.05, 1.5]                                 # what the fake server reports (camera frame)
seen_requests = []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send({"status": "ok", "backend": "fake-smoke"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen_requests.append(req)
        c_map = (R_MC @ np.array(MUG_CAM) + np.array(CAM_T)).tolist()
        dets = [{"label": "mug", "score": 0.8, "box_xyxy": [300, 150, 340, 190], "n_points": 120,
                 "centroid_cam": MUG_CAM, "centroid_cam_completed": MUG_CAM, "extent_cam": [0.08, 0.1, 0.08],
                 "centroid_map": c_map, "extent_map": [0.09, 0.08, 0.1]}]
        if len(seen_requests) == 1:                       # a one-frame hallucination
            ph = [-0.4, 0.1, 1.8]
            dets.append({"label": "plate", "score": 0.7, "box_xyxy": [100, 160, 160, 190], "n_points": 90,
                         "centroid_cam": ph, "centroid_cam_completed": ph, "extent_cam": [0.25, 0.02, 0.25],
                         "centroid_map": (R_MC @ np.array(ph) + np.array(CAM_T)).tolist(), "extent_map": [0.25, 0.25, 0.02]})
        self._send({"detections": dets, "timing_ms": {"total": 5.0}, "backend": "fake-smoke", "stamp": req.get("stamp")})


def start_server():
    srv = HTTPServer(("127.0.0.1", PORT), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class Harness(Node):
    def __init__(self):
        super().__init__("smoke_harness")
        self.p_rgb = self.create_publisher(Image, "/camera/head/color/image_raw", 2)
        self.p_dep = self.create_publisher(Image, "/camera/head/depth/image_raw", 2)
        self.p_ci = self.create_publisher(CameraInfo, "/camera/head/color/camera_info", 2)
        self.p_plan = self.create_publisher(String, "/planner/status", 10)
        self.dets, self.status, self.summary, self.max_tentative = [], {}, {}, 0
        self.create_subscription(String, "/world/summary", self.on_summary, 10)
        self.create_subscription(Detection3DArray, "/perception/detections", lambda m: self.dets.append(m), 10)
        self.create_subscription(String, "/perception/status", lambda m: self.status.update(json.loads(m.data)), 10)
        tfb = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = "map", "head_camera_optical"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = CAM_T
        q = quat(R_MC)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        tfb.sendTransform(t)
        self.create_timer(0.2, self.frame)

    def frame(self):
        stamp = self.get_clock().now().to_msg()
        rgb = np.full((H, W, 3), 120, np.uint8)
        dep = np.full((H, W), 2.0, np.float32)
        for pub, data, enc, step in ((self.p_rgb, rgb, "rgb8", W * 3), (self.p_dep, dep, "32FC1", W * 4)):
            m = Image(); m.header.stamp, m.header.frame_id = stamp, "head_camera_optical"
            m.height, m.width, m.encoding, m.step = H, W, enc, step; m.data = data.tobytes(); pub.publish(m)
        ci = CameraInfo(); ci.header.stamp, ci.header.frame_id = stamp, "head_camera_optical"
        ci.width, ci.height = W, H; ci.k = [FX, 0.0, (W - 1) / 2, 0.0, FX, (H - 1) / 2, 0.0, 0.0, 1.0]
        self.p_ci.publish(ci)

    def on_summary(self, m):
        self.summary = json.loads(m.data)
        self.max_tentative = max(self.max_tentative, self.summary.get("tentative", 0))

    def plan(self):
        self.p_plan.publish(String(data=json.dumps({"state": "executing", "plan": [
            {"name": "find_object", "target": "teapot", "hints": ["counter"], "text": ""}]})))


def quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return ((R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w)


def quat_to_mat(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def spin_until(node, cond, timeout):
    end = time.time() + timeout
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
        if cond():
            return True
    return False


def main(world, objects):
    srv = start_server()
    client = subprocess.Popen(["ros2", "run", "hsim_perception", "perception_client", "--ros-args",
                               "-p", f"world:={world}", "-p", f"objects:={objects}",
                               "-p", f"server_url:=http://127.0.0.1:{PORT}", "-p", "retry_s:=2.0", "-p", "timeout_s:=1.0"])
    wm = subprocess.Popen(["ros2", "run", "hsim_perception", "world_model", "--ros-args",
                           "-p", "confirm_views:=2", "-p", "tentative_timeout_s:=3.0"])
    rclpy.init()
    h = Harness()
    ok = True
    try:
        spin_until(h, lambda: False, 3.0)
        h.create_timer(1.0, h.plan)                        # the planner republishes its status on changes; repeat here
        got = spin_until(h, lambda: any(len(m.detections) for m in h.dets), 20.0)
        spin_until(h, lambda: any("teapot" in r.get("prompts", []) for r in seen_requests), 10.0)
        print("1. detection received:", got); ok &= got
        req = seen_requests[-1] if seen_requests else {}
        good_req = bool(req) and all(k in req for k in ("rgb_jpeg_b64", "depth_png16_b64", "intrinsics", "T_map_cam")) \
            and abs(req["intrinsics"]["fx"] - FX) < 1e-6
        plan_ok = "teapot" in req.get("prompts", []) and "teapot" in (req.get("priority_prompts") or [])
        print("   request valid (images, intrinsics, camera pose):", good_req, "| prompts:", len(req.get("prompts", [])))
        print("   planner target 'teapot' added to prompts + priority:", plan_ok, req.get("priority_prompts")); ok &= good_req and plan_ok
        T = np.array(req.get("T_map_cam", np.eye(4).ravel().tolist())).reshape(4, 4)
        pose_ok = np.allclose(T[:3, 3], CAM_T, atol=1e-6) and np.allclose(T[:3, :3], R_MC, atol=1e-6)
        print("   camera pose from TF correct:", pose_ok); ok &= pose_ok
        d = next(m for m in h.dets if len(m.detections)).detections[0]
        c = np.array([d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z])
        q = d.bbox.center.orientation
        R_box_map = R_MC @ quat_to_mat(q.x, q.y, q.z, q.w)
        mug_map = R_MC @ c + np.array(CAM_T)
        place_ok = d.results[0].hypothesis.class_id == "mug" and np.allclose(mug_map, R_MC @ np.array(MUG_CAM) + CAM_T, atol=1e-4) \
            and np.allclose(R_box_map, np.eye(3), atol=1e-4) and abs(d.bbox.size.x - 0.09) < 1e-6
        print(f"2. mug in map at {np.round(mug_map, 3).tolist()}, box map-aligned, frame {d.header.frame_id}:", place_ok); ok &= place_ok
        labels = lambda: [o["label"] for o in h.summary.get("objects", [])]
        mug = spin_until(h, lambda: "mug" in labels(), 15.0)
        views = next((o["views"] for o in h.summary.get("objects", []) if o["label"] == "mug"), 0)
        spin_until(h, lambda: False, 4.0)                  # let the phantom's tentative track time out (3 s)
        conf_ok = mug and views >= 2 and "plate" not in labels() and h.max_tentative >= 1 and h.summary.get("tentative") == 0
        print(f"3. memory: mug committed after {views} views: {mug}; phantom plate never committed: {'plate' not in labels()}; "
              f"tentative seen {h.max_tentative} -> now {h.summary.get('tentative')}:", conf_ok); ok &= conf_ok
        srv.shutdown(); srv.server_close()
        fb = spin_until(h, lambda: h.status.get("mode") == "fallback", 20.0)
        print("4. server stopped -> fallback to stub:", fb, h.status.get("last_error", "")[:80]); ok &= fb
        srv = start_server()
        back = spin_until(h, lambda: h.status.get("mode") == "server", 20.0)
        print("   server back -> real detections again:", back); ok &= back
    finally:
        client.terminate(); client.wait(10)
        wm.terminate(); wm.wait(10)
        srv.shutdown()
        rclpy.try_shutdown()
    print("SMOKE TEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
