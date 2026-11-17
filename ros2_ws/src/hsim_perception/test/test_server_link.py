"""Tests for the perception client's ROS-free logic (hsim_perception/server_link.py).

Mac:  <venv with numpy, opencv-python-headless, fastapi, uvicorn>/bin/python -m pytest ros2_ws/src/hsim_perception/test -q
The HTTP test starts the real perception server (fake backend) on a free local port and sends exactly
the request the ROS node builds.
"""
import json
import math
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "hsim_perception"))
sys.path.insert(0, str(REPO / "services" / "perception_server"))
from hsim_perception import server_link as sl  # noqa: E402

W, H = 640, 320
FX = (W / 2) / math.tan(math.radians(110) / 2)
K = {"fx": FX, "fy": FX, "cx": (W - 1) / 2, "cy": (H - 1) / 2, "width": W, "height": H}
R_OPT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)      # optical axes in a map where the camera looks along +x


def quat_to_mat(x, y, z, w):                                        # same as world_model.py
    """Rotation matrix from a quaternion (x, y, z, w); the same as world_model.py."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_vocab_and_planner_targets():
    objs = {"objects": [{"label": "sugar cube"}, {"label": "wall"}, {"label": "fridge"},
                        {"label": "milk", "detectable": False}, {"label": "fridge"}]}
    assert sl.env_vocab(objs) == ["sugar cube", "fridge"]
    status = json.dumps({"state": "executing", "plan": [
        {"name": "search_for", "target": "coffee_mug", "hints": ["counter", "table"], "text": ""},
        {"name": "say", "target": "", "hints": [], "text": "found it"}]})
    assert sl.planner_targets(status) == ["coffee mug", "counter", "table"]
    assert sl.planner_targets("not json") == []


@pytest.mark.parametrize("R", [R_OPT, R_OPT @ np.array([[1, 0, 0], [0, math.cos(0.4), -math.sin(0.4)], [0, math.sin(0.4), math.cos(0.4)]])])
def test_boxes_match_world_model_semantics(R):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [1.0, 2.0, 1.25]
    c_map = np.array([2.6, 2.1, 0.95])
    c_cam = R.T @ (c_map - T[:3, 3])
    det = {"label": "mug", "score": 0.8, "n_points": 300, "centroid_cam": c_cam.tolist(),
           "centroid_cam_completed": c_cam.tolist(), "extent_cam": [0.1, 0.1, 0.05],
           "centroid_map": c_map.tolist(), "extent_map": [0.09, 0.08, 0.11]}
    (label, score, c, q, size), = sl.detections_to_boxes([det], T)
    # world_model: pos = R_mc @ c + t ; box axes in the map = R_mc @ R(q)  (must be map-aligned)
    assert np.allclose(R @ np.array(c) + T[:3, 3], c_map, atol=1e-9)
    assert np.allclose(R @ quat_to_mat(*q), np.eye(3), atol=1e-9)
    assert size == [0.09, 0.08, 0.11] and label == "mug"
    det["score"] = 0.1
    assert sl.detections_to_boxes([det], T, min_score=0.3) == []


def test_rot_to_quat_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        A = rng.normal(size=(3, 3)); Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        assert np.allclose(quat_to_mat(*sl.rot_to_quat(Q)), Q, atol=1e-9)


class FakeImg:
    """Stand-in for sensor_msgs/Image with the fields image_to_numpy reads."""
    def __init__(self, a, enc):
        """Wrap array `a` with encoding `enc`."""
        self.height, self.width = a.shape[:2]
        self.encoding = enc
        self.step = a.shape[1] * a.itemsize * (a.shape[2] if a.ndim == 3 else 1)
        self.data = a.tobytes()


def test_image_to_numpy():
    rgb = (np.arange(H * W * 3) % 255).astype(np.uint8).reshape(H, W, 3)
    assert np.array_equal(sl.image_to_numpy(FakeImg(rgb, "rgb8")), rgb)
    d = np.random.default_rng(0).random((H, W)).astype(np.float32)
    assert np.array_equal(sl.image_to_numpy(FakeImg(d, "32FC1")), d)


def free_port():
    """An unused local TCP port."""
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_http_roundtrip_with_fake_server():
    pytest.importorskip("fastapi"); uvicorn = pytest.importorskip("uvicorn")
    from perception_server import rle, server
    from perception_server.models import FakeBackend
    server.STATE["backend"] = FakeBackend()
    port = free_port()
    cfg = uvicorn.Config(server.make_app(), host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(cfg)
    th = threading.Thread(target=srv.run, daemon=True); th.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            if sl.get_json(url + "/health", 1.0)["status"] == "ok":
                break
        except Exception:
            time.sleep(0.1)
    # a box 1.5 m ahead in the optical frame, wall at 3 m
    v, u = np.mgrid[0:H, 0:W]
    x, y = (u - K["cx"]) / FX, (v - K["cy"]) / FX
    depth = np.full((H, W), 3.0, np.float32)
    mask = (np.abs(x * 1.46) <= 0.05) & (np.abs(y * 1.46) <= 0.06)
    depth[mask] = 1.46
    rgb = np.full((H, W, 3), 120, np.uint8); rgb[mask] = (200, 40, 40)
    T = np.eye(4); T[:3, :3] = R_OPT; T[:3, 3] = [1.0, 2.0, 1.25]
    req = sl.build_request(rgb, depth, K, T, ["red block", "fridge"], "head_camera_optical", 3.5)
    ys, xs = np.nonzero(mask)
    req["gt"] = [{"label": "red block", "box_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                  "mask_rle": rle.encode(mask)}]
    res = sl.post_json(url + "/detect", req, 5.0)
    srv.should_exit = True
    boxes = sl.detections_to_boxes(res["detections"], T)
    assert len(boxes) == 1 and boxes[0][0] == "red block"
    c_map = R_OPT @ np.array(boxes[0][2]) + T[:3, 3]
    assert abs(c_map[0] - 2.5) < 0.05 and abs(c_map[1] - 2.0) < 0.03 and abs(c_map[2] - 1.25) < 0.03
    assert res["stamp"] == 3.5
