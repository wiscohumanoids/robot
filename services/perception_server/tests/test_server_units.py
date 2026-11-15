"""Unit tests (no GPU, no models): RLE, mask+depth lifting, phrase mapping, wire format + fake backend.
Run:  python -m pytest services/perception_server/tests -q   (needs numpy, opencv-python-headless; fastapi+httpx for the HTTP test)
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_server import geometry, protocol, rle, server  # noqa: E402
from perception_server.models import FakeBackend, map_phrase, nms  # noqa: E402

W, H = 640, 320
FX = (W / 2) / math.tan(math.radians(110) / 2)          # ZED X sim: 110 deg horizontal FOV
K = {"fx": FX, "fy": FX, "cx": (W - 1) / 2, "cy": (H - 1) / 2, "width": W, "height": H}


def scene(box_c=(0.2, 0.1, 1.5), box_half=(0.05, 0.06, 0.04), bg=3.0):
    """Depth of a wall at `bg` m with a box (optical frame) in front; returns depth, mask."""
    v, u = np.mgrid[0:H, 0:W]
    x, y = (u - K["cx"]) / FX, (v - K["cy"]) / FX       # ray direction per pixel (z = 1)
    depth = np.full((H, W), bg, np.float32)
    cx, cy, cz = box_c
    hx, hy, hz = box_half
    zf = cz - hz                                          # front face
    hit = (np.abs(x * zf - cx) <= hx) & (np.abs(y * zf - cy) <= hy)
    depth[hit] = zf
    return depth, hit


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = rng.random((37, 53)) > rng.random()
        r = rle.encode(m)
        assert np.array_equal(rle.decode(r), m)
        assert rle.area(r) == m.sum()
    m = np.ones((4, 5), bool)
    assert rle.encode(m)["counts"][0] == 0


def test_lift_camera_frame():
    depth, mask = scene()
    g = geometry.lift(mask, depth, K)
    assert g is not None and g["n_points"] > 100
    c = np.array(g["centroid_cam"])
    assert abs(c[0] - 0.2) < 0.01 and abs(c[1] - 0.1) < 0.01
    assert abs(c[2] - 1.46) < 0.01                        # front face (camera sees the surface)
    assert abs(g["extent_cam"][0] - 0.10) < 0.02


def test_lift_half_res_depth_and_leaky_mask():
    depth, mask = scene()
    leaky = mask.copy()
    leaky[:, :] |= np.roll(mask, 25, axis=1)              # mask leaks onto the wall behind
    g = geometry.lift(leaky, depth[::2, ::2], K)
    assert abs(g["centroid_cam"][2] - 1.46) < 0.03        # wall pixels rejected as depth outliers


def test_lift_map_frame_and_completion():
    depth, mask = scene(box_c=(0.0, 0.0, 1.5))
    # camera at (1, 2, 1.2) in the map, looking along map +x (optical z = map x, x = -y, y = -z)
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [1, 2, 1.2]
    g = geometry.lift(mask, depth, K, T)
    c = np.array(g["centroid_map"])
    assert g["completed"]
    assert abs(c[0] - 2.5) < 0.03                          # box centre 1.5 m ahead, completed to ~the true centre
    assert abs(c[1] - 2.0) < 0.02 and abs(c[2] - 1.2) < 0.02


def test_lift_rejects_empty():
    depth, mask = scene()
    depth[:] = np.nan
    assert geometry.lift(mask, depth, K) is None


def test_map_phrase_and_nms():
    labels = ["coffee mug", "plate", "sugar cube", "fridge"]
    assert map_phrase("plate", labels) == "plate"
    assert map_phrase("mug", labels) == "coffee mug"
    assert map_phrase("sugar", labels) == "sugar cube"
    assert map_phrase("table", labels) is None
    b = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], np.float32)
    assert sorted(nms(b, np.array([0.9, 0.8, 0.7]), 0.5)) == [0, 2]


def make_request(depth_ds=2, with_pose=True):
    depth, mask = scene(box_c=(0.0, 0.0, 1.5))
    rgb = np.full((H, W, 3), 128, np.uint8)
    rgb[mask] = (200, 30, 30)
    ys, xs = np.nonzero(mask)
    req = {"rgb_jpeg_b64": protocol.encode_rgb_jpeg(rgb), "depth_png16_b64": protocol.encode_depth_png16(depth, depth_ds),
           "intrinsics": K, "prompts": ["red block", "plate"], "frame_id": "head_camera_optical", "stamp": 12.5,
           "gt": [{"label": "red block", "box_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], "mask_rle": rle.encode(mask)},
                  {"label": "fridge", "box_xyxy": [0, 0, 10, 10]}]}
    if with_pose:
        R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [1, 2, 1.2]
        req["T_map_cam"] = T.ravel().tolist()
    return req


def test_fake_backend_end_to_end():
    server.STATE["backend"] = FakeBackend()
    out = server.run_detect(make_request())
    assert [d["label"] for d in out["detections"]] == ["red block"]      # fridge not prompted
    d = out["detections"][0]
    assert d["n_points"] > 50 and abs(d["centroid_map"][0] - 2.5) < 0.04
    assert rle.decode(d["mask_rle"]).shape == (H, W)
    assert out["stamp"] == 12.5 and set(out["timing_ms"]) >= {"decode", "detect", "segment", "lift", "total"}


def test_http_fake():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    server.STATE["backend"] = FakeBackend()
    c = TestClient(server.make_app())
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/detect", json=make_request(with_pose=False))
    assert r.status_code == 200 and r.json()["detections"][0]["label"] == "red block"
    assert "centroid_map" not in r.json()["detections"][0]
    assert c.post("/detect", json={"prompts": []}).status_code == 400


def test_mask_nms_and_priors():
    from perception_server import postprocess, priors
    a = np.zeros((10, 10), bool); a[2:8, 2:8] = True
    b = np.zeros((10, 10), bool); b[2:8, 3:8] = True            # IoU with a = 30/36
    c = np.zeros((10, 10), bool); c[0:2, 0:2] = True
    dets = [{"label": "cup", "score": 0.5, "mask": b}, {"label": "cup", "score": 0.7, "mask": a},
            {"label": "bowl", "score": 0.6, "mask": b}, {"label": "cup", "score": 0.4, "mask": c}]
    kept = postprocess.mask_nms(dets)
    assert [(d["label"], d["score"]) for d in kept] == [("cup", 0.7), ("bowl", 0.6), ("cup", 0.4)]
    # priors: a "fridge" whose top is at 0.86 m (a dishwasher) fails; a real one passes; unknown labels pass
    assert not priors.check("fridge", [0, 0, 0.45], [0.6, 0.6, 0.8])[0]
    assert priors.check("fridge", [0, 0, 0.9], [0.8, 0.7, 1.7])[0]
    assert not priors.check("counter", [0, 0, 0.02], [2.0, 1.0, 0.04])[0]          # the floor
    assert not priors.check("cup", [0, 0, 1.05], [0.12, 0.12, 0.30])[0]            # paper towel holder
    assert priors.check("mug", [0, 0, 0.97], [0.11, 0.09, 0.10])[0]
    assert priors.check("teapot", [0, 0, 0.0], [5, 5, 5])[0]
    assert priors.check("fridge", None, None)[0]
    kept, rej = postprocess.apply_priors([{"label": "plate", "centroid_map": [0, 0, 1.0], "extent_map": [0.2, 0.2, 0.25]}])
    assert not kept and "height" in rej[0]["rejected"]
