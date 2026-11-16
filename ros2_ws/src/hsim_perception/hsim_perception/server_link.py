"""ROS-free helpers for the perception client (testable without ROS): vocabulary, request encoding,
HTTP calls to the perception server, and turning server detections into box poses for
/perception/detections (same semantics as the line-of-sight stub: centre in the camera frame,
box axis-aligned with the map, its orientation given in the camera frame).

Wire format: services/perception_server/perception_server/protocol.py (kept in sync by hand:
the ROS container can't import the server package).
"""
import base64
import json
import urllib.request

import cv2
import numpy as np

# What the robot asks the detector for by default (open vocabulary: any name works). Kept short on purpose:
# every extra name costs latency and adds false labels (bench sweep 2026-11-10). The environment's own labels
# and the planner's targets are added on top.
DEFAULT_VOCAB = ["fridge", "sink", "stove", "microwave", "cabinet", "counter", "table",
                 "cup", "mug", "plate", "bowl", "bottle"]
# Structural labels in the RoboCasa objects files that are not useful prompts
SKIP_LABELS = {"wall", "floor", "wallaccessory", "accessory", "box"}
# Skill arguments that name objects (from /planner/status plans)
TARGET_KEYS = ("target", "object", "place", "hints")


def env_vocab(objects_yaml_data):
    """Object labels from an environment's objects.yaml (kinds of things in the room, not their positions)."""
    out = []
    for o in (objects_yaml_data or {}).get("objects", []):
        lab = o.get("label", "")
        if lab and lab not in SKIP_LABELS and o.get("detectable", True):
            out.append(lab)
    return list(dict.fromkeys(out))


def planner_targets(status_json):
    """Object names mentioned in a /planner/status plan (find_object(mug), navigate_to(fridge), hints...)."""
    try:
        s = json.loads(status_json)
    except (ValueError, TypeError):
        return []
    names = []
    for step in s.get("plan") or []:
        for k in TARGET_KEYS:
            v = step.get(k)
            for x in (v if isinstance(v, list) else [v]):
                if isinstance(x, str) and x.strip():
                    names.append(x.strip().replace("_", " "))
    return list(dict.fromkeys(names))


def image_to_numpy(msg):
    """sensor_msgs/Image (rgb8 / bgr8 / 32FC1) -> numpy."""
    h, w = msg.height, msg.width
    if msg.encoding in ("rgb8", "bgr8"):
        a = np.frombuffer(bytes(msg.data), np.uint8).reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3)
        return a[..., ::-1].copy() if msg.encoding == "bgr8" else a
    if msg.encoding == "32FC1":
        return np.frombuffer(bytes(msg.data), np.float32).reshape(h, msg.step // 4)[:, :w]
    if msg.encoding == "16UC1":                                  # millimetres
        mm = np.frombuffer(bytes(msg.data), np.uint16).reshape(h, msg.step // 2)[:, :w].astype(np.float32)
        return np.where(mm > 0, mm / 1000.0, np.nan).astype(np.float32)
    raise ValueError(f"unsupported image encoding {msg.encoding}")


def build_request(rgb, depth, K, T_map_cam, prompts, frame_id, stamp, jpeg_quality=90, depth_downsample=2,
                  box_threshold=None, text_threshold=None, priority=None):
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    d = depth[::depth_downsample, ::depth_downsample] if depth_downsample > 1 else depth
    mm = np.where(np.isfinite(d) & (d > 0), np.clip(d * 1000.0, 1, 65535), 0).astype(np.uint16)
    ok2, png = cv2.imencode(".png", mm)
    assert ok and ok2
    req = {"rgb_jpeg_b64": base64.b64encode(jpg.tobytes()).decode("ascii"),
           "depth_png16_b64": base64.b64encode(png.tobytes()).decode("ascii"),
           "intrinsics": K, "prompts": list(prompts), "frame_id": frame_id, "stamp": stamp, "return_masks": False}
    if T_map_cam is not None:
        req["T_map_cam"] = np.asarray(T_map_cam, float).ravel().tolist()
    if priority:
        req["priority_prompts"] = list(priority)[:4]
    if box_threshold is not None:
        req["box_threshold"] = box_threshold
    if text_threshold is not None:
        req["text_threshold"] = text_threshold
    return req


def post_json(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def rot_to_quat(R):
    """Rotation matrix -> (x, y, z, w)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


def detections_to_boxes(dets, T_map_cam, min_score=0.3, min_points=20):
    """Server detections -> [(label, score, centre_cam xyz, quat_cam xyzw, size xyz)].
    With the camera pose: centre = completed centre, box axis-aligned with the map (like the stub).
    Without it: centre and size in camera axes, identity orientation."""
    out = []
    R_cm = None if T_map_cam is None else np.asarray(T_map_cam, float)[:3, :3].T
    q = rot_to_quat(R_cm) if R_cm is not None else (0.0, 0.0, 0.0, 1.0)
    for d in dets:
        if d.get("score", 0) < min_score or d.get("n_points", 0) < min_points:
            continue
        if R_cm is not None and "extent_map" in d:
            c = d.get("centroid_cam_completed", d["centroid_cam"])
            size = d["extent_map"]
        else:
            c, size = d["centroid_cam"], d["extent_cam"]
        out.append((d["label"], float(d["score"]), [float(v) for v in c], q, [abs(float(v)) for v in size]))
    return out
