"""Wire format for POST /detect (JSON), shared by the server and the ROS client.

Request:
  rgb_jpeg_b64 | rgb_png_b64     colour image (RGB order inside the file is standard JPEG/PNG)
  depth_png16_b64                depth, uint16 millimetres, 0 = no depth (may be lower resolution than RGB)
  intrinsics                     {fx, fy, cx, cy, width, height} of the RGB image
  T_map_cam                      optional 4x4 (row-major list), camera (optical) -> map
  prompts                        list of object names, e.g. ["cup", "plate", "fridge"]
  priority_prompts               optional names to look for with extra sensitivity (each gets its own detector pass,
                                 max 4): what the planner is searching for right now
  distractors                    optional names prompted but never returned (absorb confusions; suppress overlapping
                                 lower-score targets at IoU > distractor_iou, default 0.6). See models.KITCHEN_DISTRACTORS
  chunk, cross_label_iou         optional per-request overrides
  box_threshold, text_threshold  optional Grounding DINO thresholds
  return_masks                   default true (RLE)
  frame_id, stamp                echoed back
  gt                             fake backend only: [{label, box_xyxy, mask_rle?, score?}]
Response:
  detections: [{label, score, box_xyxy, mask_rle?, n_points, centroid_cam, extent_cam,
                centroid_map?, extent_map?, completed?}]
  timing_ms: {decode, detect, segment, lift, total};  backend;  image_size [H, W];  frame_id, stamp
Detections whose depth support is too small (< 20 points) are returned with n_points 0 and no 3D.
"""
import base64

import numpy as np

try:
    import cv2
except ImportError:          # server installs opencv-python-headless; clients in ROS have python3-opencv
    cv2 = None


def b64(data: bytes) -> str:
    """Bytes to base64 text."""
    return base64.b64encode(data).decode("ascii")


def unb64(s: str) -> bytes:
    """Base64 text to bytes."""
    return base64.b64decode(s)


def encode_rgb_jpeg(rgb: np.ndarray, quality: int = 90) -> str:
    """RGB image as base64 JPEG."""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return b64(buf.tobytes())


def encode_depth_png16(depth_m: np.ndarray, downsample: int = 1) -> str:
    """Depth in metres as a base64 16-bit PNG in millimetres (0 = no data)."""
    d = depth_m[::downsample, ::downsample] if downsample > 1 else depth_m
    mm = np.where(np.isfinite(d) & (d > 0), np.clip(d * 1000.0, 1, 65535), 0).astype(np.uint16)
    ok, buf = cv2.imencode(".png", mm)
    assert ok
    return b64(buf.tobytes())


def decode_rgb(req: dict) -> np.ndarray:
    """RGB image from a request (JPEG or PNG)."""
    raw = np.frombuffer(unb64(req.get("rgb_jpeg_b64") or req["rgb_png_b64"]), np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_depth(req: dict) -> np.ndarray:
    """Depth in metres from a request (NaN where there is no data)."""
    raw = np.frombuffer(unb64(req["depth_png16_b64"]), np.uint8)
    mm = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return np.where(mm > 0, mm / 1000.0, np.nan).astype(np.float32)
