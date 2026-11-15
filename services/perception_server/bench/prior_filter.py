"""Offline filter for audit.py --filter: drop detections that fail the 3D size/placement priors."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_server import priors  # noqa: E402


def keep(det, frame):
    return priors.check(det["label"], det.get("centroid_map"), det.get("extent_map"))[0]
