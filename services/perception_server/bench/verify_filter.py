"""Offline filters for audit.py --filter from verify_crops.py decisions (priors applied first, as in the prototype).
Set VERIFY_DECISIONS=<decisions.json> in the environment."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from perception_server import priors  # noqa: E402
from verify_crops import key  # noqa: E402

_D = json.load(open(os.environ["VERIFY_DECISIONS"]))


def _ok(det, frame, rule):
    if not priors.check(det["label"], det.get("centroid_map"), det.get("extent_map"))[0]:
        return False
    return _D.get(key(frame["file"], det), {}).get(rule, True)


def top1(det, frame):
    return _ok(det, frame, "top1")


def top2(det, frame):
    return _ok(det, frame, "top2")


# Large fixtures seen in part are ambiguous as crops (SigLIP rejected 25/73 good cabinets, 12/24 good counters);
# they are left to the 3D priors. Everything else (tableware, food, appliances, windows) is verified.
NO_VERIFY = {"cabinet", "drawer", "shelf", "counter", "fridge", "sink", "stove", "stovetop", "oven", "range hood",
             "dishwasher", "table"}


def selective_top1(det, frame):
    return _ok(det, frame, "top1") if det["label"] not in NO_VERIFY else _ok(det, frame, "_none_")


def selective_top2(det, frame):
    return _ok(det, frame, "top2") if det["label"] not in NO_VERIFY else _ok(det, frame, "_none_")


def selective_top3(det, frame):
    if det["label"] in NO_VERIFY:
        return _ok(det, frame, "_none_")
    if not priors.check(det["label"], det.get("centroid_map"), det.get("extent_map"))[0]:
        return False
    from audit import synonym
    v = _D.get(key(frame["file"], det))
    return True if v is None else any(n == det["label"] or synonym(n, det["label"]) for n in v["best"][:3])
