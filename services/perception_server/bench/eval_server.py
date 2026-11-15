"""Score the running perception server on frames from render_frames.py (run on the VM, perception env).

  ~/hsim_envs/perception/bin/python bench/eval_server.py --frames ~/hsim_data/perc_bench [--url http://127.0.0.1:8100] [--overlays 6]

Prompts per frame = DEFAULT_VOCAB (same list as the ROS client) + the kitchen's object labels.
Scores:
  small objects (largest side < 0.3 m): detected = a detection with the same label whose box overlaps the
     true visible box (IoU >= 0.3); 3D error = |estimated centre - true centre| in the map frame
  fixtures: detected = same label, IoU >= 0.3; 3D "inside" = estimated centre inside the true box + 0.15 m
     (a centre distance is unfair for a 3 m counter seen in part)
  precision: detections that overlap (IoU >= 0.3) any visible true object with the same label
  latency: server total (decode + detect + segment + lift) and round trip; GPU peak memory from /health
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_server import protocol  # noqa: E402

FULL_VOCAB = ["fridge", "sink", "stove", "oven", "microwave", "dishwasher", "cabinet", "drawer", "counter",
              "table", "chair", "stool", "shelf", "cup", "mug", "plate", "bowl", "bottle", "can", "box",
              "fruit", "apple", "banana", "orange", "vegetable", "pan", "pot", "kettle", "toaster",
              "coffee machine"]
# same list as ros2_ws/src/hsim_perception/hsim_perception/server_link.py DEFAULT_VOCAB
LEAN_VOCAB = ["fridge", "sink", "stove", "microwave", "cabinet", "counter", "table",
              "cup", "mug", "plate", "bowl", "bottle"]
DEFAULT_VOCAB = FULL_VOCAB                     # sweep.py imports this name
SKIP = {"wall", "floor", "wallaccessory", "accessory", "box"}


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def post(url, body, timeout=30):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--overlays", type=int, default=6)
    ap.add_argument("--box-threshold", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--vocab", choices=["full", "lean"], default="lean")
    ap.add_argument("--priority", action="store_true",
                    help="send the kitchen's movable objects as priority prompts (what a planner would be looking for)")
    args = ap.parse_args()
    fdir = Path(args.frames)
    idx = json.load(open(fdir / "index.json"))
    K = idx["K"]
    small_hit, small_err, small_n = 0, [], 0
    fix_hit, fix_inside, fix_n = 0, 0, 0
    per_label = {}
    tp_det, n_det = 0, 0
    lat_srv, lat_rt = [], []
    scored, dump = [], []
    out_dir = Path(args.out or fdir / "eval"); out_dir.mkdir(exist_ok=True)
    import cv2
    for k, f in enumerate(idx["frames"]):
        z = np.load(fdir / f["file"])
        rgb, depth, T = z["rgb"], z["depth"], z["T_map_cam"]
        base = LEAN_VOCAB if args.vocab == "lean" else FULL_VOCAB
        prompts = list(dict.fromkeys(base + [l for l in f["vocab"] if l not in SKIP]))
        body = {"rgb_jpeg_b64": protocol.encode_rgb_jpeg(rgb), "depth_png16_b64": protocol.encode_depth_png16(depth, 2),
                "intrinsics": K, "T_map_cam": T.ravel().tolist(), "prompts": prompts, "return_masks": False}
        if args.box_threshold is not None:
            body["box_threshold"] = args.box_threshold
        if args.priority:
            body["priority_prompts"] = f.get("movable", [])[:4]
        t0 = time.perf_counter()
        res = post(args.url + "/detect", body)
        lat_rt.append((time.perf_counter() - t0) * 1000)
        lat_srv.append(res["timing_ms"]["total"])
        dets = res["detections"]
        n_det += len(dets)
        gts = f["gt"]
        prio = set(body.get("priority_prompts", []))
        for d in dets:
            ok = any(g["label"] == d["label"] and iou(g["box_xyxy"], d["box_xyxy"]) >= 0.3 for g in gts)
            tp_det += ok
            scored.append((d["score"], bool(ok), d["label"] in prio))
            dump.append({"frame": f["file"], "label": d["label"], "score": d["score"], "tp": bool(ok),
                         "priority": d["label"] in prio, "extent_map": d.get("extent_map"), "box": d["box_xyxy"],
                         "n_points": d.get("n_points", 0)})
        for g in gts:
            cands = [d for d in dets if d["label"] == g["label"] and iou(g["box_xyxy"], d["box_xyxy"]) >= 0.3]
            best = max(cands, key=lambda d: iou(g["box_xyxy"], d["box_xyxy"])) if cands else None
            pl = per_label.setdefault(g["label"], [0, 0]); pl[1] += 1
            if best:
                pl[0] += 1
            if max(g["size"]) < 0.3:
                small_n += 1
                if best:
                    small_hit += 1
                    if "centroid_map" in best:
                        small_err.append(float(np.linalg.norm(np.array(best["centroid_map"]) - np.array(g["center_map"]))))
            else:
                fix_n += 1
                if best:
                    fix_hit += 1
                    if "centroid_map" in best:
                        c = np.array(best["centroid_map"]); lo = np.array(g["center_map"]) - np.array(g["size"]) / 2 - 0.15
                        hi = np.array(g["center_map"]) + np.array(g["size"]) / 2 + 0.15
                        fix_inside += bool(np.all((c >= lo) & (c <= hi)))
        if k < args.overlays:
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for g in gts:
                x1, y1, x2, y2 = g["box_xyxy"]; cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 1)
            for d in dets:
                x1, y1, x2, y2 = map(int, d["box_xyxy"]); cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f'{d["label"]} {d["score"]:.2f}', (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.imwrite(str(out_dir / f"overlay_{f['file'].replace('.npz', '.jpg')}"), img)
    health = json.loads(urllib.request.urlopen(args.url + "/health", timeout=10).read())
    rep = {
        "frames": len(idx["frames"]),
        "small_objects": {"n": small_n, "recall": round(small_hit / max(small_n, 1), 3),
                          "err_median_m": round(float(np.median(small_err)), 3) if small_err else None,
                          "err_p90_m": round(float(np.percentile(small_err, 90)), 3) if small_err else None},
        "fixtures": {"n": fix_n, "recall": round(fix_hit / max(fix_n, 1), 3),
                     "centre_inside_true_box": round(fix_inside / max(fix_hit, 1), 3)},
        "precision": round(tp_det / max(n_det, 1), 3), "detections": n_det,
        "per_label_recall": {l: f"{a}/{b}" for l, (a, b) in sorted(per_label.items())},
        "latency_ms": {"server_median": round(float(np.median(lat_srv)), 1), "server_p95": round(float(np.percentile(lat_srv, 95)), 1),
                       "roundtrip_median": round(float(np.median(lat_rt)), 1)},
        "server": health,
    }
    curve = {}
    for thr in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
        for kind, sel in (("all", lambda p: True), ("priority", lambda p: p), ("other", lambda p: not p)):
            s = [ok for sc, ok, p in scored if sc >= thr and sel(p)]
            curve[f"{kind}@{thr}"] = f"{sum(s)}/{len(s)} = {sum(s) / max(len(s), 1):.2f}"
    rep["precision_vs_min_score"] = curve
    json.dump(dump, open(out_dir / "detections.json", "w"))
    json.dump(rep, open(out_dir / "report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
