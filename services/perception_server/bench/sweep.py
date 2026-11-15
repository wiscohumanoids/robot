"""Parameter sweep for the detection backend on rendered frames (VM, perception env, short GPU test).

  ~/hsim_envs/perception/bin/python bench/sweep.py --frames ~/hsim_data/perc_bench

For each (model, box threshold, chunk size): recall for small objects and fixtures (strict label, and
lenient = near-synonym groups count), precision (strict / lenient), and median detection time.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_server.models import GroundedSam2Backend  # noqa: E402
from eval_server import DEFAULT_VOCAB, SKIP, iou  # noqa: E402

GROUPS = [{"cabinet", "drawer", "shelf"}, {"stove", "stovetop", "oven", "range hood"},
          {"toaster", "toaster oven", "microwave"}, {"juice", "milk", "bottled drink", "bottle", "can", "box"},
          {"carrot", "broccoli", "vegetable", "bell pepper"}, {"counter", "table"}]


def same(a, b, lenient):
    if a == b:
        return True
    return lenient and any(a in g and b in g for g in GROUPS)


def score(frames, results, lenient):
    sh, sn, fh, fn, tp, nd = 0, 0, 0, 0, 0, 0
    for f, dets in zip(frames, results):
        gts = f["gt"]
        nd += len(dets)
        tp += sum(any(same(g["label"], d["label"], lenient) and iou(g["box_xyxy"], d["box_xyxy"]) >= 0.3 for g in gts) for d in dets)
        for g in gts:
            hit = any(same(g["label"], d["label"], lenient) and iou(g["box_xyxy"], d["box_xyxy"]) >= 0.3 for d in dets)
            if max(g["size"]) < 0.3:
                sn += 1; sh += hit
            else:
                fn += 1; fh += hit
    return round(sh / max(sn, 1), 2), round(fh / max(fn, 1), 2), round(tp / max(nd, 1), 2), nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--models", nargs="+", default=["IDEA-Research/grounding-dino-base", "IDEA-Research/grounding-dino-tiny"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.2, 0.25, 0.3])
    ap.add_argument("--chunks", nargs="+", type=int, default=[4, 8, 12])
    args = ap.parse_args()
    fdir = Path(args.frames)
    idx = json.load(open(fdir / "index.json"))
    frames = idx["frames"]
    imgs = [np.load(fdir / f["file"])["rgb"] for f in frames]
    rows = []
    for mid in args.models:
        be = GroundedSam2Backend(dino_id=mid)
        for chunk in args.chunks:
            be.chunk = chunk
            for thr in args.thresholds:
                res, times = [], []
                for f, rgb in zip(frames, imgs):
                    prompts = list(dict.fromkeys(DEFAULT_VOCAB + [l for l in f["vocab"] if l not in SKIP]))
                    t0 = time.perf_counter()
                    dets = detect_boxes(be, rgb, prompts, thr)
                    times.append((time.perf_counter() - t0) * 1000)
                    res.append(dets)
                strict, len_ = score(frames, res, False), score(frames, res, True)
                row = {"model": mid.split("/")[-1], "chunk": chunk, "thr": thr, "small_recall": strict[0],
                       "fixture_recall": strict[1], "precision": strict[2], "lenient_small": len_[0],
                       "lenient_fixture": len_[1], "lenient_precision": len_[2], "dets": strict[3],
                       "detect_ms": round(float(np.median(times)), 1)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    json.dump(rows, open(fdir / "sweep.json", "w"), indent=1)


def detect_boxes(be, rgb, prompts, thr):
    dets, _ = be.detect(rgb, prompts, box_threshold=thr, text_threshold=min(thr, 0.25), segment=False)
    return dets


if __name__ == "__main__":
    main()
