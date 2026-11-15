"""Simulate the world model's object memory on the benchmark (offline, from audit.py detections.json).

Each kitchen's frames are taken as successive views. Detections are associated like world_model.py
(same label, 2D distance < max(0.4 m, half the larger horizontal size), running-average position) and a
track is committed after `confirm` views. Memory precision = committed tracks whose detections are mostly
correct (audit categories exact/exact_small/synonym/group/part) / committed tracks. Memory recall = labelled
objects visible in >= 2 frames that some correct committed track covers / those objects.

  python3 bench/memory_sim.py AUDIT_DIR [AUDIT_DIR ...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CORRECT = {"exact", "exact_small", "synonym", "group", "part"}


def simulate(dets, confirm):
    by_k = defaultdict(list)
    for d in dets:
        if d.get("centroid_map") and d.get("extent_map"):
            by_k[d["kitchen"]].append(d)
    committed, n_correct, covered = 0, 0, set()
    for k, ds in by_k.items():
        tracks = []
        for frame in sorted({d["frame"] for d in ds}):
            for d in [x for x in ds if x["frame"] == frame]:
                pos = np.array(d["centroid_map"][:2]); size = max(d["extent_map"][:2])
                gate = max(0.4, 0.5 * size)
                best, bd = None, gate
                for tr in tracks:
                    if tr["label"] == d["label"]:
                        dd = float(np.linalg.norm(tr["pos"] - pos))
                        if dd < bd:
                            best, bd = tr, dd
                if best is None:
                    best = {"label": d["label"], "pos": pos, "frames": set(), "dets": []}
                    tracks.append(best)
                else:
                    best["pos"] = 0.7 * best["pos"] + 0.3 * pos
                best["frames"].add(frame); best["dets"].append(d)
        for tr in tracks:
            if len(tr["frames"]) >= confirm:
                committed += 1
                good = [d for d in tr["dets"] if d["cat"] in CORRECT]
                if len(good) * 2 > len(tr["dets"]):
                    n_correct += 1
                    covered |= {(k, d["dom"]) for d in good if d["cat"] in ("exact", "synonym", "group")}
    return committed, n_correct, covered


def main():
    for adir in sys.argv[1:]:
        dets = json.load(open(Path(adir) / "detections.json"))
        frames_idx = json.load(open(Path(adir).parents[0].parent.parent.parent / "hsim_data/perc_bench_v2/index.json")) if False else None
        # ground-truth objects seen (as some detection's dominant object in an exact/synonym/group sense) in >= 2 frames:
        # use the frame index from the benchmark for visibility
        idx = json.load(open(Path.home() / "hsim_data/perc_bench_v2/index.json"))
        vis = defaultdict(set)
        for f in idx["frames"]:
            objs = f["objects_all"]
            names = {o["name"]: i for i, o in enumerate(objs)}
            for g in f["gt"]:
                vis[(f["kitchen"], names[g["name"]])].add(f["file"])
        gt2 = {k for k, fr in vis.items() if len(fr) >= 2}
        print(Path(adir).name)
        for c in (1, 2, 3):
            n, good, cov = simulate(dets, c)
            print(f"  confirm {c} view(s): committed {n:4d}, memory precision {good / max(n, 1):.3f}, "
                  f"memory recall {len(cov & gt2) / max(len(gt2), 1):.3f} ({len(cov & gt2)}/{len(gt2)} objects seen in >= 2 frames)")


if __name__ == "__main__":
    main()
