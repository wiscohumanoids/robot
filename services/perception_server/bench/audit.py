"""Mask-based audit of a running perception server (VM, perception env). Uses the per-pixel object map from
render_frames.py (every object incl. walls/floor/"box" parts) to see what each detection's MASK actually sits on.

  ~/hsim_envs/perception/bin/python bench/audit.py --frames ~/hsim_data/perc_bench_v2 --url http://127.0.0.1:8101 --out DIR [--vocab lean|full]

Scorer v2 (2026-11-10, after looking at the crops; every rule is listed so the numbers stay honest).
Each detection gets exactly one category. dominant = the object under most of its mask pixels.
  exact        dominant object has the detection's label (first, highest-score detection per object)
  exact_small  same label, dominant object visible but below the 150-pixel ground-truth cutoff (a correct detection
               the ground truth leaves out, e.g. a far carrot): correct, not in recall
  duplicate    same label, but that object already has a higher-scoring detection with that label   -> FALSE positive
  synonym      dominant object's label is a RoboCasa near-synonym (SYNONYMS) of the detection's label   -> correct
  group        no single object covers 50 % of the mask, but objects with the detection's label or a synonym
               together do (a "drawer" mask over a stack of drawer fronts, each its own RoboCasa object) -> correct
  part         dominant object is a RoboCasa "box" (cabinet carcass) and the label is cabinet/drawer/shelf -> correct
  wrong        dominant object is a real object with an unrelated label, incl. RoboCasa "accessory" objects
               (knife blocks, paper towels, utensil crocks: calling them cup/plate/bottle is a real error)
  background   mask mostly on wall / floor / wall accessory, or on no labelled object
Precision: strict = exact / all;  correct = (exact + exact_small + synonym + group + part) / all  (duplicates are errors).
Recall per ground-truth object: a detection whose dominant object it is, exact label (strict) or exact/synonym/group (lenient).
Responses are cached (--cache) so scorer changes and offline filters can be re-scored without the GPU (--rescore).
Writes report.json, detections.json and crops_<category>.jpg (montages, label + score on each crop).
"""
import argparse
import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_server import protocol, rle  # noqa: E402

LEAN_VOCAB = ["fridge", "sink", "stove", "microwave", "cabinet", "counter", "table", "cup", "mug", "plate", "bowl", "bottle"]
SKIP = {"wall", "floor", "wallaccessory", "accessory", "box"}
BACKGROUND = {"wall", "floor", "wallaccessory"}
# RoboCasa names one thing several ways (documented; used only for the "synonym" category / lenient numbers)
SYNONYMS = [{"cabinet", "drawer", "shelf"}, {"stove", "stovetop", "range hood", "oven"}, {"toaster oven", "microwave"},
            {"bottle", "juice", "bottled drink"}]
CORRECT = ("exact", "exact_small", "synonym", "group", "part")


def synonym(a, b):
    return a != b and any(a in g and b in g for g in SYNONYMS)


def post(url, body, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def montage(items, path, n=48, size=96):
    import cv2
    if not items:
        return
    items = items[:n]
    cols = 8
    rows = (len(items) + cols - 1) // cols
    canvas = np.full((rows * (size + 14), cols * size, 3), 255, np.uint8)
    for k, (crop, text) in enumerate(items):
        r, c = divmod(k, cols)
        h, w = crop.shape[:2]
        s = size / max(h, w)
        img = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
        canvas[r * (size + 14):r * (size + 14) + img.shape[0], c * size:c * size + img.shape[1]] = img
        cv2.putText(canvas, text[:18], (c * size + 2, r * (size + 14) + size + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8101")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", choices=["lean", "full", "none"], default="lean")
    ap.add_argument("--extra", nargs="*", default=[], help="extra prompt names (e.g. distractor classes)")
    ap.add_argument("--targets", default="", help="comma list: if set, only these labels are scored as detections (others dropped)")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--request-extra", default="{}", help="JSON merged into every request (server options)")
    ap.add_argument("--cache", default=None, help="dir: save raw server responses here (or read them with --rescore)")
    ap.add_argument("--rescore", action="store_true", help="score cached responses, no server calls")
    ap.add_argument("--filter", default="", help="python module:function(det, frame_meta) -> bool, applied offline")
    ap.add_argument("--offline-nms", type=float, default=None, help="apply postprocess.mask_nms(thr) offline (cached masks)")
    ap.add_argument("--offline-priors", action="store_true", help="apply postprocess.apply_priors offline")
    args = ap.parse_args()
    fdir, out = Path(args.frames), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = json.load(open(fdir / "index.json"))
    K = idx["K"]
    cats = Counter()
    by_label = defaultdict(Counter)
    crops = defaultdict(list)
    dets_dump = []
    rec_strict, rec_len, rec_n = Counter(), Counter(), Counter()
    lat = []
    targets = {t.strip() for t in args.targets.split(",") if t.strip()}
    filt = None
    if args.filter:
        import importlib
        mod, fn = args.filter.split(":")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        filt = getattr(importlib.import_module(mod), fn)
    extra_req = json.loads(args.request_extra)
    for f in idx["frames"]:
        z = np.load(fdir / f["file"])
        rgb, depth, T, objmap = z["rgb"], z["depth"], z["T_map_cam"], z["objmap"]
        objs = f["objects_all"]
        base = {"lean": LEAN_VOCAB, "full": [], "none": []}[args.vocab]
        prompts = list(dict.fromkeys(base + [l for l in f["vocab"] if l not in SKIP] + args.extra))
        body = {"rgb_jpeg_b64": protocol.encode_rgb_jpeg(rgb), "depth_png16_b64": protocol.encode_depth_png16(depth, 2),
                "intrinsics": K, "T_map_cam": T.ravel().tolist(), "prompts": prompts, "return_masks": True, **extra_req}
        cache = Path(args.cache) / (f["file"] + ".json") if args.cache else None
        if args.rescore:
            res = json.load(open(cache))
        else:
            res = post(args.url + "/detect", body)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(cache, "w"))
        lat.append(res["timing_ms"]["total"])
        dets = [d for d in res["detections"] if d["score"] >= args.min_score and (not targets or d["label"] in targets)]
        if filt:
            dets = [d for d in dets if filt(d, f)]
        if args.offline_nms is not None or args.offline_priors:
            from perception_server import postprocess
            for d in dets:
                d["mask"] = rle.decode(d["mask_rle"])
            if args.offline_nms is not None:
                dets = postprocess.mask_nms(dets, args.offline_nms)
            if args.offline_priors:
                dets, _ = postprocess.apply_priors(dets)
        gt_ids = {}                                       # object index -> ground-truth entry (visible, labelled)
        pix = np.bincount(objmap[objmap >= 0].ravel(), minlength=len(objs))
        for oi, o in enumerate(objs):
            if o["label"] not in SKIP and pix[oi] >= 150:
                gt_ids[oi] = o
        matched = set()
        hit_s, hit_l = set(), set()
        for d in sorted(dets, key=lambda d: -d["score"]):
            m = rle.decode(d["mask_rle"]) if "mask_rle" in d else None
            dom, cov, grp = -1, 0.0, 0.0
            dl = d["label"]
            if m is not None and m.any():
                ids = objmap[m]
                ids = ids[ids >= 0]
                if ids.size:
                    cnt = np.bincount(ids, minlength=len(objs))
                    dom = int(cnt.argmax()); cov = float(cnt[dom] / m.sum())
                    grp = float(sum(cnt[i] for i in np.flatnonzero(cnt) if objs[i]["label"] == dl or synonym(objs[i]["label"], dl)) / m.sum())
            ol = objs[dom]["label"] if dom >= 0 else ""
            if dom >= 0 and cov >= 0.5 and ol not in BACKGROUND:
                if ol == dl:
                    if dom not in gt_ids:
                        cat = "exact_small"
                    elif (dom, dl) in matched:
                        cat = "duplicate"
                    else:
                        cat = "exact"; matched.add((dom, dl)); hit_s.add(dom); hit_l.add(dom)
                elif synonym(ol, dl):
                    cat = "synonym"
                    if dom in gt_ids:
                        hit_l.add(dom)
                elif objs[dom]["class"] == "box" and dl in ("cabinet", "drawer", "shelf"):
                    cat = "part"
                else:
                    cat = "wrong"                         # incl. accessories (knife blocks, paper towels) and "box" parts
            elif grp >= 0.5:
                cat = "group"
                if dom in gt_ids and (ol == dl or synonym(ol, dl)):
                    hit_l.add(dom)
            else:
                cat = "background"
            cats[cat] += 1
            by_label[dl][cat] += 1
            under = objs[dom]["label"] + "/" + objs[dom]["class"] if dom >= 0 else "-"
            dets_dump.append({"frame": f["file"], "kitchen": f["kitchen"], "dom": dom, "label": dl, "score": d["score"], "cat": cat, "under": under,
                              "coverage": round(cov, 2), "box": d["box_xyxy"], "extent_map": d.get("extent_map"),
                              "centroid_map": d.get("centroid_map"), "n_points": d.get("n_points", 0)})
            x1, y1, x2, y2 = [int(v) for v in d["box_xyxy"]]
            if x2 > x1 and y2 > y1:
                crops[cat].append((rgb[max(0, y1):y2, max(0, x1):x2], f'{dl} {d["score"]:.2f}>{under.split("/")[0]}'))
        for oi, o in gt_ids.items():
            rec_n[o["label"]] += 1
            rec_strict[o["label"]] += oi in hit_s
            rec_len[o["label"]] += oi in hit_l
    n = sum(cats.values())
    small = {l for f in idx["frames"] for l in f.get("movable", [])}
    def recall(labels, c):
        a = sum(c[l] for l in labels); b = sum(rec_n[l] for l in labels)
        return round(a / max(b, 1), 3)
    fixtures = [l for l in rec_n if l not in small]
    rep = {
        "detections": n, "categories": dict(cats),
        "precision_strict": round(cats["exact"] / max(n, 1), 3),
        "precision_lenient": round((cats["exact"] + cats["synonym"]) / max(n, 1), 3),
        "precision_correct": round(sum(cats[c] for c in CORRECT) / max(n, 1), 3),
        "recall_small_strict": recall(small, rec_strict), "recall_small_lenient": recall(small, rec_len),
        "recall_fixture_strict": recall(fixtures, rec_strict), "recall_fixture_lenient": recall(fixtures, rec_len),
        "per_label_recall_strict": {l: f"{rec_strict[l]}/{rec_n[l]}" for l in sorted(rec_n)},
        "per_detected_label": {l: dict(c) for l, c in sorted(by_label.items())},
        "latency_ms_median": round(float(np.median(lat)), 1), "prompts_example": prompts,
    }
    json.dump(rep, open(out / "report.json", "w"), indent=1)
    json.dump(dets_dump, open(out / "detections.json", "w"))
    for c, items in crops.items():
        montage(sorted(items, key=lambda t: t[1]), out / f"crops_{c}.jpg")
    print(json.dumps({k: rep[k] for k in ("detections", "categories", "precision_strict", "precision_lenient", "precision_correct",
                                          "recall_small_strict", "recall_small_lenient", "recall_fixture_strict",
                                          "recall_fixture_lenient", "latency_ms_median")}))


if __name__ == "__main__":
    main()
