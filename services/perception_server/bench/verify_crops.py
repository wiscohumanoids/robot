"""Prototype: verify each detection's crop with SigLIP 2 (VM, perception env). Offline on cached responses.

  ~/hsim_envs/perception/bin/python bench/verify_crops.py --frames ~/hsim_data/perc_bench_v2 --cache CACHE_DIR \
      --model google/siglip2-base-patch16-224 --out DECISIONS.json

For each detection (after the 3D priors): crop the box (+15 % padding, at least 40 px), score it against
"a photo of a {name}" for every candidate name (the frame's target labels + kitchen distractors + wall/floor),
softmax over the names. Rules written to DECISIONS.json (keyed by frame + box) for audit.py --filter verify_filter:<rule>:
  top1   the detection's label (or a RoboCasa near-synonym) is the best-scoring name
  top2   ... is among the best two
Also reports verification time per frame (all crops of a frame in one batch).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from perception_server import priors  # noqa: E402
from perception_server.models import KITCHEN_DISTRACTORS  # noqa: E402
from audit import LEAN_VOCAB, SKIP, synonym  # noqa: E402


def emb(out):
    """transformers >= 5 returns an output object (pooler_output = the embedding); older versions a tensor."""
    return out.pooler_output if hasattr(out, "pooler_output") else out


def key(frame, det):
    return f'{frame}|{",".join(f"{v:.1f}" for v in det["box_xyxy"])}|{det["label"]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
    fdir = Path(args.frames)
    idx = json.load(open(fdir / "index.json"))
    decisions, times, n = {}, [], 0
    text_cache = {}
    for f in idx["frames"]:
        res = json.load(open(Path(args.cache) / (f["file"] + ".json")))
        rgb = np.load(fdir / f["file"])["rgb"]
        H, W = rgb.shape[:2]
        dets = [d for d in res["detections"] if priors.check(d["label"], d.get("centroid_map"), d.get("extent_map"))[0]]
        if not dets:
            continue
        names = list(dict.fromkeys(LEAN_VOCAB + [l for l in f["vocab"] if l not in SKIP] + KITCHEN_DISTRACTORS))
        tkey = tuple(names)
        if tkey not in text_cache:
            with torch.inference_mode():
                ti = proc(text=[f"a photo of a {x}." for x in names], padding="max_length", max_length=64,
                          return_tensors="pt").to("cuda")
                te = emb(model.get_text_features(**ti))
                text_cache[tkey] = te / te.norm(dim=-1, keepdim=True)
        te = text_cache[tkey]
        crops = []
        for d in dets:
            x1, y1, x2, y2 = d["box_xyxy"]
            pw, ph = max((x2 - x1) * 0.15, 20), max((y2 - y1) * 0.15, 20)
            x1, y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
            x2, y2 = int(min(W, x2 + pw)), int(min(H, y2 + ph))
            crops.append(Image.fromarray(rgb[y1:y2, x1:x2]))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            ii = proc(images=crops, return_tensors="pt").to("cuda")
            ii["pixel_values"] = ii["pixel_values"].half()
            ie = emb(model.get_image_features(**ii))
            ie = ie / ie.norm(dim=-1, keepdim=True)
            probs = (100.0 * ie.float() @ te.float().T).softmax(-1).cpu().numpy()
        torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000)
        for d, p in zip(dets, probs):
            order = [names[i] for i in np.argsort(-p)]
            ok = lambda lab: lab == d["label"] or synonym(lab, d["label"])
            decisions[key(f["file"], d)] = {"top1": ok(order[0]), "top2": ok(order[0]) or ok(order[1]),
                                            "best": order[:3], "p_label": float(p[names.index(d["label"])]) if d["label"] in names else None}
            n += 1
    json.dump(decisions, open(args.out, "w"))
    print(json.dumps({"model": args.model, "verified": n, "ms_per_frame_median": round(float(np.median(times)), 1),
                      "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                      "kept_top1": sum(v["top1"] for v in decisions.values()), "kept_top2": sum(v["top2"] for v in decisions.values())}))


if __name__ == "__main__":
    main()
