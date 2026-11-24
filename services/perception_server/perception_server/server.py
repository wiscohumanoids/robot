"""HTTP perception server (FastAPI). Runs on the GPU VM at 127.0.0.1:8100; the Mac reaches it through
an SSH tunnel (ssh -N -L 8100:localhost:8100 hsim-gpu).

  GET  /health   {status, backend, models, gpu memory}
  POST /detect   see protocol.py

  python -m perception_server.server --backend real|sam3|fake [--dino ...] [--sam ...] [--sam3 ...] [--port 8100]

Backends: real = Grounding DINO + SAM 2.1; sam3 = SAM 3 alone (text prompt -> instance masks; run_sam3_server.sh
starts it on port 8101 beside the real one); fake = ground truth from the request.
"""
from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from . import geometry, postprocess, protocol, rle
from .models import KITCHEN_DISTRACTORS

STATE = {"backend": None, "verifier": None, "lock": threading.Lock(), "n": 0, "started": time.time(),
         "post": {"mask_nms": True, "priors": True, "verify": True}}


def run_detect(req: dict) -> dict:
    """Handle one /detect request: decode, detect, lift to 3D, post-process, encode the reply."""
    be = STATE["backend"]
    t0 = time.perf_counter()
    rgb = protocol.decode_rgb(req)
    depth = protocol.decode_depth(req) if req.get("depth_png16_b64") else None
    K = req.get("intrinsics") or {}
    K.setdefault("width", rgb.shape[1]); K.setdefault("height", rgb.shape[0])
    T = np.array(req["T_map_cam"], float).reshape(4, 4) if req.get("T_map_cam") else None
    t1 = time.perf_counter()
    kw = {k: req[k] for k in ("box_threshold", "text_threshold") if req.get(k) is not None}
    kw["priority"] = req.get("priority_prompts") or []
    for k in ("distractors", "distractor_iou", "chunk", "cross_label_iou", "distractor_mode"):
        if req.get(k) is not None:
            kw[k] = req[k]
    if "box_threshold" not in kw and getattr(be, "default_box_threshold", None):
        kw["box_threshold"] = be.default_box_threshold
    with STATE["lock"]:                                   # one GPU pass at a time
        dets, tm = be.detect(rgb, req.get("prompts") or [], gt=req.get("gt"), **kw)
    t2 = time.perf_counter()
    for d in dets:                                        # 3D from mask + depth
        m = d.get("mask")
        if m is not None and depth is not None and "fx" in K:
            g = geometry.lift(m, depth, K, T)
            if g:
                d.update({k: (np.round(v, 4).tolist() if isinstance(v, list) else v) for k, v in g.items()})
    t3 = time.perf_counter()
    post = {**STATE["post"], **(req.get("postprocess") or {})}
    rejected = []
    if post["mask_nms"]:
        dets = postprocess.mask_nms(dets)
    if post["priors"]:
        dets, rej = postprocess.apply_priors(dets); rejected += rej
    if post["verify"] and STATE["verifier"] is not None and dets:
        names = list(req.get("prompts") or []) + list(req.get("distractors") or []) + KITCHEN_DISTRACTORS
        with STATE["lock"]:
            dets, rej = STATE["verifier"].verify(rgb, dets, names)
        rejected += rej
    t4 = time.perf_counter()
    want_masks = req.get("return_masks", True)

    def pack(d):
        o = {"label": d["label"], "score": round(d["score"], 4), "box_xyxy": [round(v, 1) for v in d["box_xyxy"]]}
        m = d.get("mask")
        if m is not None:
            o["mask_area"] = int(m.sum())
            if want_masks:
                o["mask_rle"] = rle.encode(m)
        for k in ("n_points", "centroid_cam", "extent_cam", "depth_median", "centroid_map", "extent_map",
                  "centroid_cam_completed", "completed", "rejected", "verify_top"):
            if k in d:
                o[k] = d[k]
        o.setdefault("n_points", 0)
        return o
    out = [pack(d) for d in dets]
    STATE["n"] += 1
    t5 = time.perf_counter()
    resp = {"detections": out, "backend": be.name, "image_size": list(rgb.shape[:2]), "postprocess": post,
            "frame_id": req.get("frame_id"), "stamp": req.get("stamp"),
            "timing_ms": {"decode": round((t1 - t0) * 1000, 1), "detect": round(tm["detect"], 1),
                          "segment": round(tm["segment"], 1), "lift": round((t3 - t2) * 1000, 1),
                          "postprocess": round((t4 - t3) * 1000, 1), "total": round((t5 - t0) * 1000, 1)}}
    if req.get("return_rejected"):
        resp["rejected"] = [pack(d) for d in rejected]
    else:
        resp["n_rejected"] = len(rejected)
    return resp


def make_app():
    """FastAPI app with /health and /detect."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="hsim perception server")

    @app.get("/health")
    def health():
        be = STATE["backend"]
        return {"status": "ok" if be else "loading", "requests": STATE["n"],
                "uptime_s": round(time.time() - STATE["started"], 1), **(be.info() if be else {})}

    @app.post("/detect")
    def detect(req: dict):
        if STATE["backend"] is None:
            raise HTTPException(503, "models loading")
        try:
            return run_detect(req)
        except KeyError as e:
            raise HTTPException(400, f"missing field {e}")

    return app


def load_backend(args):
    """The fake or the real detection backend, with the post-processing options."""
    from .models import FakeBackend, GroundedSam2Backend, Sam3Backend
    if args.backend == "fake":
        STATE["post"] = {"mask_nms": True, "priors": True, "verify": False}
        return FakeBackend()
    if args.backend == "sam3":
        be = Sam3Backend(model_id=args.sam3, device=args.device)
    else:
        be = GroundedSam2Backend(dino_id=args.dino, sam_id=args.sam, device=args.device, chunk=args.chunk)
        be.default_box_threshold = 0.30
    if args.box_threshold is not None:
        be.default_box_threshold = args.box_threshold
    STATE["post"] = {"mask_nms": not args.no_mask_nms, "priors": not args.no_priors, "verify": args.verifier != "none"}
    if args.verifier != "none":
        STATE["verifier"] = postprocess.CropVerifier(args.verifier, device=args.device)
    be.priority_mode = args.priority_mode
    # warm-up: first CUDA pass compiles kernels / allocates
    be.detect(np.zeros((320, 640, 3), np.uint8), ["cup", "plate"])
    return be


def main():
    """Parse the options, load the backend and serve on localhost."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["real", "sam3", "fake"], default="real")
    # Defaults chosen on a kitchen benchmark: base + tiny ensemble at 0.30
    # finds fridges (8/13 views vs 2/13 for base alone) and small objects (45 %) at 45 % precision, ~0.6 s/frame.
    ap.add_argument("--dino", default="IDEA-Research/grounding-dino-base,IDEA-Research/grounding-dino-tiny",
                    help="one id, or comma-separated ids for an ensemble")
    ap.add_argument("--sam", default="facebook/sam2.1-hiera-base-plus")
    ap.add_argument("--sam3", default="facebook/sam3", help="SAM 3 weights (gated on Hugging Face) for --backend sam3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--box-threshold", type=float, default=None,
                    help="default when a request does not set one (real: 0.30, the DINO box score; sam3: 0.5, the "
                         "instance score)")
    ap.add_argument("--no-mask-nms", action="store_true")
    ap.add_argument("--no-priors", action="store_true", help="disable the 3D size/placement priors")
    ap.add_argument("--verifier", default="google/siglip2-base-patch16-224",
                    help="SigLIP 2 crop verifier for object/appliance classes; 'none' to disable")
    ap.add_argument("--priority-mode", choices=["none", "contrast", "solo"], default="none",
                    help="none: priority names just join the prompt list. solo/contrast: extra detector passes, both "
                         "hallucinate or suppress the target in the bench, kept for experiments")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    import uvicorn
    app = make_app()
    STATE["backend"] = load_backend(args)
    print(f"[perception] backend ready: {STATE['backend'].info()}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
