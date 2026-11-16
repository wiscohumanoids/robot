"""Post-processing after detection + segmentation + 3D lifting (shared by the server and the offline benchmark,
so the benchmark measures exactly what the server does). Precision study 2026-11-10: docs/runs/2026-11-10_worklog_E.md.

  mask_nms        same label, mask IoU > `thr` -> keep the higher score (the base + tiny ensemble finds one object
                  twice with slightly different boxes; box NMS keeps both). IoU, not overlap-of-the-smaller: a big
                  "cabinet" mask over several units must not swallow a correct single-cabinet detection inside it.
  apply_priors    drop detections whose 3D box is implausible for the label (priors.py)
  CropVerifier    SigLIP 2 check of the crop: the label (or a near-synonym) must be in the top-3 names among the
                  frame's prompts + kitchen distractors. Only for object / appliance classes: large fixtures seen
                  in part are ambiguous as crops and are left to the priors (NO_VERIFY).
"""
from __future__ import annotations

import numpy as np

from . import priors

NO_VERIFY = {"cabinet", "drawer", "shelf", "counter", "fridge", "sink", "stove", "stovetop", "oven", "range hood",
             "dishwasher", "table"}
# names that mean the same thing for the robot (RoboCasa uses several); a verifier vote for a synonym counts
SYNONYMS = [{"cabinet", "drawer", "shelf"}, {"stove", "stovetop", "range hood", "oven"}, {"toaster oven", "microwave"},
            {"bottle", "juice", "bottled drink"}]


def synonym(a: str, b: str) -> bool:
    return a == b or any(a in g and b in g for g in SYNONYMS)


def mask_nms(dets: list[dict], thr: float = 0.3) -> list[dict]:
    """dets: dicts with "label", "score", "mask" (bool HxW). Returns the kept ones, input order preserved."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["score"])
    keep = []
    areas = [int(d["mask"].sum()) if d.get("mask") is not None else 0 for d in dets]
    for i in order:
        mi = dets[i].get("mask")
        dup = False
        if mi is not None and areas[i] > 0:
            for j in keep:
                if dets[j]["label"] != dets[i]["label"] or dets[j].get("mask") is None or areas[j] == 0:
                    continue
                inter = int(np.logical_and(mi, dets[j]["mask"]).sum())
                if inter / (areas[i] + areas[j] - inter) > thr:
                    dup = True
                    break
        if not dup:
            keep.append(i)
    keep = set(keep)
    return [d for k, d in enumerate(dets) if k in keep]


def apply_priors(dets: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (kept, rejected); rejected ones get a "rejected" reason."""
    kept, rej = [], []
    for d in dets:
        ok, why = priors.check(d["label"], d.get("centroid_map"), d.get("extent_map"))
        if ok:
            kept.append(d)
        else:
            rej.append({**d, "rejected": why})
    return kept, rej


class CropVerifier:
    def __init__(self, model_id="google/siglip2-base-patch16-224", device="cuda", topk=3):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.device, self.topk = torch, device, topk
        self.model_id = model_id
        self.proc = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.dtype = dtype
        self._text = {}

    @staticmethod
    def _emb(out):
        return out.pooler_output if hasattr(out, "pooler_output") else out

    def _text_emb(self, names: tuple):
        if names not in self._text:
            with self.torch.inference_mode():
                ti = self.proc(text=[f"a photo of a {x}." for x in names], padding="max_length", max_length=64,
                               return_tensors="pt").to(self.device)
                te = self._emb(self.model.get_text_features(**ti))
                self._text[names] = te / te.norm(dim=-1, keepdim=True)
            if len(self._text) > 64:
                self._text.pop(next(iter(self._text)))
        return self._text[names]

    def verify(self, rgb: np.ndarray, dets: list[dict], names: list[str]) -> tuple[list[dict], list[dict]]:
        """Check dets whose label is not in NO_VERIFY. Returns (kept, rejected)."""
        from PIL import Image
        todo = [i for i, d in enumerate(dets) if d["label"] not in NO_VERIFY]
        if not todo:
            return dets, []
        names = list(dict.fromkeys(list(names) + [dets[i]["label"] for i in todo]))
        H, W = rgb.shape[:2]
        crops = []
        for i in todo:
            x1, y1, x2, y2 = dets[i]["box_xyxy"]
            pw, ph = max((x2 - x1) * 0.15, 20), max((y2 - y1) * 0.15, 20)
            x1, y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
            x2, y2 = int(min(W, x2 + pw)), int(min(H, y2 + ph))
            crops.append(Image.fromarray(rgb[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]))
        te = self._text_emb(tuple(names))
        with self.torch.inference_mode():
            ii = self.proc(images=crops, return_tensors="pt").to(self.device)
            ii["pixel_values"] = ii["pixel_values"].to(self.dtype)
            ie = self._emb(self.model.get_image_features(**ii))
            ie = ie / ie.norm(dim=-1, keepdim=True)
            probs = (100.0 * ie.float() @ te.float().T).softmax(-1).cpu().numpy()
        bad = set()
        for i, p in zip(todo, probs):
            top = [names[k] for k in np.argsort(-p)[:self.topk]]
            dets[i]["verify_top"] = top
            if not any(synonym(t, dets[i]["label"]) for t in top):
                bad.add(i)
        kept = [d for k, d in enumerate(dets) if k not in bad]
        rej = [{**d, "rejected": f"crop check: looks like {d['verify_top'][0]}"} for k, d in enumerate(dets) if k in bad]
        return kept, rej
