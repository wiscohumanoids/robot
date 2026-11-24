"""Detection backends.

GroundedSam2Backend: Grounding DINO (open-vocabulary boxes from text) -> SAM 2.1 (masks from those boxes).
FakeBackend: no models; returns the ground-truth boxes/masks passed in the request (plumbing tests).

Prompting: Grounding DINO takes one text of "name. name. name." with a 256-token limit, and its
accuracy drops with long prompts, so labels are split into chunks of <= `chunk` names, run in batches of
<= `max_batch` (same image repeated). `priority` names (what the planner is looking for) each get a chunk
of their own. Returned phrases are mapped back to the requested label (exact match first, then the label
sharing the most words).
"""
from __future__ import annotations

import time

import numpy as np

from . import rle


def _norm(s: str) -> str:
    """Lowercase, underscores to spaces, single spaces."""
    return " ".join(s.lower().replace("_", " ").split())


def map_phrase(phrase: str, labels: list[str]) -> str | None:
    """The prompt label a detector phrase refers to, or None."""
    p = _norm(phrase)
    if not p:
        return None
    norm = {_norm(l): l for l in labels}
    if p in norm:
        return norm[p]
    pw = set(p.split())
    best, score = None, 0.0
    for nl, l in norm.items():
        lw = set(nl.split())
        s = len(pw & lw) / max(len(lw), 1)
        if nl in p or p in nl:
            s = max(s, 0.75)
        if s > score:
            best, score = l, s
    return best if score >= 0.5 else None


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Indices kept by non-maximum suppression at IoU `iou`."""
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        r = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[r, 0]); y1 = np.maximum(boxes[i, 1], boxes[r, 1])
        x2 = np.minimum(boxes[i, 2], boxes[r, 2]); y2 = np.minimum(boxes[i, 3], boxes[r, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[r, 2] - boxes[r, 0]) * (boxes[r, 3] - boxes[r, 1])
        order = r[inter / (a + b - inter + 1e-9) < iou]
    return keep


def iou_xyxy(a, b) -> float:
    """Intersection over union of two xyxy boxes."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / u) if u > 0 else 0.0


ABSORBERS = ["cabinet", "counter", "drawer", "wall", "floor"]

# Kitchen vocabulary used as DISTRACTORS: names the detector may assign instead of forcing a target name onto
# something else (dishwasher instead of "fridge", utensil holder instead of "cup"). Distractor detections are never
# returned; they suppress overlapping target detections with a lower score. FAMILIES orders all names so confusable
# classes land in the same prompt chunk and compete directly.
KITCHEN_DISTRACTORS = ["dishwasher", "oven", "stove", "stovetop", "range hood", "microwave", "toaster", "toaster oven",
                       "coffee machine", "kettle", "fridge", "cabinet", "drawer", "shelf", "counter", "sink", "faucet",
                       "window", "stool", "chair", "table", "knife block", "paper towel", "utensil holder", "drip tray",
                       "burner", "wall", "floor", "cutting board", "pot", "pan"]
FAMILIES = [
    ["fridge", "dishwasher", "oven", "microwave", "toaster oven", "toaster", "coffee machine", "drip tray", "kettle"],
    ["cup", "mug", "bowl", "plate", "glass", "utensil holder", "knife block", "paper towel", "bottle", "jar", "can"],
    ["stove", "stovetop", "burner", "range hood", "pot", "pan", "sink", "faucet"],
    ["cabinet", "drawer", "shelf", "counter", "table", "stool", "chair", "window", "wall", "floor", "cutting board"],
]
_FAMILY_RANK = {n: (i, j) for i, fam in enumerate(FAMILIES) for j, n in enumerate(fam)}


def _clash(a, b):
    """Names sharing a word ("oven" / "toaster oven", "stove" / "stovetop") must not share a prompt: Grounding DINO
    then returns the shorter sub-phrase for the longer object and the label maps wrongly (seen on the kitchen benchmark)."""
    wa, wb = set(_norm(a).split()), set(_norm(b).split())
    na, nb = _norm(a).replace(" ", ""), _norm(b).replace(" ", "")
    return bool(wa & wb) or na in nb or nb in na


def build_chunks(names, size):
    """Greedy: keep FAMILIES order, but put each name in the first chunk with room and no clashing name."""
    chunks = []
    for n in names:
        for c in chunks:
            if len(c) < size and not any(_clash(n, m) for m in c):
                c.append(n)
                break
        else:
            chunks.append([n])
    return chunks


def order_prompts(names):
    """Group confusable classes together (FAMILIES order); unknown names (fruit, food...) go with the containers."""
    def key(n):
        return _FAMILY_RANK.get(_norm(n), (1, 99))
    return sorted(dict.fromkeys(names), key=key)


class FakeBackend:
    """Detector stand-in that returns the ground truth sent in the request (for plumbing tests)."""
    name = "fake"

    def info(self) -> dict:
        """Backend name."""
        return {"backend": self.name}

    def detect(self, rgb, labels, gt=None, **_):
        """Detections for the requested labels from the request's ground truth."""
        t0 = time.perf_counter()
        want = {_norm(l) for l in labels}
        out = []
        for g in gt or []:
            if _norm(g["label"]) not in want:
                continue
            d = {"label": g["label"], "score": float(g.get("score", 0.9)), "box_xyxy": list(map(float, g["box_xyxy"]))}
            if "mask_rle" in g:
                d["mask"] = rle.decode(g["mask_rle"])
            else:
                m = np.zeros(rgb.shape[:2], bool)
                x1, y1, x2, y2 = [int(round(v)) for v in g["box_xyxy"]]
                m[max(y1, 0):y2, max(x1, 0):x2] = True
                d["mask"] = m
            out.append(d)
        ms = (time.perf_counter() - t0) * 1000
        return out, {"detect": ms, "segment": 0.0}


class GroundedSam2Backend:
    """Grounding DINO (one or several models, ensembled) for boxes, then SAM 2.1 for masks."""
    name = "grounded_sam2"

    def __init__(self, dino_id="IDEA-Research/grounding-dino-base,IDEA-Research/grounding-dino-tiny",
                 sam_id="facebook/sam2.1-hiera-base-plus",
                 device="cuda", chunk=12, dtype="fp16", max_batch=4):
        """Load the detector(s) and the segmenter on `device`."""
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model,
                                  Sam2Processor)
        self.torch = torch
        self.device = device
        self.chunk = chunk
        self.max_batch = max_batch
        self.priority_mode = "none"
        self.dino_id, self.sam_id = dino_id, sam_id
        self.dinos = []                                   # comma-separated ids = ensemble (boxes merged before NMS)
        for did in [d.strip() for d in dino_id.split(",") if d.strip()]:
            self.dinos.append((AutoProcessor.from_pretrained(did),
                               AutoModelForZeroShotObjectDetection.from_pretrained(did).to(device).eval()))
        self.dino_proc, self.dino = self.dinos[0]
        self.sam_proc = Sam2Processor.from_pretrained(sam_id)
        self.sam_dtype = torch.float16 if (dtype == "fp16" and device.startswith("cuda")) else torch.float32
        try:                                                    # transformers >= 5: dtype=
            self.sam = Sam2Model.from_pretrained(sam_id, dtype=self.sam_dtype)
        except TypeError:
            self.sam = Sam2Model.from_pretrained(sam_id, torch_dtype=self.sam_dtype)
        self.sam = self.sam.to(device).eval()

    def info(self) -> dict:
        """Backend name, models, device and peak GPU memory."""
        t = self.torch
        mem = t.cuda.max_memory_allocated() / 1e9 if self.device.startswith("cuda") else 0.0
        return {"backend": self.name, "dino": self.dino_id, "sam": self.sam_id, "device": self.device,
                "gpu_mem_peak_gb": round(mem, 2), "chunk": self.chunk}

    def detect(self, rgb, labels, box_threshold=0.30, text_threshold=0.25, nms_iou=0.6, max_dets=40, segment=True,
               priority=None, cross_label_iou=0.8, priority_mode=None, distractors=None, distractor_iou=0.6,
               chunk=None, distractor_mode="separate", **_):
        """Boxes for the labels (in prompt chunks, with priority passes), NMS, then masks."""
        torch = self.torch
        from PIL import Image
        img = Image.fromarray(rgb)
        H, W = rgb.shape[:2]
        mode = priority_mode or self.priority_mode
        prio = list(dict.fromkeys(l for l in (priority or []) if l.strip()))[:4]
        uniq = [l for l in dict.fromkeys(l for l in labels if l.strip()) if l not in prio]
        # priority names (what the planner is looking for right now) get a short pass of their own (more sensitive).
        # "contrast": with absorber names that soak up generic regions a lone name would be forced onto
        # (a single-name prompt hallucinates the object almost anywhere on the kitchen benchmark).
        if mode == "solo":
            pchunks = [[p] for p in prio]
        elif mode == "contrast":
            pchunks = [[p] + [a for a in ABSORBERS if a != p] for p in prio]
        else:
            pchunks, uniq = [], list(dict.fromkeys(prio + uniq))
        targets = set(uniq) | set(prio)
        distract = [x for x in dict.fromkeys(distractors or []) if x not in targets]
        cs = chunk or self.chunk
        if distract and distractor_mode == "incontext":   # confusable names compete in one chunk, no word clashes
            body = build_chunks(order_prompts(uniq + distract), cs)
        elif distract:                                     # targets chunked as without distractors; distractors apart
            body = [uniq[i:i + cs] for i in range(0, len(uniq), cs)] + build_chunks(order_prompts(distract), cs)
        else:
            body = [uniq[i:i + cs] for i in range(0, len(uniq), cs)]
        chunks = pchunks + body
        chunks = chunks or [[]]
        exempt = set(prio) if mode == "solo" else set()
        texts = [". ".join(_norm(l) for l in c) + "." for c in chunks]
        t0 = time.perf_counter()
        results = []                                       # (chunk, result) pairs, from every detector
        with torch.inference_mode():                       # at most `max_batch` prompt chunks per forward (GPU memory)
            for proc, model in self.dinos:
                for b in range(0, len(texts), self.max_batch):
                    tb = texts[b:b + self.max_batch]
                    inputs = proc(images=[img] * len(tb), text=tb, return_tensors="pt", padding=True).to(self.device)
                    outputs = model(**inputs)
                    rs = proc.post_process_grounded_object_detection(
                        outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
                        target_sizes=[(H, W)] * len(tb))
                    results += list(zip(chunks[b:b + self.max_batch], rs))
        boxes, scores, labs = [], [], []
        for c, r in results:
            phrases = r["text_labels"] if "text_labels" in r else r["labels"]
            for b, s, ph in zip(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), phrases):
                lab = map_phrase(ph if isinstance(ph, str) else str(ph), c)
                if lab is None:
                    continue
                boxes.append(b); scores.append(float(s)); labs.append(lab)
        dets = []
        if boxes:
            boxes, scores = np.array(boxes, np.float32), np.array(scores, np.float32)
            labs = np.array(labs, dtype=object)
            keep = []
            for l in dict.fromkeys(labs.tolist()):                 # per-label NMS
                idx = np.flatnonzero(labs == l)
                keep += [int(idx[k]) for k in nms(boxes[idx], scores[idx], nms_iou)]
            keep = sorted(keep, key=lambda i: -scores[i])
            # one object, two names (a cabinet door called "drawer" too): keep the higher score.
            # priority names are never suppressed by other names.
            final = []
            dset = set(distract)
            for i in keep:                                 # keep is sorted by score: earlier = stronger
                if labs[i] not in exempt and any(
                        iou_xyxy(boxes[i], boxes[j]) > (distractor_iou if labs[j] in dset else cross_label_iou)
                        for j in final):
                    continue
                final.append(i)
            # absorber names only exist to soak up generic regions: never report them unless actually requested
            requested = set(labels) | set(prio)
            keep = [i for i in final if labs[i] in requested][:max_dets]
            dets = [{"label": str(labs[i]), "score": float(scores[i]), "box_xyxy": boxes[i].tolist()} for i in keep]
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if dets and segment:
            with torch.inference_mode():
                sin = self.sam_proc(images=img, input_boxes=[[d["box_xyxy"] for d in dets]], return_tensors="pt").to(self.device)
                sin = {k: (v.to(self.sam_dtype) if hasattr(v, "dtype") and v.dtype.is_floating_point else v) for k, v in sin.items()}
                sout = self.sam(**sin, multimask_output=False)
                masks = self.sam_proc.post_process_masks(sout.pred_masks.float().cpu(), sin["original_sizes"].cpu())[0]
            masks = masks.numpy().astype(bool)                   # (n, 1, H, W) or (n, H, W)
            if masks.ndim == 4:
                masks = masks[:, 0]
            for d, m in zip(dets, masks):
                d["mask"] = m
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        return dets, {"detect": (t1 - t0) * 1000, "segment": (t2 - t1) * 1000}


class Sam3Backend:
    """SAM 3 (Meta, Nov 2025): one model, text prompt in, every matching instance's mask out ("promptable concept
    segmentation"). No separate detector: the label is the prompt.

    The image is encoded once per request; each requested label then gets its own text pass on those features
    (`get_vision_features` / `vision_embeds`, as in the transformers docs). `box_threshold` in a request is the
    instance score threshold here (default `default_box_threshold`, 0.5 as in the SAM 3 release).
    """
    name = "sam3"
    default_box_threshold = 0.5

    def __init__(self, model_id="facebook/sam3", device="cuda", dtype="bf16"):
        """Load SAM 3 on `device` (the weights are gated: the HF token must have been granted access)."""
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch = torch
        self.device = device
        self.model_id = model_id
        self.dtype = torch.bfloat16 if (dtype == "bf16" and device.startswith("cuda")) else torch.float32
        self.proc = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id, dtype=self.dtype).to(device).eval()

    def info(self) -> dict:
        """Backend name, model, device and peak GPU memory."""
        t = self.torch
        mem = t.cuda.max_memory_allocated() / 1e9 if self.device.startswith("cuda") else 0.0
        return {"backend": self.name, "model": self.model_id, "device": self.device,
                "dtype": str(self.dtype).replace("torch.", ""), "gpu_mem_peak_gb": round(mem, 2)}

    def _cast(self, inputs):
        return {k: (v.to(self.device, self.dtype) if v.dtype.is_floating_point else v.to(self.device))
                for k, v in inputs.items() if hasattr(v, "dtype")}

    def detect(self, rgb, labels, box_threshold=None, mask_threshold=0.5, max_dets=40, **_):
        """Instances of each label, with masks, boxes and scores."""
        torch = self.torch
        from PIL import Image
        img = Image.fromarray(rgb)
        H, W = rgb.shape[:2]
        thr = self.default_box_threshold if box_threshold is None else float(box_threshold)
        uniq = list(dict.fromkeys(n for n in labels if n.strip()))
        t0 = time.perf_counter()
        dets = []
        with torch.inference_mode():
            img_in = self.proc(images=img, return_tensors="pt")
            vision = self.model.get_vision_features(pixel_values=self._cast(img_in)["pixel_values"])
            for label in uniq:
                txt = self._cast(self.proc(text=_norm(label), return_tensors="pt"))
                out = self.model(vision_embeds=vision, input_ids=txt["input_ids"],
                                 attention_mask=txt.get("attention_mask"))
                r = self.proc.post_process_instance_segmentation(
                    out, threshold=thr, mask_threshold=mask_threshold, target_sizes=[(H, W)])[0]
                for m, b, s in zip(r["masks"].cpu().numpy(), r["boxes"].float().cpu().numpy(),
                                   r["scores"].float().cpu().numpy()):
                    dets.append({"label": label, "score": float(s), "box_xyxy": b.tolist(), "mask": m.astype(bool)})
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        dets = sorted(dets, key=lambda d: -d["score"])[:max_dets]
        return dets, {"detect": (t1 - t0) * 1000, "segment": 0.0}
