"""Frozen DINOv3 patch features, on this machine (no server).

The backbone is the one on the codex/dino-ik branch (`panthera_sim/perception/backbone.py`): DINOv3 ViT-S/16 from
timm (`vit_small_patch16_dinov3.lvd1689m`, pinned revision), frozen, ImageNet-normalized input, one 384-d feature per
16 x 16 pixel patch. Downloaded once from Hugging Face into the usual cache (~90 MB); runs on Apple's GPU (MPS), CUDA
or the CPU. A 960 x 720 overhead image takes about 40 ms on an M-series Mac.

Features are L2-normalized, so a dot product is a cosine similarity. That is all the few-shot finder needs
(`perception/dino_local.py`): a few feature vectors on an object, a few on the background, and every patch scored by
which it resembles more.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np

MODEL_NAME = 'vit_small_patch16_dinov3.lvd1689m'
MODEL_REVISION = '3bf4720a82ec2066db88137180ff1f83a675cef0'
PATCH = 16
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def default_device() -> str:
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class DinoFeatures:
    """The backbone and a small cache of recent images' features."""

    def __init__(self, device: str | None = None, cache: int = 8):
        import timm
        import torch
        from huggingface_hub import hf_hub_download
        self.torch = torch
        self.device = device or default_device()
        weights = hf_hub_download(repo_id=f'timm/{MODEL_NAME}', filename='model.safetensors', token=False,
                                  revision=MODEL_REVISION)
        model = timm.create_model(MODEL_NAME, pretrained=False, checkpoint_path=weights, dynamic_img_size=True)
        self.model = model.requires_grad_(False).eval().to(self.device)
        self.dim = int(model.num_features)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache

    def __call__(self, rgb: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """(gh, gw, C) unit feature vectors for an (H, W, 3) uint8 RGB image, one per 16-pixel patch (after resizing
        by `scale`; the image is cropped to a multiple of 16)."""
        key = hashlib.sha1(rgb.tobytes()[::97] + rgb.shape.__repr__().encode() + str(scale).encode()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        torch = self.torch
        img = rgb
        if scale != 1.0:
            import cv2
            img = cv2.resize(rgb, (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        h, w = (img.shape[0] // PATCH) * PATCH, (img.shape[1] // PATCH) * PATCH
        x = (img[:h, :w].astype(np.float32) / 255.0 - _MEAN) / _STD
        t = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device)
        with torch.no_grad():
            tokens = self.model.forward_features(t)[:, self.model.num_prefix_tokens:]
            f = tokens.reshape(1, h // PATCH, w // PATCH, -1)[0]
            f = torch.nn.functional.normalize(f, dim=-1)
        out = f.float().cpu().numpy()
        self._cache[key] = out
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out


_SHARED: DinoFeatures | None = None


def shared() -> DinoFeatures:
    """One backbone per process (loading it takes a second or two)."""
    global _SHARED
    if _SHARED is None:
        _SHARED = DinoFeatures()
    return _SHARED


def patch_of(uv, grid_shape) -> tuple[int, int]:
    """The (row, col) patch holding pixel (u, v) (features at scale 1)."""
    gh, gw = grid_shape
    return int(np.clip(uv[1] // PATCH, 0, gh - 1)), int(np.clip(uv[0] // PATCH, 0, gw - 1))


def upsample(score: np.ndarray, image_shape) -> np.ndarray:
    """A (gh, gw) patch map stretched over the image's pixels (bilinear; the cropped border repeats the edge)."""
    import cv2
    gh, gw = score.shape
    H, W = image_shape[:2]
    up = cv2.resize(score.astype(np.float32), (gw * PATCH, gh * PATCH), interpolation=cv2.INTER_LINEAR)
    out = np.empty((H, W), np.float32)
    out[:up.shape[0], :up.shape[1]] = up[:H, :W]
    if H > up.shape[0]:
        out[up.shape[0]:, :up.shape[1]] = up[-1:, :W]
    if W > up.shape[1]:
        out[:, up.shape[1]:] = out[:, up.shape[1] - 1:up.shape[1]]
    return out
