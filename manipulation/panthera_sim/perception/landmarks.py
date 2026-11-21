"""Learn visible block corners, physical stud centers, and cuboid body edges."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LandmarkHead(nn.Module):
    def __init__(self, feature_dim=384):
        super().__init__()
        self.context = nn.Sequential(nn.Conv2d(feature_dim, 48, 1), nn.GELU())
        self.detail = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.GELU(),
        )
        self.decode = nn.Sequential(
            nn.Conv2d(72, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 24, 3, padding=1),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(27, 24, 3, padding=1), nn.GELU(), nn.Conv2d(24, 3, 1)
        )
        nn.init.constant_(self.refine[-1].bias, -2.19)
        with torch.no_grad():
            self.refine[-1].bias[:2].fill_(-4.6)

    def forward(self, features, rgb):
        size = (rgb.shape[-2] // 2, rgb.shape[-1] // 2)
        context = F.interpolate(
            self.context(features), size, mode="bilinear", align_corners=False
        )
        detail = self.detail(
            F.interpolate(rgb, size, mode="bilinear", align_corners=False)
        )
        decoded = F.interpolate(
            self.decode(torch.cat((context, detail), 1)),
            rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(torch.cat((decoded, rgb), 1))


def decode_landmarks(logits, thresholds=None):
    """Decode one image with 7-pixel local-maximum suppression for point heads."""
    thresholds = thresholds or {"corner": 0.2, "stud_center": 0.2, "body_edge": 0.5}
    probabilities = logits.detach().sigmoid()
    result = {}
    for channel, name, key in [
        (0, "corner", "corners"),
        (1, "stud_center", "stud_centers"),
    ]:
        heat = probabilities[channel]
        peaks = (heat == F.max_pool2d(heat[None, None], 7, 1, 3)[0, 0]) & (
            heat >= thresholds[name]
        )
        ys, xs = peaks.nonzero(as_tuple=True)
        order = heat[ys, xs].argsort(descending=True)[:128]
        result[key] = [
            {"xy": [float(xs[i]), float(ys[i])], "score": float(heat[ys[i], xs[i]])}
            for i in order
        ]
    result["edges"] = (probabilities[2] >= thresholds["body_edge"]).cpu().numpy()
    return result
