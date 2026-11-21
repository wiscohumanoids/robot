"""Dense semantic decoder for frozen DINOv3 patch features and RGB boundaries."""
import torch
from torch import nn
from torch.nn import functional as F


class DenseAdapter(nn.Module):
    def __init__(self, feature_dim=384, num_classes=3):
        super().__init__()
        self.tokens = nn.Sequential(nn.Conv2d(feature_dim, 64, 1), nn.GELU())
        self.rgb = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(24, 24, 3, padding=1), nn.GELU())
        self.decode = nn.Sequential(nn.Conv2d(88, 64, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(32, num_classes, 1))

    def forward(self, features, rgb):
        # RGB provides sub-patch boundaries; DINO features supply scene context.
        size = (rgb.shape[-2] // 2, rgb.shape[-1] // 2)
        context = F.interpolate(self.tokens(features), size=size, mode='bilinear', align_corners=False)
        detail = self.rgb(F.interpolate(rgb, size=size, mode='bilinear', align_corners=False))
        logits = self.decode(torch.cat((context, detail), dim=1))
        return F.interpolate(logits, size=rgb.shape[-2:], mode='bilinear', align_corners=False)


def patch_features(backbone, images):
    tokens = backbone.forward_features(images)
    if isinstance(tokens, dict):
        tokens = tokens['x_norm_patchtokens']
    else:
        tokens = tokens[:, backbone.num_prefix_tokens:]
    return tokens.transpose(1, 2).reshape(images.shape[0], -1, images.shape[-2] // 16, images.shape[-1] // 16)


def normalize(images):
    return (images - images.new_tensor([.485, .456, .406])[None, :, None, None]) / images.new_tensor([.229, .224, .225])[None, :, None, None]
