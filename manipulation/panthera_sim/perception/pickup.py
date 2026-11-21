"""Locate Mega Bloks in overhead and wrist RGB frames with the trained DINO head."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .instances import PLASTIC_COLORS, instance_regions, measure_instances, plastic_colors, split_color_regions


@dataclass(frozen=True)
class BlockDetection:
    """A visible block region in original-image pixels, with x right and y down.

    The pickup point is the mean of the visible stud centers when available.
    Otherwise it is the visible region centroid. Stud points represent visible
    pixel centroids. A body corner can belong to the top or bottom face.
    The bounding box excludes the right and bottom edges.
    """

    xy: tuple[float, float]
    color: str | None
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    studs_xy: tuple[tuple[float, float], ...]
    area_pixels: int
    corners_xy: tuple[tuple[float, float], ...] = ()


def detect_blocks(
    rgb: np.ndarray,
    probabilities: np.ndarray,
    *,
    color: str | None = None,
    corners_xy=(),
    stud_centers_xy=(),
) -> list[BlockDetection]:
    """Measure blocks from background, body and stud probabilities in that order."""
    if color is not None and color not in PLASTIC_COLORS:
        raise ValueError(f"Unknown block color {color!r}. Choose from {', '.join(PLASTIC_COLORS)}.")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or rgb.size == 0:
        raise ValueError('Expected a nonempty uint8 H×W×3 RGB image')
    if probabilities.shape != (3, *rgb.shape[:2]):
        raise ValueError('Expected three semantic probability maps aligned with RGB')
    mask = probabilities.argmax(axis=0).astype(np.uint8)
    colors, saturated = plastic_colors(rgb)
    regions = split_color_regions(instance_regions(mask), colors, saturated)
    corners_by_region = _associate_points(corners_xy, regions)
    studs_by_region = _associate_points(stud_centers_xy, regions)
    foreground = probabilities[1] + probabilities[2]
    minimum_area = max(6, round(mask.size * 0.00015))
    minimum_stud_area = max(3, round(mask.size * 0.00003))
    detections = []
    for region in measure_instances(mask, regions):
        area = region['area_pixels']
        if area < minimum_area:
            continue
        region_mask = regions == region['id']
        hue_votes = np.bincount(colors[region_mask & saturated], minlength=len(PLASTIC_COLORS))
        block_color = PLASTIC_COLORS[hue_votes.argmax()] if hue_votes.max() >= area * 0.2 else None
        if color is not None and block_color != color:
            continue
        studs = tuple(studs_by_region.get(region['id'], ()))
        if not studs:
            studs = tuple(tuple(stud['centroid_xy']) for stud in region['studs']
                          if stud['area_pixels'] >= minimum_stud_area)
        if studs:
            xy = tuple(np.mean(studs, axis=0))
        else:
            ys, xs = np.nonzero(region_mask)
            xy = (float(xs.mean()), float(ys.mean()))
        detections.append(BlockDetection(
            xy=tuple(map(float, xy)),
            color=block_color,
            confidence=float(foreground[region_mask].mean()),
            bbox_xyxy=tuple(region['bbox_xyxy']),
            studs_xy=studs,
            area_pixels=area,
            corners_xy=tuple(corners_by_region.get(region['id'], ())),
        ))
    return sorted(detections, key=lambda detection: detection.confidence * detection.area_pixels,
                  reverse=True)


def _associate_points(points, regions):
    by_region = {}
    for x, y in points:
        ix, iy = round(x), round(y)
        neighbors = regions[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4]
        region_ids = np.unique(neighbors[neighbors > 0])
        if len(region_ids) == 1:
            by_region.setdefault(int(region_ids[0]), []).append((x, y))
    return by_region


def select_block(detections, *, near_xy=None, max_distance=None):
    """Select a ranked detection or the detection closest to a predicted pixel."""
    if max_distance is not None and (
        near_xy is None or not np.isfinite(max_distance) or max_distance < 0
    ):
        raise ValueError('A finite nonnegative pixel distance requires near_xy')
    if not detections:
        return None
    if near_xy is None:
        return detections[0]
    xy = np.asarray(near_xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError('near_xy must contain two finite pixel coordinates')
    positions = np.array([detection.xy for detection in detections])
    distances = np.linalg.norm(positions - xy, axis=1)
    index = int(distances.argmin())
    if max_distance is not None and distances[index] > max_distance:
        return None
    return detections[index]


class DinoPickupPerception:
    """Load one frozen encoder and semantic head for repeated RGB inference."""

    def __init__(
        self, checkpoint: str | Path, *, landmark_checkpoint=None, device=None, input_size=None,
    ):
        import torch

        from .adapter import DenseAdapter
        from .backbone import MODEL_NAME, MODEL_REVISION, load_backbone

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if weights['backbone'] != MODEL_NAME or weights['backbone_revision'] != MODEL_REVISION:
            raise ValueError('The checkpoint backbone name or revision does not match the loader')
        if list(weights['classes']) not in (
            ['background', 'body', 'stud'], ['background', 'block_body', 'stud']
        ):
            raise ValueError('Pickup requires a Mega Bloks body and stud checkpoint')
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = tuple(weights['input_size'] if input_size is None else input_size)
        if len(self.input_size) != 2 or any(
            size <= 0 or size % 16 for size in self.input_size
        ):
            raise ValueError('Input height and width must be positive multiples of 16')
        self.model = DenseAdapter(weights['feature_dim'], len(weights['classes']))
        self.model.load_state_dict(weights['adapter'])
        self.model = self.model.to(self.device).eval()
        self.landmark_model = None
        landmark_path = (Path(landmark_checkpoint) if landmark_checkpoint is not None
                         else Path(checkpoint).with_name('landmarks.pt'))
        if landmark_checkpoint is not None or landmark_path.is_file():
            from .landmarks import LandmarkHead

            landmarks = torch.load(landmark_path, map_location='cpu', weights_only=True)
            if (landmarks['backbone'] != MODEL_NAME
                    or landmarks.get('backbone_revision', MODEL_REVISION) != MODEL_REVISION):
                raise ValueError('The landmark checkpoint backbone name or revision does not match the loader')
            self.landmark_model = LandmarkHead(landmarks['feature_dim'])
            self.landmark_model.load_state_dict(landmarks['head'])
            self.landmark_model = self.landmark_model.to(self.device).eval()
            self.landmark_thresholds = landmarks['thresholds']
        self.backbone = load_backbone().to(self.device).eval()

    def describe_surface(self, rgb: np.ndarray) -> np.ndarray:
        """Return normalized spatial DINOv3 descriptors for aligned RGB crops."""
        import torch
        from torch.nn import functional as F

        from .adapter import normalize

        image = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(self.device).float() / 255
        with torch.inference_mode():
            # Early blocks retain local boundaries that distinguish nearby
            # depth hypotheses in these aligned crops.
            features, = self.backbone.forward_intermediates(
                normalize(image), indices=(2,), stop_early=True, intermediates_only=True)
            return F.normalize(features, dim=1).permute(0, 2, 3, 1).cpu().numpy()

    def _predict_frame(self, rgb: np.ndarray, *, include_landmarks: bool):
        import torch
        from torch.nn import functional as F

        from .adapter import normalize, patch_features

        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or rgb.size == 0:
            raise ValueError('Expected a nonempty uint8 H×W×3 RGB image')
        resized = Image.fromarray(rgb).resize(self.input_size[::-1], Image.Resampling.BILINEAR)
        image = torch.from_numpy(np.array(resized)).permute(2, 0, 1)[None].to(self.device).float() / 255
        with torch.inference_mode():
            features = patch_features(self.backbone, normalize(image))
            logits = self.model(features, image)
            logits = F.interpolate(logits, size=rgb.shape[:2], mode='bilinear', align_corners=False)
            probabilities = logits.softmax(1)[0].cpu().numpy()
            corners, studs = (), ()
            if include_landmarks and self.landmark_model is not None:
                from .landmarks import decode_landmarks

                points = decode_landmarks(self.landmark_model(features, image)[0],
                                          self.landmark_thresholds)
                height, width = rgb.shape[:2]
                network_height, network_width = self.input_size

                def resize_points(key):
                    return tuple(
                        ((x + .5) * width / network_width - .5,
                         (y + .5) * height / network_height - .5)
                        for x, y in (point['xy'] for point in points[key])
                    )

                corners, studs = resize_points('corners'), resize_points('stud_centers')
        return probabilities, corners, studs

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Return background, body and stud probabilities at the input resolution."""
        probabilities, _, _ = self._predict_frame(rgb, include_landmarks=False)
        return probabilities

    def detect(self, rgb: np.ndarray, *, color: str | None = None) -> list[BlockDetection]:
        """Rank visible blocks by foreground confidence weighted by region area."""
        probabilities, corners, studs = self._predict_frame(rgb, include_landmarks=True)
        return detect_blocks(rgb, probabilities, color=color, corners_xy=corners, stud_centers_xy=studs)

    def select(self, rgb, *, color=None, near_xy=None, max_distance=None):
        """Use a requested color and optional projected pixel to associate a target."""
        return select_block(self.detect(rgb, color=color), near_xy=near_xy, max_distance=max_distance)
