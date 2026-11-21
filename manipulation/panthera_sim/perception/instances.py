"""Measure visible connected block regions and stud centers in semantic masks."""

import numpy as np
from PIL import ImageDraw
from scipy import ndimage


PLASTIC_COLORS = ('red', 'orange', 'yellow', 'green', 'blue', 'purple')


def plastic_colors(rgb):
    """Return hue category indices and a saturation mask from an RGB image."""
    scale = np.sqrt(rgb.shape[0] * rgb.shape[1] / 512 ** 2)
    median_size = max(1, 2 * round(scale) + 1)
    color = ndimage.median_filter(rgb.astype(float), size=(median_size, median_size, 1)) / 255
    maximum, minimum = color.max(-1), color.min(-1)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    red = (maximum == color[..., 0]) & (delta > 0)
    green = (maximum == color[..., 1]) & (delta > 0) & ~red
    blue = (maximum == color[..., 2]) & (delta > 0) & ~red & ~green
    hue[red] = ((color[..., 1] - color[..., 2])[red] / delta[red]) % 6
    hue[green] = (color[..., 2] - color[..., 0])[green] / delta[green] + 2
    hue[blue] = (color[..., 0] - color[..., 1])[blue] / delta[blue] + 4
    categories = np.digitize(hue * 60, [8, 30, 80, 170, 255, 330]) % 6
    saturated = delta / np.maximum(maximum, 1e-6) > .2
    return categories, saturated


def instance_regions(mask, rgb=None):
    """Label visible regions, optionally splitting touching plastic colors.

    RGB splitting uses hue categories for red, orange, yellow, green, blue and
    purple plastic. Saturated color patches seed separate regions; nearby pixels
    inherit the closest seed. Same-color touching blocks remain ambiguous.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 1, 2]).all():
        raise ValueError('Expected a 2-D semantic mask with labels 0, 1 and 2')
    components, _ = ndimage.label(mask > 0, structure=np.ones((3, 3)))
    if rgb is None:
        return components
    rgb = np.asarray(rgb)
    if rgb.shape != (*mask.shape, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected a uint8 RGB image aligned with the semantic mask')
    categories, saturated = plastic_colors(rgb)
    return split_color_regions(components, categories, saturated)


def split_color_regions(components, categories, saturated):
    """Split connected foreground labels using the frame's plastic color map."""
    scale = np.sqrt(components.size / 512 ** 2)
    # Scale the minimum saturated color patch with image resolution.
    minimum_area = max(1, round(80 * scale ** 2))
    regions = np.zeros(components.shape, dtype=np.int32)
    next_id = 1
    for component_id, slices in enumerate(ndimage.find_objects(components), start=1):
        if slices is None:
            continue
        region = components[slices] == component_id
        seeds = np.zeros(region.shape, dtype=np.int32)
        for category in range(len(PLASTIC_COLORS)):
            parts, _ = ndimage.label(region & (categories[slices] == category) & saturated[slices],
                                    structure=np.ones((3, 3)))
            areas = np.bincount(parts.ravel())
            for part in np.flatnonzero(areas >= minimum_area):
                if part == 0:
                    continue
                seeds[parts == part] = next_id
                next_id += 1
        if not seeds.any():
            regions[slices][region] = next_id
            next_id += 1
            continue
        indices = ndimage.distance_transform_edt(seeds == 0, return_distances=False, return_indices=True)
        assigned = seeds[tuple(indices)]
        regions[slices][region] = assigned[region]
    return regions


def extract_instances(mask, rgb=None):
    """Return connected foreground regions in original-image pixel coordinates.

    Labels are background=0, body=1 and stud=2. Without RGB, eight-connected
    foreground components define regions. RGB separates different plastic colors
    within a component. Same-color touching blocks can merge and occlusion can
    split one block. Studs are eight-connected label-2 components in each region.
    Bounding boxes use [left, top, right, bottom] with exclusive right and bottom
    edges. Centroids average integer pixel coordinates (x, y).
    """
    return measure_instances(mask, instance_regions(mask, rgb))


def measure_instances(mask, regions):
    """Measure boxes and stud centers from an existing instance-ID raster.

    Region IDs are preserved so callers can associate landmarks with the same
    raster without recomputing color splits.
    """
    mask, regions = np.asarray(mask), np.asarray(regions)
    if mask.shape != regions.shape or mask.ndim != 2:
        raise ValueError('Semantic and instance masks must be aligned 2-D arrays')
    objects = []
    for region_id, slices in enumerate(ndimage.find_objects(regions), start=1):
        if slices is None:
            continue
        ys, xs = slices
        region = regions[slices] == region_id
        studs, _ = ndimage.label((mask[slices] == 2) & region,
                                structure=np.ones((3, 3)))
        centers = []
        for stud_id, stud_slices in enumerate(ndimage.find_objects(studs), start=1):
            if stud_slices is None:
                continue
            sy, sx = np.nonzero(studs[stud_slices] == stud_id)
            sy = sy + stud_slices[0].start + ys.start
            sx = sx + stud_slices[1].start + xs.start
            centers.append({'id': stud_id, 'centroid_xy': [float(sx.mean()), float(sy.mean())],
                            'area_pixels': int(sx.size)})
        objects.append({'id': region_id,
                        'bbox_xyxy': [xs.start, ys.start, xs.stop, ys.stop],
                        'area_pixels': int(region.sum()), 'studs': centers})
    return objects


def annotate_instances(image, objects):
    """Return a PIL image copy with region boxes and stud centroid crosses."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for obj in objects:
        x0, y0, x1, y1 = obj['bbox_xyxy']
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline='yellow', width=2)
        draw.text((x0 + 2, y0 + 2), str(obj['id']), fill='yellow')
        for stud in obj['studs']:
            x, y = stud['centroid_xy']
            draw.line((x - 3, y, x + 3, y), fill='white', width=1)
            draw.line((x, y - 3, x, y + 3), fill='white', width=1)
    return annotated
