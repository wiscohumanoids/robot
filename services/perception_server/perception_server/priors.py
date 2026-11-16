"""3D size / placement priors per class: reject detections whose measured 3D box (mask + depth, map frame,
z up, floor at 0) is physically implausible for the label. Real-world kitchen dimensions, not values fitted
to the benchmark. Only classes listed here are checked; everything else passes.

Measured quantities (from geometry.lift; visible surface only, so upper bounds are the safe kind of rule):
  bottom = centre_z - extent_z / 2, top = centre_z + extent_z / 2, height = extent_z,
  width = max(extent_x, extent_y), size = max of all three.
"""

M = float("inf")

# label: (bottom_min, bottom_max, top_min, top_max, height_max, width_min, width_max)
PRIORS = {
    # appliances
    "fridge":         (-M, 0.35, 1.2, M, M, 0.45, M),       # stands on the floor, tall
    "dishwasher":     (-M, 0.30, -M, 1.0, M, 0.4, M),       # under the counter
    "oven":           (-M, M, -M, M, M, 0.5, M),            # at least 50 cm wide (a toaster oven is ~35-45 cm)
    "microwave":      (0.5, M, -M, M, 0.6, -M, 0.8),        # on a counter or above the range
    "toaster oven":   (0.6, M, -M, M, 0.45, -M, 0.65),
    "toaster":        (0.6, M, -M, M, 0.35, -M, 0.45),
    "coffee machine": (0.6, M, -M, M, 0.55, -M, 0.6),
    "kettle":         (0.6, M, -M, M, 0.35, -M, 0.35),
    "range hood":     (1.45, M, -M, M, M, 0.4, M),          # 65-75 cm above a ~0.9 m cooktop
    "stove":          (-M, M, 0.75, 1.2, M, 0.4, M),        # cooking surface at counter height
    "stovetop":       (-M, M, 0.75, 1.2, M, 0.4, M),
    "sink":           (0.45, M, 0.75, 1.2, 0.5, -M, M),     # basin set into a counter
    # furniture / surfaces
    "counter":        (-M, M, 0.5, 1.3, M, 0.3, M),         # its top is a work surface, never the floor
    "table":          (-M, M, 0.4, 1.3, M, 0.3, M),
    "stool":          (-M, 0.3, 0.3, 1.1, M, -M, M),
    "chair":          (-M, 0.3, 0.3, 1.3, M, -M, M),
    # tableware (hand-sized)
    "cup":            (-M, M, -M, M, 0.18, -M, 0.18),
    "mug":            (-M, M, -M, M, 0.16, -M, 0.18),
    "glass":          (-M, M, -M, M, 0.2, -M, 0.14),
    "bowl":           (-M, M, -M, M, 0.15, 0.06, 0.35),
    "plate":          (-M, M, -M, M, 0.07, 0.12, 0.38),     # flat
    "bottle":         (-M, M, -M, M, 0.45, -M, 0.16),
    "can":            (-M, M, -M, M, 0.25, -M, 0.14),
}


def check(label: str, centroid_map, extent_map, margin: float = 0.03):
    """(ok, reason). Unknown labels and detections without 3D always pass. `margin` (m) absorbs depth noise."""
    p = PRIORS.get(label)
    if p is None or centroid_map is None or extent_map is None:
        return True, ""
    bmin, bmax, tmin, tmax, hmax, wmin, wmax = p
    cz, ez = centroid_map[2], extent_map[2]
    bottom, top = cz - ez / 2, cz + ez / 2
    width = max(extent_map[0], extent_map[1])
    tests = (("bottom", bottom, bmin, bmax), ("top", top, tmin, tmax), ("height", ez, -M, hmax),
             ("width", width, wmin, wmax))
    for name, v, lo, hi in tests:
        if v < lo - margin or v > hi + margin:
            return False, f"{label}: {name} {v:.2f} m outside [{lo:.2f}, {hi:.2f}]"
    return True, ""
