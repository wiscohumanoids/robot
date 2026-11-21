"""Plan a tabletop row and a tower in the coordinate frame of a two-stud block."""

from dataclasses import dataclass
from typing import Hashable

import numpy as np

from .mega_bloks import PITCH_M
from .mega_bloks_stack import STACK_CENTER_GAP_M


def _frame(studs, reference=None):
    row = np.asarray(studs, dtype=float)
    if row.shape != (2, 3) or not np.isfinite(row).all():
        raise ValueError("A grid block needs two finite world-space stud centers")
    axis = row[1, :2] - row[0, :2]
    length = np.linalg.norm(axis)
    if abs(length - PITCH_M) > .010:
        raise ValueError("The two studs must be one pitch apart")
    axis /= length
    # Studs are interchangeable. Preserve the initial axis sign even if the
    # detector returns them in the opposite order on a later frame.
    direction = axis[np.argmax(np.abs(axis))] if reference is None else axis @ reference
    if direction < 0:
        axis = -axis
    return row.mean(axis=0), axis


def grid_coordinates(stud_rows, anchor):
    """Express measured centers in anchor pitches and layer heights, before rounding."""
    origin, axis = _frame(stud_rows[anchor])
    across = np.array([-axis[1], axis[0]])
    coordinates = {}
    for block, row in stud_rows.items():
        center, _ = _frame(row)
        delta = center - origin
        coordinates[block] = [float(delta[:2] @ axis / PITCH_M),
                              float(delta[:2] @ across / PITCH_M),
                              float(delta[2] / STACK_CENTER_GAP_M)]
    return coordinates


@dataclass(frozen=True)
class GridPlacement:
    source: Hashable
    support: Hashable | None
    cell: tuple[int, int, int]
    axis_xy: tuple[float, float]

    def target_studs(self, anchor_studs, support_studs=None):
        """Return seated stud centers, using the measured support for the next layer."""
        x, y, level = self.cell
        if level and support_studs is not None:
            center, axis = _frame(support_studs, self.axis_xy)
            center[2] += STACK_CENTER_GAP_M
        else:
            center, axis = _frame(anchor_studs, self.axis_xy)
            across = np.array([-axis[1], axis[0]])
            center[:2] += PITCH_M * (x * axis + y * across)
            center[2] += level * STACK_CENTER_GAP_M
        half_step = np.r_[axis * PITCH_M / 2, 0.]
        return np.stack((center - half_step, center + half_step))


def _overlap(center, axis, other_center, other_axis):
    # Project both 2-by-1 footprints onto each rectangle's two edge normals.
    bases = [np.array([a, [-a[1], a[0]]]) for a in (axis, other_axis)]
    half_size = np.array([PITCH_M, PITCH_M / 2])
    for normal in np.concatenate(bases):
        radii = sum(np.abs(basis @ normal) @ half_size for basis in bases)
        if abs((center - other_center) @ normal) >= radii + .002:
            return False
    return True


def plan_grid(stud_rows, anchor, *, base_count=2, first_source=None, reachable=None):
    """Order every loose block into a straight base and a tower above the anchor.

    IDs are keys of stud_rows, which holds measured pairs of XYZ stud centers.
    base_count includes the stationary anchor and leaves at least one block for
    the tower. A requested first_source is used for the first transfer. The row
    runs along or across the anchor, on integer stud pitches.
    """
    if anchor not in stud_rows or len(stud_rows) < 2:
        raise ValueError("Choose a known anchor and at least one other block")
    if first_source is not None and (first_source not in stud_rows or first_source == anchor):
        raise ValueError("The first source must be a known block distinct from the anchor")
    if not isinstance(base_count, int) or base_count < 1:
        raise ValueError("base_count must be a positive integer")
    base_count = min(base_count, len(stud_rows) - 1)
    frames = {key: _frame(row) for key, row in stud_rows.items()}
    origin, axis = frames[anchor]
    across = np.array([-axis[1], axis[0]])
    layouts = []
    # An end-to-end base leaves the long sides of the anchor clear for the
    # gripper. Try a side-by-side base if both ends are obstructed.
    for dx, dy in ((2, 0), (-2, 0), (0, 1), (0, -1)):
        remaining = [key for key in frames if key != anchor]
        placements = []
        travel = 0.
        close_neighbors = 0
        support = anchor
        for i in range(1, len(frames)):
            tabletop = i < base_count
            cell = (dx * i, dy * i, 0) if tabletop else (0, 0, i - base_count + 1)
            xy = origin[:2] + PITCH_M * (cell[0] * axis + cell[1] * across)
            choices = remaining
            if i == 1 and first_source is not None:
                choices = [first_source]
            if tabletop:
                occupied = [key for key in remaining
                            if _overlap(xy, axis, frames[key][0][:2], frames[key][1])]
                # A block may be picked from its own destination.
                choices = [key for key in choices if all(key == other for other in occupied)]
            if not choices:
                break
            # The tilted open jaws sweep beyond the block footprint on descent.
            # Picking a nearby loose block first also removes it from that sweep.
            nearby = {key for key in remaining if np.linalg.norm(xy - frames[key][0][:2]) < .080}
            source = min(choices, key=lambda key: (len(nearby - {key}),
                                                   np.linalg.norm(xy - frames[key][0][:2])))
            close_neighbors += len(nearby - {source})
            travel += np.linalg.norm(xy - frames[source][0][:2])
            placements.append(GridPlacement(source, None if tabletop else support, cell, tuple(axis)))
            remaining.remove(source)
            if not tabletop:
                support = source
        else:
            cost = (close_neighbors, bool(dy), travel)
            layouts.append((cost, tuple(placements)))
            if base_count == 1:
                break
    for _, layout in sorted(layouts, key=lambda item: item[0]):
        if reachable is None or all(reachable(step.source, step.target_studs(stud_rows[anchor]))
                                    for step in layout):
            return layout
    raise ValueError("No clear, reachable base row and tower")
