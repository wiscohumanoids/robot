"""Place socketed stud rows on a lattice defined by a stationary block."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Hashable

import numpy as np

from .mega_bloks import PITCH_M
from .mega_bloks_stack import STACK_CENTER_GAP_M


_DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
_PRESETS = {
    "castle": [(0, 0, 0, 2, 0), (0, 3, 0, 2, 0), (0, 0, 1, 4, 1), (0, 1, 2, 2, 1)],
    "pyramid": [(0, 0, 0, 2, 0), (2, 0, 0, 2, 0), (1, 0, 1, 2, 0)],
    "one-stud": [(0, 0, 0, 2, 0), (0, -1, 1, 3, 1)],
}


@dataclass(frozen=True)
class _Slot:
    cell: tuple[int, int, int]
    length: int
    turn: int

    def footprint(self):
        x, y, z = self.cell
        dx, dy = _DIRECTIONS[self.turn]
        return [(x + i * dx, y + i * dy, z) for i in range(self.length)]


def _slots(spec):
    if isinstance(spec, str):
        if spec not in _PRESETS:
            raise ValueError(f"Unknown blueprint {spec!r}")
        return [_Slot((x, y, z), n, t) for x, y, z, n, t in _PRESETS[spec]]
    if not isinstance(spec, dict) or not isinstance(spec.get("blocks"), list):
        raise ValueError("A blueprint needs a blocks array")
    slots = []
    for entry in spec["blocks"]:
        if not isinstance(entry, dict):
            raise ValueError("Each blueprint block must be an object")
        cell = entry.get("cell")
        n, turn = entry.get("length"), entry.get("turn", 0)
        if (not isinstance(cell, (list, tuple)) or len(cell) != 3
                or any(type(v) is not int for v in cell) or cell[2] < 0):
            raise ValueError("Each cell needs integer x, y and a nonnegative level")
        if type(n) is not int or not 2 <= n <= 4 or type(turn) is not int or not 0 <= turn < 4:
            raise ValueError("Each block needs a length from 2 to 4 and a turn from 0 to 3")
        slots.append(_Slot(tuple(cell), n, turn))
    if not 2 <= len(slots) <= 6:
        raise ValueError("A blueprint needs two to six blocks")
    if slots[0].cell != (0, 0, 0) or slots[0].turn != 0:
        raise ValueError("Start with a stationary row of at least two studs at [0, 0, 0], turn 0")
    return sorted(slots, key=lambda s: s.cell[2])


def _contacts(slots):
    occupied = {}
    contacts = []
    for index, slot in enumerate(slots):
        cells = slot.footprint()
        under = []
        for i, (x, y, z) in enumerate(cells):
            if (x, y, z) in occupied:
                raise ValueError(f"Overlapping blueprint blocks at {(x, y, z)}")
            if (x, y, z - 1) in occupied:
                support, stud = occupied[x, y, z - 1]
                under.append((i, support, stud))
            occupied[x, y, z] = index, i
        if slot.cell[2] and not under:
            raise ValueError(f"At least one stud must support the raised row at {slot.cell}")
        contacts.append(under)
    return contacts


def load_blueprint(spec):
    """Normalize a preset or a JSON object and check its stud contacts, bottom to top.

    Each cell locates the first stud in pitches from the anchor's first stud and
    in seated layers above the table. A turn rotates the row counterclockwise
    in quarter turns. The first entry is the stationary anchor. Every raised
    row needs at least one engaged socket. Overhang stability depends on the
    contacts after release.
    """
    slots = _slots(spec)
    _contacts(slots)
    return {"blocks": [dict(cell=list(s.cell), length=s.length, turn=s.turn) for s in slots]}


def _measured_row(row):
    row = np.asarray(row, dtype=float)
    if row.ndim != 2 or row.shape[1] != 3 or not 2 <= len(row) <= 4 or not np.isfinite(row).all():
        raise ValueError("Measured blocks need two to four finite XYZ studs")
    xy = row[:, :2] - row[:, :2].mean(axis=0)
    axis = np.linalg.svd(xy, full_matrices=False)[2][0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    ordered = row[np.argsort(xy @ axis)]
    if np.max(np.linalg.norm(np.diff(ordered[:, :2], axis=0) - PITCH_M * axis, axis=1)) > .010:
        raise ValueError("Measured studs must form a straight row one pitch apart")
    return ordered, axis


def _anchor_frame(anchor, reference=None):
    measured, axis = _measured_row(anchor)
    if reference is not None and axis @ reference < 0:
        axis = -axis
    across = np.array([-axis[1], axis[0]])
    origin = measured.mean(axis=0) - np.r_[axis * PITCH_M * (len(measured) - 1) / 2, 0.]
    return origin, np.stack((axis, across))


def blueprint_coordinates(stud_rows, anchor):
    """Express measured block centers in pitches from the first anchor stud and in layers."""
    origin, axes = _anchor_frame(stud_rows[anchor])
    coordinates = {}
    for key, row in stud_rows.items():
        measured, _ = _measured_row(row)
        delta = measured.mean(axis=0) - origin
        coordinates[key] = [*(axes @ delta[:2] / PITCH_M).tolist(), float(delta[2] / STACK_CENTER_GAP_M)]
    return coordinates


def _world_row(slot, anchor, reference):
    origin, axes = _anchor_frame(anchor, reference)
    cells = np.asarray(slot.footprint(), dtype=float)
    xyz = np.empty_like(cells)
    xyz[:, :2] = cells[:, :2] @ axes * PITCH_M
    xyz[:, 2] = cells[:, 2] * STACK_CENTER_GAP_M
    return origin + xyz


@dataclass(frozen=True)
class BlueprintPlacement(_Slot):
    source: Hashable
    supports: tuple
    axis_xy: tuple[float, float]
    # For each supporting block, store its blueprint slot and the indices of
    # its studs that actually touch this placement.
    _contact_rows: tuple

    def target_studs(self, anchor_studs, support_studs=None):
        """Fit the target to observed contact studs, preserving pitch and levelness.

        support_studs maps IDs to measured rows. With one contact the anchor
        supplies yaw. Two or more contacts also refine yaw about their midpoint.
        """
        target = _world_row(self, anchor_studs, self.axis_xy)
        expected, observed = [], []
        for key, slot, indices in self._contact_rows:
            if support_studs is None or key not in support_studs:
                continue
            row, _ = _measured_row(support_studs[key])
            if len(row) != slot.length:
                raise ValueError("A measured support must match its blueprint length")
            ideal = _world_row(slot, anchor_studs, self.axis_xy)
            if (row[-1, :2] - row[0, :2]) @ (ideal[-1, :2] - ideal[0, :2]) < 0:
                row = row[::-1]
            expected.extend(ideal[list(indices)])
            observed.extend(row[list(indices)])
        if not observed:
            return target
        src, dst = np.array(expected), np.array(observed)
        a, b = src.mean(axis=0), dst.mean(axis=0)
        xy = target[:, :2] - a[:2]
        if len(src) >= 2:
            p, q = src[:, :2] - a[:2], dst[:, :2] - b[:2]
            theta = np.arctan2(np.sum(p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]), np.sum(p * q))
            c, s = np.cos(theta), np.sin(theta)
            xy = xy @ np.array([[c, s], [-s, c]])
        target[:, :2] = xy + b[:2]
        target[:, 2] += b[2] - a[2]
        return target


def _intersects(a, b):
    # The separating axes of two rectangular footprints are their edge normals.
    def footprint(row):
        ordered, axis = _measured_row(row)
        return (ordered[:, :2].mean(axis=0), np.array([axis, [-axis[1], axis[0]]]),
                PITCH_M / 2 * np.array([len(row), 1.]))
    ca, aa, ra = footprint(a)
    cb, ab, rb = footprint(b)
    for axis in np.vstack((aa, ab)):
        extent = np.abs(aa @ axis) @ ra + np.abs(ab @ axis) @ rb
        if abs((ca - cb) @ axis) >= extent + .002:
            return False
    return True


def plan_blueprint(stud_rows, anchor, blueprint="castle", *, first_source=None, reachable=None):
    """Assign measured blocks to slots, keeping the anchor fixed.

    reachable(source_id, target_studs) can reject a transfer. The planner tries
    alternative length-matched sources when a loose block obstructs the table
    cell or when the IK callback rejects an assignment.
    """
    slots = _slots(blueprint)
    contacts = _contacts(slots)
    if anchor not in stud_rows or len(slots) != len(stud_rows):
        raise ValueError("The blueprint needs one measured block per slot and a known anchor")
    rows = {k: _measured_row(v)[0] for k, v in stud_rows.items()}
    if (len(rows[anchor]) != slots[0].length
            or sorted(map(len, rows.values())) != sorted(s.length for s in slots)):
        raise ValueError("Measured block lengths must match the blueprint and its anchor")
    if first_source is not None and (first_source not in rows or first_source == anchor):
        raise ValueError("The first source must be a known block distinct from the anchor")
    axis = tuple(_measured_row(rows[anchor])[1])
    targets = [_world_row(slot, rows[anchor], axis) for slot in slots]

    @lru_cache(maxsize=None)
    def feasible(index, source):
        return reachable is None or reachable(source, targets[index])

    def assign(chosen, remaining):
        if not remaining:
            return chosen
        i = len(chosen)
        target = targets[i]
        obstructing = {key for key in remaining if slots[i].cell[2] == 0 and _intersects(target, rows[key])}
        candidates = [k for k in remaining if len(rows[k]) == slots[i].length and not obstructing - {k}]
        candidates.sort(key=lambda k: np.linalg.norm(target.mean(axis=0) - rows[k].mean(axis=0)))
        for key in candidates:
            if i == 1 and first_source is not None and key != first_source:
                continue
            if feasible(i, key):
                result = assign(chosen + [key], [other for other in remaining if other != key])
                if result is not None:
                    return result
        return None

    chosen = assign([anchor], [k for k in rows if k != anchor])
    if chosen is None:
        raise ValueError("No clear, reachable assignment to the blueprint")
    placements = []
    for i, slot in enumerate(slots[1:], 1):
        supports = {}
        for _, parent, stud in contacts[i]:
            supports.setdefault(parent, []).append(stud)
        contact_rows = tuple((chosen[j], slots[j], tuple(indices)) for j, indices in supports.items())
        placements.append(BlueprintPlacement(slot.cell, slot.length, slot.turn, chosen[i],
                                              tuple(chosen[j] for j in supports), axis, contact_rows))
    return tuple(placements)
