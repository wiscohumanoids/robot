"""Score released block structures from simulator snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from sim.mega_bloks import PITCH_M
from sim.mega_bloks_stack import STACK_CENTER_GAP_M
from tools.dino_platform_score import _box_ray


_STAGES = ("start", "lift", "release", "final")


@dataclass
class _Snapshot:
    poses: dict
    supports: set
    seated_studs: dict
    clean_blocks: set
    time: float


@dataclass(frozen=True)
class _GridGoal:
    body: str
    along_m: float
    across_m: float
    level: int
    turn: int
    supports: tuple
    stud_contacts: tuple = ()


class StackObserver:
    """Record outcomes without feeding simulator poses or contacts to the policy."""

    def __init__(self, model, data, metadata):
        self._model = model
        self._data = data
        self._geoms = {block["body_name"]: model.geom(block["body_geom"]).id for block in metadata}
        self._lengths = {b["body_name"]: len(b["stud_geoms"]) for b in metadata}
        self._stud_geoms = {model.geom(stud).id for b in metadata for stud in b["stud_geoms"]}
        self._names = {model.body(name).id: name for name in self._geoms}
        self._half_sizes = {name: model.geom_size[geom].copy() for name, geom in self._geoms.items()}
        self._bench_geom = model.geom("table_top").id
        self._selection_geoms = {
            block["body_name"]: [model.geom(name).id for name in (*block["shell_geoms"], *block["stud_geoms"])]
            for block in metadata
        }
        self._snapshots = {}
        self._hold = []
        self._build_start = None
        self._build_lifts = []

    def capture(self, stage):
        """Capture ordered checkpoints and optional samples during the settled hold."""
        if stage == "hold":
            if tuple(self._snapshots) != _STAGES[:3]:
                raise ValueError("Hold samples belong between release and final")
        elif len(self._snapshots) == len(_STAGES) or stage != _STAGES[len(self._snapshots)]:
            raise ValueError(f"Capture must follow stage order {_STAGES}")
        snapshot = self._snapshot()
        if stage == "hold":
            self._hold.append(snapshot)
            return
        if stage == "start":
            self._selection_boxes = {}
            for name, studs in self._selection_geoms.items():
                boxes = []
                for geom in studs:
                    axes = self._data.geom_xmat[geom].reshape(3, 3).copy()
                    bounds = self._model.geom_aabb[geom]
                    boxes.append((self._data.geom_xpos[geom] + axes @ bounds[:3],
                                  axes, bounds[3:].copy()))
                self._selection_boxes[name] = boxes
        self._snapshots[stage] = snapshot

    def _snapshot(self):
        poses = {
            name: (self._data.geom_xpos[geom].copy(), self._data.geom_xmat[geom].reshape(3, 3).copy())
            for name, geom in self._geoms.items()
        }
        supports = set()
        seated_studs = {}
        clean_blocks = set(poses)
        force = np.zeros(6)
        for index in range(self._data.ncon):
            contact = self._data.contact[index]
            if contact.efc_address < 0:
                continue
            first, second = (int(geom) for geom in contact.geom)
            if min(first, second) < 0:
                continue
            a, b = (self._names.get(self._model.geom_bodyid[geom]) for geom in (first, second))
            if a is None and b is None:
                continue
            mujoco.mj_contactForce(self._model, self._data, index, force)
            if force[0] <= 1e-4:
                continue
            if first == self._bench_geom:
                a = "bench"
            if second == self._bench_geom:
                b = "bench"
            if a is None or b is None:
                clean_blocks.discard(a)
                clean_blocks.discard(b)
                continue
            # MuJoCo's normal points from the first geom toward the second.
            normal_z = contact.frame[2]
            if abs(normal_z) <= .7:
                continue
            lower, upper, bottom_geom, top_geom = (a, b, first, second) if normal_z > 0 else (b, a, second, first)
            supports.add((lower, upper))
            if bottom_geom in self._stud_geoms and top_geom == self._geoms.get(upper):
                xy = self._data.geom_xpos[bottom_geom, :2].copy()
                seated_studs.setdefault((lower, upper), {})[bottom_geom] = xy
        return _Snapshot(poses, supports, seated_studs, clean_blocks, float(self._data.time))

    def _selection(self, ray):
        if ray is None:
            return None
        hits = [(depth, name) for name, boxes in self._selection_boxes.items()
                for box in boxes if (depth := _box_ray(ray, *box)) is not None]
        return min(hits)[1] if hits else None

    def _bottom(self, snapshot, name):
        center, axes = snapshot.poses[name]
        return center[2] - np.abs(axes[2]) @ self._half_sizes[name]

    def score(self, *, source_pixel_ray, destination_ray, min_lift_m=.04, min_hold_s=.2,
              placement_tolerance_m=.012, base_tolerance_m=.01, drift_tolerance_m=.003,
              upright_tolerance_deg=10., yaw_tolerance_deg=15., seating_tolerance_m=.005):
        """Match the original selection rays and check load-bearing contact after release."""
        if tuple(self._snapshots) != _STAGES:
            raise RuntimeError(f"Capture {_STAGES} before scoring")
        source, base = map(self._selection, (source_pixel_ray, destination_ray))
        start, lift, release, final = (self._snapshots[stage] for stage in _STAGES)
        heights = {name: float(self._bottom(lift, name) - self._bottom(start, name))
                   for name in self._geoms}
        lifted = [name for name, rise in heights.items() if rise >= min_lift_m]
        pickup = source is not None and lifted == [source]
        distinct = source is not None and base is not None and source != base
        hold_s = final.time - release.time
        result = {
            "success": False, "placement_success": False, "intended_body": source,
            "destination_body": base, "distinct_selections": distinct, "pickup_success": pickup,
            "lifted_bodies": lifted, "wrong_object": None if source is None else any(name != source for name in lifted),
            "pickup_height_m": heights.get(source), "placement_error_m": None,
            "base_displacement_m": None, "base_rotation_deg": None, "yaw_error_deg": None,
            "stack_contact": False, "base_on_bench": False, "upright": False, "stable_hold": False,
            "hold_duration_s": hold_s, "hold_drift_m": None, "released": False,
            "stack_gap_m": None, "seated": False,
        }
        if not distinct:
            return result

        hold = [release, *self._hold, final]
        base_displacement = self._max_displacement(base, start, [lift, *hold])
        base_rotation = max(self._rotation_error(start.poses[base][1], frame.poses[base][1])
                            for frame in [lift, *hold])
        base_on_bench = all(("bench", base) in frame.supports for frame in [start, lift, *hold])
        supported = all((base, source) in frame.supports and ("bench", source) not in frame.supports
                        for frame in hold)
        released = all({source, base} <= frame.clean_blocks for frame in hold)
        drift = max(self._max_displacement(name, release, hold) for name in (source, base))
        max_tilt = max(self._angle(frame.poses[name][1][2, 2])
                       for frame in hold for name in (source, base))
        upright = max_tilt <= upright_tolerance_deg
        yaw = max(self._angle(abs(frame.poses[source][1][:, 0] @ frame.poses[base][1][:, 0]))
                  for frame in hold)
        alignment = [float(np.linalg.norm(frame.poses[source][0][:2] - frame.poses[base][0][:2]))
                     for frame in hold]
        gaps = [float(frame.poses[source][0][2] - frame.poses[base][0][2]) for frame in hold]
        seated = all(abs(gap - STACK_CENTER_GAP_M) <= seating_tolerance_m for gap in gaps)
        stable = hold_s + 1e-9 >= min_hold_s and drift <= drift_tolerance_m and supported and released
        success = (pickup and base_on_bench and base_displacement <= base_tolerance_m
                   and base_rotation <= yaw_tolerance_deg and seated and upright and stable
                   and max(alignment) <= placement_tolerance_m and yaw <= yaw_tolerance_deg)
        result.update(
            success=success, placement_success=success, placement_error_m=alignment[-1],
            base_displacement_m=base_displacement, base_rotation_deg=base_rotation, yaw_error_deg=yaw,
            stack_contact=supported, base_on_bench=base_on_bench, upright=upright, stable_hold=stable,
            hold_drift_m=drift, released=released, stack_gap_m=gaps[-1], seated=seated,
        )
        return result

    def capture_build_start(self):
        """Record all block heights immediately before the next build grasp."""
        if self._build_start is not None:
            raise ValueError("Capture the previous build lift before starting another grasp")
        self._build_start = self._snapshot()

    def capture_build_lift(self):
        """Pair a post-grasp sample with the heights at the start of that grasp."""
        if self._build_start is None:
            raise ValueError("Capture the start of the grasp before the build lift")
        self._build_lifts.append((self._build_start, self._snapshot()))
        self._build_start = None

    def score_build(self, *, anchor_ray, placements, min_hold_s=.2, min_lift_m=.04,
                    placement_tolerance_m=.012, drift_tolerance_m=.003,
                    seating_tolerance_m=.005, angle_tolerance_deg=15.):
        """Check the final build against the original anchor pose throughout the hold.

        Each placement gives an original overhead selection ray, two horizontal
        offsets in meters along/across the anchor stud row, and a stack level.
        The row axis points toward its positive dominant world coordinate.
        """
        anchor = self._selection(anchor_ray)
        layout = [(anchor, 0., 0., 0)]
        layout.extend((self._selection(ray), along, across, level)
                      for ray, along, across, level in placements)
        cells = {(along, across, level): name for name, along, across, level in layout}
        error = "two blocks were assigned the same grid position" if len(cells) != len(layout) else ""
        goals = [_GridGoal(name, along, across, level, 0,
                           ("bench",) if level == 0 else (cells.get((along, across, level - 1)),))
                 for name, along, across, level in layout]
        return self._score_layout(goals, error, min_hold_s, min_lift_m, placement_tolerance_m,
                                  drift_tolerance_m, seating_tolerance_m, angle_tolerance_deg)

    def score_blueprint(self, *, anchor_ray, placements, min_hold_s=.2, min_lift_m=.04,
                        placement_tolerance_m=.012, drift_tolerance_m=.003,
                        seating_tolerance_m=.005, angle_tolerance_deg=15.):
        """Score socket placements in the original anchor's end-stud coordinate frame.

        Each placement gives an original selection ray, first stud cell (x,y,layer),
        counterclockwise quarter turns, and row length in studs. Every block under
        an occupied cell must carry load for the full hold, including bridge piers.
        """
        anchor = self._selection(anchor_ray)
        anchor_length = self._lengths.get(anchor, 0)
        layout = [(anchor, (0, 0, 0), 0, anchor_length)]
        layout.extend((self._selection(ray), cell, turn, length) for ray, cell, turn, length in placements)
        directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        owners = {}
        error = ""
        pieces = []
        for body, (x, y, level), turn, length in layout:
            if body is None or self._lengths.get(body) != length or length <= 0 or turn not in range(4):
                error = "blueprint selection or stud row does not match the original block"
                break
            dx, dy = directions[turn]
            footprint = [(x + i * dx, y + i * dy, level) for i in range(length)]
            if level < 0 or any(cell in owners for cell in footprint):
                error = "blueprint contains overlapping blocks or a negative level"
                break
            owners.update((cell, body) for cell in footprint)
            along = x + dx * (length - 1) / 2 - (anchor_length - 1) / 2
            across = y + dy * (length - 1) / 2
            pieces.append((body, along * PITCH_M, across * PITCH_M, level, turn, footprint))

        goals = []
        for body, along, across, level, turn, footprint in pieces:
            beneath = {owners[(x, y, level - 1)] for x, y, _ in footprint if (x, y, level - 1) in owners}
            supports = tuple(sorted(beneath)) if level else ("bench",)
            if not supports:
                error = "blueprint contains an elevated block without a supporting stud"
            contacts = tuple((owners[x, y, level - 1], (x - (anchor_length - 1) / 2) * PITCH_M, y * PITCH_M)
                             for x, y, _ in footprint if (x, y, level - 1) in owners)
            goals.append(_GridGoal(body, along, across, level, turn, supports, contacts))
        return self._score_layout(goals, error, min_hold_s, min_lift_m, placement_tolerance_m,
                                  drift_tolerance_m, seating_tolerance_m, angle_tolerance_deg)

    def _score_layout(self, goals, error, min_hold_s, min_lift_m, placement_tolerance_m,
                      drift_tolerance_m, seating_tolerance_m, angle_tolerance_deg):
        if tuple(self._snapshots) != _STAGES:
            raise RuntimeError(f"Capture {_STAGES} before scoring")
        start = self._snapshots["start"]
        hold = [self._snapshots["release"], *self._hold, self._snapshots["final"]]
        duration = hold[-1].time - hold[0].time
        result = dict(success=False, placement_success=False, stable_hold=False,
                      hold_duration_s=duration, blocks=[], pickups=[], pickup_success=False, reason=error)
        if error:
            return result
        names = [goal.body for goal in goals]
        if len(names) != len(self._geoms) or set(names) != set(self._geoms):
            result["reason"] = "build selections did not identify every block exactly once"
            return result
        for index, name in enumerate(names[1:]):
            heights = {}
            if index < len(self._build_lifts):
                before, after = self._build_lifts[index]
                heights = {body: float(self._bottom(after, body) - self._bottom(before, body))
                           for body in names}
            lifted = [body for body, height in heights.items() if height >= min_lift_m]
            result["pickups"].append(dict(intended_body=name, lifted_bodies=lifted,
                                          pickup_height_m=heights.get(name), success=lifted == [name]))
        result["pickup_success"] = (len(self._build_lifts) == len(names) - 1 and self._build_start is None
                                    and all(pickup["success"] for pickup in result["pickups"]))
        origin, anchor_axes = start.poses[names[0]]
        row = anchor_axes[:2, 0].copy()
        row /= np.linalg.norm(row)
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
        across_axis = np.array([-row[1], row[0]])
        for goal in goals:
            name = goal.body
            target_xy = origin[:2] + goal.along_m * row + goal.across_m * across_axis
            expected = set(goal.supports)
            supported = all({lower for lower, upper in frame.supports if upper == name} == expected
                            for frame in hold)
            if goal.stud_contacts:
                # An upright wall can rest on a shell edge. Verify that a lower
                # stud at the intended joint actually carries the upper roof.
                joint_targets = {support: [] for support in expected}
                for support, along, across in goal.stud_contacts:
                    joint_targets[support].append(origin[:2] + along * row + across * across_axis)
                supported = supported and all(
                    any(np.linalg.norm(xy - target) <= placement_tolerance_m
                        for xy in frame.seated_studs.get((support, name), {}).values()
                        for target in targets)
                    for support, targets in joint_targets.items() for frame in hold
                )
            released = all(name in frame.clean_blocks for frame in hold)
            drift = self._max_displacement(name, hold[0], hold)
            xy_error = max(float(np.linalg.norm(frame.poses[name][0][:2] - target_xy))
                           for frame in hold)
            z_error = max(abs(float(frame.poses[name][0][2] - origin[2] - goal.level * STACK_CENTER_GAP_M))
                          for frame in hold)
            tilt = max(self._angle(frame.poses[name][1][2, 2]) for frame in hold)
            target_axis = across_axis if goal.turn % 2 else row
            yaw = max(self._angle(abs(frame.poses[name][1][:2, 0] @ target_axis)) for frame in hold)
            stable = (supported and released and drift <= drift_tolerance_m
                      and duration + 1e-9 >= min_hold_s)
            placed = (stable and xy_error <= placement_tolerance_m
                      and z_error <= seating_tolerance_m and tilt <= 10.
                      and yaw <= angle_tolerance_deg)
            result["blocks"].append(dict(body=name, supports=list(goal.supports),
                                         support=goal.supports[0] if len(goal.supports) == 1 else None,
                                         supported=supported,
                                         released=released, hold_drift_m=drift,
                                         placement_error_m=xy_error, height_error_m=z_error,
                                         tilt_deg=tilt, yaw_error_deg=yaw,
                                         stable_hold=stable, success=placed))
        result["stable_hold"] = all(block["stable_hold"] for block in result["blocks"])
        result["placement_success"] = all(block["success"] for block in result["blocks"])
        result["success"] = result["pickup_success"] and result["placement_success"]
        if not result["pickup_success"]:
            result["reason"] = "could not verify a separate lift of each selected block"
        elif not result["placement_success"]:
            result["reason"] = "one or more blocks failed the support, stability, or alignment checks"
        return result

    @staticmethod
    def _angle(cosine):
        return float(np.degrees(np.arccos(np.clip(cosine, -1., 1.))))

    def _rotation_error(self, initial, actual):
        return self._angle((np.trace(initial.T @ actual) - 1.) / 2.)

    @staticmethod
    def _max_displacement(name, initial, frames):
        return max(float(np.linalg.norm(frame.poses[name][0] - initial.poses[name][0])) for frame in frames)
