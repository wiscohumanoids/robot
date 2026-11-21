"""Evaluate a settled cross-platform transfer using simulator geometry."""

from __future__ import annotations

import numpy as np

from tools.dino_pick_score import PickObserver, _xy


def _box_ray(ray, center, rotation, half_size):
    """Return the forward ray's first intersection with an oriented box."""
    origin, direction = (np.asarray(value, dtype=float) for value in ray)
    if (origin.shape != (3,) or direction.shape != (3,) or not np.isfinite(origin).all()
            or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-10):
        raise ValueError("A pixel ray requires a finite origin and nonzero direction")
    origin = rotation.T @ (origin - center)
    direction = rotation.T @ direction
    lower, upper = 0., float("inf")
    for offset, slope, limit in zip(origin, direction, half_size):
        if abs(slope) < 1e-10:
            if abs(offset) > limit:
                return None
            continue
        ends = sorted(((-limit - offset) / slope, (limit - offset) / slope))
        lower, upper = max(lower, ends[0]), min(upper, ends[1])
        if lower > upper:
            return None
    return lower


class PlatformObserver(PickObserver):
    """Record the same action boundaries as the bench evaluator."""

    def __init__(self, model, data, blocks, platform_geom_names):
        super().__init__(model, data, blocks)
        self._supports = {}
        for name in platform_geom_names:
            geom = model.geom(name).id
            self._supports[name] = (data.geom_xpos[geom].copy(),
                                    data.geom_xmat[geom].reshape(3, 3).copy(),
                                    model.geom_size[geom].copy())
        self._stud_boxes = {
            block["body_name"]: [(model.geom(name).id, model.geom_aabb[model.geom(name).id].copy())
                                 for name in block["stud_geoms"]]
            for block in blocks
        }
        self._initial_studs = {}

    def capture(self, stage):
        super().capture(stage)
        if stage == "start":
            for name, studs in self._stud_boxes.items():
                boxes = []
                for geom, bounds in studs:
                    rotation = self._data.geom_xmat[geom].reshape(3, 3).copy()
                    center = self._data.geom_xpos[geom] + rotation @ bounds[:3]
                    boxes.append((center, rotation, bounds[3:]))
                self._initial_studs[name] = boxes

    def _source(self, source_xy, ray, match_radius):
        if ray is not None:
            hits = []
            for name, (center, rotation) in self._poses["start"].items():
                boxes = [(center, rotation, self._half_sizes[name]), *self._initial_studs[name]]
                hits.extend((depth, name) for box in boxes
                            if (depth := _box_ray(ray, *box)) is not None)
            return min(hits)[1] if hits else None
        if source_xy is not None:
            nearest = min((float(np.linalg.norm(center[:2] - source_xy)), name)
                          for name, (center, _) in self._poses["start"].items())
            if nearest[0] <= match_radius:
                return nearest[1]
        return None

    def _destination(self, destination_xy, ray):
        if ray is not None:
            hits = [(depth, name) for name, box in self._supports.items()
                    if (depth := _box_ray(ray, *box)) is not None]
            if hits:
                depth, name = min(hits)
                point = np.asarray(ray[0]) + depth * np.asarray(ray[1])
                center, rotation, half_size = self._supports[name]
                if abs((rotation.T @ (point - center))[2] - half_size[2]) < .00001:
                    return name, point[:2]
        elif destination_xy is not None:
            for name, (center, rotation, half_size) in self._supports.items():
                local = rotation[:2, :2].T @ (destination_xy - center[:2])
                if np.all(np.abs(local) <= half_size[:2]):
                    return name, destination_xy
        return None, None

    def _support_z(self, name):
        return float(self._supports[name][0][2] + self._supports[name][2][2])

    def _gap(self, body, pose, support, tolerance):
        center, rotation = pose
        position, axes, half_size = self._supports[support]
        local = axes.T @ (center - position)
        extent = np.abs(axes.T @ rotation) @ self._half_sizes[body]
        gap = float(local[2] - extent[2] - half_size[2])
        contained = bool(np.all(np.abs(local[:2]) + extent[:2] <= half_size[:2] + tolerance))
        return gap, contained

    def score(
        self, *, source_xy_m=None, destination_xy_m=None,
        source_surface_z_m=None, destination_surface_z_m=None,
        destination_ray=None, source_pixel_ray=None, min_lift_m=.04,
        placement_tolerance_m=.025, max_source_error_m=.04, support_tolerance_m=.006,
    ):
        """Use original selection rays when available and report perception errors separately.

        Source ray attribution uses the first intersected initial block. The
        destination ray must intersect the top of a platform. Without rays,
        use the supplied world XY values to identify the block and target.
        """
        if len(self._poses) != 3:
            raise RuntimeError("Capture start, lift, and final before scoring")
        source = None if source_xy_m is None else _xy(source_xy_m, "source_xy_m")
        target = None if destination_xy_m is None else _xy(destination_xy_m, "destination_xy_m")
        intended = self._source(source, source_pixel_ray, max_source_error_m)
        destination_support, actual_target = self._destination(target, destination_ray)
        start, lift, final = (self._poses[stage] for stage in ("start", "lift", "final"))
        lifted_bodies = []
        pickup_height = None
        for name, (center, rotation) in start.items():
            lift_center, lift_rotation = lift[name]
            delta = ((lift_center - center)[2]
                     + (np.abs(rotation[2]) - np.abs(lift_rotation[2])) @ self._half_sizes[name])
            if delta >= min_lift_m:
                lifted_bodies.append(name)
            if name == intended:
                pickup_height = float(delta)
        wrong_object = None if intended is None else any(name != intended for name in lifted_bodies)
        pickup_success = intended in lifted_bodies and not wrong_object

        source_support = None
        if intended is not None:
            for name in self._supports:
                gap, contained = self._gap(intended, start[intended], name, support_tolerance_m)
                if contained and abs(gap) <= support_tolerance_m:
                    source_support = name
                    break
        source_z = None if source_support is None else self._support_z(source_support)
        destination_z = None if destination_support is None else self._support_z(destination_support)
        direction = None
        if source_z is not None and destination_z is not None:
            direction = "uphill" if destination_z > source_z else "downhill" if destination_z < source_z else "level"

        placement_error = gap = on_destination = None
        if destination_support is not None and intended is not None:
            placement_error = float(np.linalg.norm(final[intended][0][:2] - actual_target))
            gap, contained = self._gap(intended, final[intended], destination_support, support_tolerance_m)
            on_destination = contained and abs(gap) <= support_tolerance_m
        success = bool(pickup_success and on_destination and direction in ("uphill", "downhill")
                       and placement_error <= placement_tolerance_m)
        return {
            "success": success,
            "placement_success": success,
            "intended_body": intended,
            "lifted_bodies": lifted_bodies,
            "wrong_object": wrong_object,
            "pickup_success": pickup_success,
            "pickup_height_m": pickup_height,
            "source_platform": source_support,
            "destination_platform": destination_support,
            "direction": direction,
            "source_surface_z_m": source_z,
            "destination_surface_z_m": destination_z,
            "source_height_error_m": (None if source_z is None or source_surface_z_m is None
                                      else float(abs(source_z - source_surface_z_m))),
            "destination_height_error_m": (None if destination_z is None or destination_surface_z_m is None
                                           else float(abs(destination_z - destination_surface_z_m))),
            "destination_true_xy_m": None if actual_target is None else actual_target.tolist(),
            "destination_projection_error_m": (None if actual_target is None or target is None
                                               else float(np.linalg.norm(target - actual_target))),
            "on_destination": on_destination,
            "destination_gap_m": gap,
            "placement_error_m": placement_error,
        }
