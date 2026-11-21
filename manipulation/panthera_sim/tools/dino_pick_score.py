"""Observe and score a DINO-assisted pick-and-place trial after execution."""

from __future__ import annotations

import numpy as np


_STAGES = ("start", "lift", "final")


def _xy(point, label):
    point = np.asarray(point, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{label} must contain two finite coordinates")
    return point


class PickObserver:
    """Record scene poses at action boundaries without exposing them to the policy."""

    def __init__(self, model, data, blocks):
        self._data = data
        self._body_geoms = {block["body_name"]: model.geom(block["body_geom"]).id for block in blocks}
        self._half_sizes = {name: model.geom_size[geom].copy()
                            for name, geom in self._body_geoms.items()}
        self._table_geom = model.geom("table_top").id
        self._table_half_size = model.geom_size[self._table_geom].copy()
        self._poses = {}

    def capture(self, stage):
        """Copy poses after settling, lifting, or releasing, in that order."""
        if len(self._poses) == len(_STAGES) or stage != _STAGES[len(self._poses)]:
            raise ValueError(f"Capture must be called once for each stage in {_STAGES}")
        self._poses[stage] = {
            name: (self._data.geom_xpos[geom].copy(), self._data.geom_xmat[geom].reshape(3, 3).copy())
            for name, geom in self._body_geoms.items()
        }
        if stage == "final":
            self._table_position = self._data.geom_xpos[self._table_geom].copy()
            self._table_rotation = self._data.geom_xmat[self._table_geom].reshape(3, 3).copy()
            self._data = None

    def score(
        self,
        *,
        source_xy_m,
        destination_xy_m,
        min_lift_m=0.04,
        placement_tolerance_m=0.025,
        max_source_error_m=0.04,
        ambiguity_margin_m=0.008,
        table_tolerance_m=0.006,
        final_height_tolerance_m=0.006,
    ):
        """Score the intended block and flag any other block raised at the lift snapshot.

        Source and destination are the policy's tabletop world XY estimates in meters.
        The source estimate identifies the nearest initial body center within the
        match radius. A second body within the ambiguity margin prevents attribution.
        Lift is the rise of the block's lowest body corner. Placement requires the
        center to reach the requested XY and return to its initial height with
        the body resting on the table.
        """
        if len(self._poses) != len(_STAGES):
            raise RuntimeError("Capture start, lift, and final before scoring")
        source = _xy(source_xy_m, "source_xy_m")
        destination = _xy(destination_xy_m, "destination_xy_m")
        start, lift, final = (self._poses[stage] for stage in _STAGES)
        candidates = sorted((float(np.linalg.norm(pose[0][:2] - source)), name)
                            for name, pose in start.items())
        source_error = candidates[0][0] if candidates else None
        intended = None
        source_match = "out_of_range"
        if candidates and source_error <= max_source_error_m:
            if len(candidates) > 1 and candidates[1][0] - source_error <= ambiguity_margin_m:
                source_match = "ambiguous"
            else:
                intended = candidates[0][1]
                source_match = "matched"

        pickup_heights = {}
        for name, (center, rotation) in start.items():
            start_bottom = center[2] - np.abs(rotation[2]) @ self._half_sizes[name]
            lift_center, lift_rotation = lift[name]
            lift_bottom = lift_center[2] - np.abs(lift_rotation[2]) @ self._half_sizes[name]
            pickup_heights[name] = float(lift_bottom - start_bottom)
        lifted_bodies = [name for name, height in pickup_heights.items() if height >= min_lift_m]
        wrong_object = None if intended is None else any(name != intended for name in lifted_bodies)
        pickup_success = intended in lifted_bodies and not wrong_object
        pickup_height = pickup_heights.get(intended)
        placement_error = None
        final_height_error = None
        table_gap = None
        on_table = None
        if intended is not None:
            center, rotation = final[intended]
            relative = self._table_rotation.T @ (center - self._table_position)
            body_extents = np.abs(self._table_rotation.T @ rotation) @ self._half_sizes[intended]
            table_gap = float(relative[2] - body_extents[2] - self._table_half_size[2])
            on_table = bool(
                abs(table_gap) <= table_tolerance_m
                and np.all(np.abs(relative[:2]) + body_extents[:2]
                           <= self._table_half_size[:2] + table_tolerance_m)
            )
            placement_error = float(np.linalg.norm(center[:2] - destination))
            final_height_error = float(abs(center[2] - start[intended][0][2]))
        placement_success = bool(
            pickup_success
            and on_table
            and placement_error <= placement_tolerance_m
            and final_height_error <= final_height_tolerance_m
        )
        return {
            "success": placement_success,
            "source_match": source_match,
            "source_match_error_m": source_error,
            "intended_body": intended,
            "lifted_bodies": lifted_bodies,
            "wrong_object": wrong_object,
            "pickup_success": pickup_success,
            "pickup_height_m": pickup_height,
            "on_table": on_table,
            "table_gap_m": table_gap,
            "final_height_error_m": final_height_error,
            "placement_success": placement_success,
            "placement_error_m": placement_error,
        }
