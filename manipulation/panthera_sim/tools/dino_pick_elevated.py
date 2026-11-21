"""Estimate block and support heights during a transfer between raised platforms."""

import mujoco
import numpy as np

from sim.mega_bloks import BODY_HEIGHT_M, STUD_HEIGHT_M
from sim.mega_bloks_survey import plan_survey_transfer, position_observer, survey
from tools.dino_pick_held import HeldBlockTracker, STUD_TO_BOTTOM_M, correct_placement
from tools.dino_pick_surface import SurfaceTracker, find_platforms
from tools.dino_pick_track import BlockTrack
from tools.dino_pick_targets import Miss, clicked_block, jaw_axes, mark

UP = np.array([0., 0., 1.])


def rotation(quat):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


class FixedCameraTrack:
    """Follow a selected block using DINO studs in the overhead and right wrist views."""

    def __init__(self, see, detector, color):
        self.see = see
        self.detector = detector
        self.color = color
        self.anchors = ()
        self.initial_xyz = None

    def _pair(self):
        for name in ("overhead", "right_wrist"):
            rgb, camera = self.see(name)
            yield camera, self.detector.detect(rgb, color=self.color), rgb.shape

    def initialize(self, xyz):
        anchors = []
        for camera, blocks, shape in self._pair():
            try:
                x, y = camera.project(xyz)
            except ValueError:
                return
            matches = [block for block in blocks if block.studs_xy
                       and block.bbox_xyxy[0] <= x < block.bbox_xyxy[2]
                       and block.bbox_xyxy[1] <= y < block.bbox_xyxy[3]
                       and self._inside_frame(block, shape)]
            if len(matches) != 1:
                return
            anchors.append((camera, matches[0]))
        self.anchors = tuple(anchors)
        self.initial_xyz = xyz.copy()

    @staticmethod
    def _inside_frame(block, shape):
        x0, y0, x1, y1 = block.bbox_xyxy
        return 0 < x0 < x1 < shape[1] and 0 < y0 < y1 < shape[0]

    @staticmethod
    def _shift(reference, block):
        """Match a coherent temporal shift in DINO stud pixels within one view."""
        old = np.asarray(reference.studs_xy)
        new = np.asarray(block.studs_xy)
        box_delta = (np.asarray(block.bbox_xyxy) - reference.bbox_xyxy).reshape(2, 2)
        coarse = box_delta.mean(axis=0)
        if np.max(np.abs(box_delta)) > 50 or np.linalg.norm(box_delta[0] - box_delta[1]) > 14:
            return None
        shifts = new[None, :, :] - old[:, None, :]
        choices = []
        for proposed in shifts.reshape(-1, 2):
            if np.linalg.norm(proposed - coarse) > 7:
                continue
            distances = np.linalg.norm(shifts - proposed, axis=2)
            indices = distances.argmin(axis=1)
            rows = np.arange(len(old))
            keep = (distances[rows, indices] < 3.5) & (distances.argmin(axis=0)[indices] == rows)
            count = int(keep.sum())
            if count < min(2, len(old), len(new)):
                continue
            offsets = shifts[rows[keep], indices[keep]]
            delta = offsets.mean(axis=0)
            cost = np.linalg.norm(offsets - delta) / np.sqrt(count) + .4 * np.linalg.norm(delta - coarse)
            choices.append((-count, cost, delta))
        if not choices:
            return None
        choices.sort(key=lambda choice: choice[:2])
        count, cost, shift = choices[0]
        if any(n == count and error < cost + 1. and np.linalg.norm(delta - shift) > 3.
               for n, error, delta in choices[1:]):
            return None
        return shift

    def locate(self):
        if not self.anchors:
            return None
        cameras, pixels, rays = [], [], []
        for (reference_camera, reference), (camera, blocks, shape) in zip(self.anchors, self._pair()):
            shifts = [self._shift(reference, block) for block in blocks
                      if block.studs_xy and self._inside_frame(block, shape)]
            matches = [shift for shift in shifts if shift is not None]
            if len(matches) != 1:
                return None
            try:
                # The initial estimate identifies the stud-pattern center. Each
                # measured image displacement transports its anchor pixel.
                pixel = reference_camera.project(self.initial_xyz) + matches[0]
                ray = camera.ray(pixel)
            except ValueError:
                return None
            cameras.append(camera)
            pixels.append(pixel)
            rays.append(ray / np.linalg.norm(ray))
        if np.linalg.norm(np.cross(*rays)) < .10:
            return None
        a, b = cameras
        distances = np.linalg.lstsq(np.column_stack((rays[0], -rays[1])), b.position - a.position,
                                    rcond=None)[0]
        if min(distances) <= 0:
            return None
        point = (a.position + distances[0] * rays[0] + b.position + distances[1] * rays[1]) / 2
        if (np.linalg.norm(point - self.initial_xyz) > .045
                or any(np.linalg.norm(c.project(point) - p) > 3 for c, p in zip(cameras, pixels))):
            return None
        return point


def elevated_policy(see, motion_for, detector, args, folder, record, lifted, select_pixels, overlay):
    image, overhead = see("overhead")
    pads = find_platforms(image)
    blocks = [b for b in detector.detect(image) if b.color is not None and b.studs_xy]
    mark(folder / "overhead.png", image, blocks)
    if not blocks or len(pads) != 2:
        raise Miss("platforms_or_block_not_visible")

    def task(source, destination=None):
        block = blocks[0] if source is None else clicked_block(blocks, source)
        # The bottom of the visible body projects closer to its support than the studs do.
        base = np.array([block.xy[0], block.bbox_xyxy[3] - 1])
        origin = min(pads, key=lambda i: np.linalg.norm(pads[i].center - base))
        other = next(i for i in pads if i != origin)
        pixel = pads[other].center if destination is None else np.asarray(destination)
        if not pads[other].contains(pixel, margin=2):
            raise Miss("choose_inside_other_platform")
        return block, origin, other, pixel

    clicks = args.source_pixel, args.destination_pixel
    if args.interactive:
        def source_ok(source):
            try:
                task(source)
            except Miss:
                return "Click a detected block with visible studs."

        def destination_ok(source, destination):
            try:
                task(source, destination)
            except Miss:
                return "Click inside the outline of the other platform."

        outlines = [pad.polygon.reshape(-1, 2).tolist() for pad in pads.values()]
        clicks = select_pixels(image, blocks, (), source_ok, destination_ok, outlines)
        if clicks is None:
            raise Miss("selection_cancelled")

    block, source_pad, dest_pad, pixel = task(*clicks)
    record.update(source_pixel=list(block.xy if clicks[0] is None else clicks[0]),
                  destination_pixel=pixel.tolist(), color=block.color, arm="left",
                  tracking=[], single_estimate=args.single_estimate)
    mark(folder / "overhead.png", image, blocks, block.xy, pixel)
    track = BlockTrack(overhead, block, neighbors=blocks)
    surface = SurfaceTracker(image, overhead, pixel, describe=detector.describe_surface, platform_id=dest_pad)
    # A ray through this platform's interior measures the source support even
    # when the selected stud pixels project outside its visible outline.
    support = SurfaceTracker(image, overhead, pads[source_pad].center, describe=detector.describe_surface, platform_id=source_pad)
    right = motion_for("right")
    if not position_observer(right):
        raise Miss("observer_pose_unreachable")
    motion = motion_for("left")
    latest_stud = latest_surface = points = None
    established = frozen = False
    current_block = block
    fixed_track = FixedCameraTrack(see, detector, block.color)
    last_sample = -float("inf")
    record["motion_tracking"] = []

    def follow(_motion, phase):
        nonlocal last_sample
        if original_step:
            original_step(_motion, phase)
        if not (phase.startswith("survey") or phase in ("approach", "preshape")):
            return
        now = float(motion.sim.data.time)
        if not fixed_track.anchors or now - last_sample < .25:
            return
        last_sample = now
        point = fixed_track.locate()
        if point is not None:
            record["motion_tracking"].append({"time_s": now, "phase": phase, "xyz_m": point.tolist()})
            overlay(overhead.project(point), pixel, f"Tracking block · base {(point[2]-STUD_TO_BOTTOM_M)*1000:.0f} mm")

    original_step = motion.on_step
    if not args.single_estimate:
        motion.on_step = follow

    def observe_scene(phase, *, camera_name="left_wrist", measure_block=True):
        nonlocal latest_stud, latest_surface, points, established, frozen, current_block
        rgb, camera = see(camera_name)
        entry = {"phase": phase, "camera": camera_name}
        if phase == "right_observer":
            mark(folder / "right_wrist.png", rgb)
        if not frozen:
            surface.add_frame(rgb, camera)
            plane = surface.estimate()
            if plane is not None:
                latest_surface = plane
                entry.update(destination_xyz_m=plane.point.tolist(), surface_views=plane.view_count,
                             plane_rmse_m=plane.plane_rmse_m)
        if measure_block and not frozen:
            detections = detector.detect(rgb, color=block.color)
            fresh, top = see("overhead")
            top_detections = detector.detect(fresh, color=block.color)
            if established:
                update = track.refresh(top, top_detections, camera, detections, frame_shape=rgb.shape)
            else:
                update = track.update(camera, detections, frame_shape=rgb.shape)
            support.add_frame(rgb, camera)
            src_plane = support.estimate()
            entry.update(track_status=update.status, reason=update.reason, block_views=update.views)
            if update.xyz is not None:
                established = True
                latest_stud = update.xyz.copy()
                points = update.studs_xyz.copy()
                entry.update(source_method="stud_triangulation", reprojection_px=update.reprojection_error_px,
                             triangulated_stud_xyz_m=update.xyz.tolist())
                if src_plane is not None:
                    # A resting upright block also has a known height above its observed support.
                    # Use the fitted surface as an independent depth measurement. Keep
                    # the matched overhead pixel on its ray when adjusting depth.
                    plane_stud_z = src_plane.point[2] + STUD_TO_BOTTOM_M
                    height = .75 * plane_stud_z + .25 * latest_stud[2]
                    latest_stud = top.on_plane(top.project(latest_stud), height)
            elif src_plane is not None and update.status != "target_ambiguous":
                # A partial wrist view can confirm a unique overhead region using the
                # measured support. Require a fresh correspondence before moving the arm.
                z = src_plane.point[2] + STUD_TO_BOTTOM_M
                previous = latest_stud if latest_stud is not None else overhead.on_plane(block.xy, z)
                candidates = []
                for candidate_block in top_detections:
                    candidate = top.on_plane(candidate_block.xy, z)
                    if np.linalg.norm(candidate[:2] - previous[:2]) > .040:
                        continue
                    expected = camera.project(candidate)
                    hits = [d for d in detections if d.bbox_xyxy[0]-3 <= expected[0] < d.bbox_xyxy[2]+3
                            and d.bbox_xyxy[1]-3 <= expected[1] < d.bbox_xyxy[3]+3]
                    if len(hits) == 1:
                        candidates.append((candidate_block, candidate))
                if len(candidates) == 1:
                    current_block, candidate = candidates[0]
                    latest_stud = candidate
                    entry["source_method"] = "support_plane_and_wrist"
            if not args.single_estimate and latest_stud is not None:
                if (not fixed_track.anchors and src_plane is not None
                        and "source_method" in entry):
                    fixed_track.initialize(latest_stud)
                fixed_point = fixed_track.locate()
                if fixed_point is not None:
                    shift = fixed_point - fixed_track.initial_xyz
                    entry["fixed_view_displacement_m"] = shift.tolist()
                    if (np.linalg.norm(shift) > .005 or "source_method" not in entry
                            or phase.startswith("preshape")):
                        latest_stud = fixed_point
                        entry["source_method"] = "overhead_and_right_studs"
            if "source_method" in entry:
                entry["stud_xyz_m"] = latest_stud.tolist()
                if phase == "preshape":
                    mark(folder / "tracked_grasp.png", fresh, top_detections,
                         top.project(latest_stud), pixel)
            if phase == "preshape":
                mark(folder / "wrist.png", rgb, detections)
        if args.single_estimate and latest_surface is not None and latest_stud is not None:
            frozen = True
        record["tracking"].append(entry)
        if latest_stud is not None and latest_surface is not None:
            overlay(overhead.project(latest_stud), pixel,
                    f"Block base {(latest_stud[2]-STUD_TO_BOTTOM_M)*1000:.0f} mm   "
                    f"Destination {latest_surface.point[2]*1000:.0f} mm")
        return entry

    observe_scene("right_observer", camera_name="right_wrist")
    if survey(motion, observe_scene) is None:
        raise Miss("survey_unreachable")
    if latest_stud is None or latest_surface is None:
        raise Miss("block_height_unobserved" if latest_stud is None else "destination_height_unobserved")

    def body():
        return latest_stud - (BODY_HEIGHT_M / 2 + STUD_HEIGHT_M / 2) * UP

    def goal():
        return latest_surface.point + BODY_HEIGHT_M / 2 * UP

    def save_geometry():
        record.update(source_xy_m=body()[:2].tolist(), destination_xy_m=goal()[:2].tolist(),
                      source_surface_z_m=float(latest_stud[2]-STUD_TO_BOTTOM_M),
                      destination_surface_z_m=float(latest_surface.point[2]))

    save_geometry()
    pattern = points if points is not None else [overhead.on_plane(p, latest_stud[2])
                                                 for p in current_block.studs_xy]
    yaws = jaw_axes(pattern)
    yaw = plan_survey_transfer(motion, body(), goal(), yaws)
    if yaw is None:
        raise Miss("unreachable_transfer")

    def refine_pick(_, yaw):
        entry = observe_scene("preshape")
        if "stud_xyz_m" not in entry:
            entry = observe_scene("preshape_right", camera_name="right_wrist")
            if "stud_xyz_m" not in entry:
                raise Miss("preshape_target_unconfirmed")
        save_geometry()
        return body(), yaw

    picked = motion.pick(body(), yaw, refine=None if args.single_estimate else refine_pick)
    lifted()
    if not picked:
        raise Miss("pick_failed")

    nominal_grip, quat = motion.nominal_grip_pose(body(), yaw)
    expected_offset = rotation(quat).T @ (latest_stud - nominal_grip)
    held = HeldBlockTracker(block.color, expected_offset, len(pattern), reference_grip_rotation=rotation(quat))

    def observe_transfer(phase):
        entry = observe_scene(phase, measure_block=False)
        grip_xyz, grip_quat = motion.sim.ee_pose()
        for name in ("overhead", "left_wrist", "right_wrist"):
            rgb, camera = see(name)
            hits = detector.detect(rgb, color=block.color)
            entry[name + "_held_match"] = held.observe(camera, hits, grip_xyz, rotation(grip_quat),
                                                       frame_shape=rgb.shape, rgb=rgb)
        fit = held.fit(rotation(grip_quat).T @ UP)
        entry.update(held_fit=fit.reason, held_observations=fit.observations)
        if fit.bottom_offset is not None:
            entry["held_bottom_local_m"] = fit.bottom_offset.tolist()
            overlay(overhead.project(grip_xyz + rotation(grip_quat) @ fit.bottom_offset), pixel,
                    f"Held bottom measured · {fit.observations} observations")

    if not motion.transfer_above(goal(), yaw, observe=None if args.single_estimate else observe_transfer):
        raise Miss("transfer_failed")

    def refine_release(_, yaw):
        target = goal()
        grip, quat = motion.nominal_grip_pose(target, yaw)
        orientation = rotation(quat)
        fit = held.fit(orientation.T @ UP)
        record["held_fit"] = fit.reason
        if fit.bottom_offset is not None:
            try:
                target = correct_placement(target, grip, orientation, fit.bottom_offset)
                record["held_offset_applied"] = True
            except ValueError:
                record["held_fit"] = "correction_out_of_bounds"
        save_geometry()
        return target, yaw

    record["held_offset_applied"] = False
    if not motion.place(goal(), yaw, refine=None if args.single_estimate else refine_release):
        raise Miss("place_failed")
