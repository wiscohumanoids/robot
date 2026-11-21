"""Pick an overhead selection and seat it on a second RGB-selected block."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy

import numpy as np

from sim.mega_bloks import BODY_HEIGHT_M, STUD_HEIGHT_M
from sim.mega_bloks_survey import position_observer, survey
from tools.dino_pick_targets import Miss, clicked_block, jaw_axes, mark
from tools.dino_stack_track import StackFrame, StackTracker


def _center(track):
    return track.xyz - np.array([0., 0., (BODY_HEIGHT_M + STUD_HEIGHT_M) / 2])


class _StackObservations:
    """Capture synchronized views on the simulation thread and deliver detections in order."""

    def __init__(self, see, detector, executor, apply, sim_time):
        self.see = see
        self.detector = detector
        self.executor = executor
        self.apply = apply
        self.sim_time = sim_time
        self.pending = deque()

    def _detect(self, views):
        return tuple(StackFrame(camera, tuple(self.detector.detect(rgb)), rgb.shape)
                     for rgb, camera in views)

    def capture(self, phase, held=False):
        views = []
        for name in ("overhead", "left_wrist", "right_wrist"):
            rgb, camera = self.see(name)
            views.append((rgb.copy(), deepcopy(camera)))
        self.pending.append((phase, held, self.sim_time(), self.executor.submit(self._detect, views)))

    def _receive(self):
        phase, held, capture_time, future = self.pending.popleft()
        overhead, *wrists = future.result()
        self.apply(phase, held, capture_time, overhead, wrists)

    def poll(self):
        while self.pending and self.pending[0][-1].done():
            self._receive()

    def wait_until(self, ready):
        while self.pending and not ready():
            self._receive()

    def finish(self):
        while self.pending:
            self._receive()

    def cancel(self):
        while self.pending:
            self.pending.popleft()[-1].cancel()


def stack_policy(see, motion_for, detector, args, folder, record, lifted, select_pixels, overlay,
                 *, executor=None, before_pick=None):
    pool = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="stack-perception")
            if executor is None else nullcontext(executor))
    with pool as worker:
        _stack_policy(see, motion_for, detector, args, folder, record, lifted, select_pixels,
                      overlay, worker, before_pick)


def _stack_policy(see, motion_for, detector, args, folder, record, lifted, select_pixels, overlay,
                  executor, before_pick):
    image, overhead = see("overhead")
    detections = executor.submit(detector.detect, image).result()
    blocks = sorted((b for b in detections if b.color and b.studs_xy),
                    key=lambda b: b.xy[0])
    mark(folder / "overhead.png", image, blocks)
    if args.blocks > 2 or getattr(args, "blueprint", None):
        from tools.dino_grid_build import build_grid

        if len(blocks) != args.blocks:
            raise Miss("grid_blocks_not_detected")
        return build_grid(image, overhead, blocks, see, motion_for, detector, args, folder,
                          record, lifted, select_pixels, overlay, executor, before_pick)
    if len(blocks) != 2:
        raise Miss("two_blocks_not_detected")

    def choose(source, target):
        base = clicked_block(blocks, target) if target is not None else None
        top = (clicked_block(blocks, source) if source is not None else
               next(b for b in blocks if b is not base))
        if base is top:
            raise Miss("choose_distinct_base")
        return top, base or next(b for b in blocks if b is not top)

    clicks = args.source_pixel, args.destination_pixel
    if args.interactive:
        def source_ok(source):
            try:
                choose(source, None)
            except Miss:
                return "Click inside a detected block."

        def destination_ok(source, dest):
            try:
                choose(source, dest)
            except Miss:
                return "Click the other block to use it as the base."

        clicks = select_pixels(image, blocks, (), source_ok, destination_ok)
        if clicks is None:
            raise Miss("selection_cancelled")
    source, base = choose(*clicks)
    record.update(arm="left", color=source.color, base_color=base.color, tracking=[])
    for key, click, block in zip(("source", "destination"), clicks, (source, base)):
        record[key + "_pixel"] = list(click if click is not None else block.xy)
        if click is not None:
            record["clicked_" + key + "_pixel"] = list(click)
    mark(folder / "overhead.png", image, blocks, source.xy, base.xy)
    tracker = StackTracker(StackFrame(overhead, tuple(blocks), image.shape), source, base)
    if not position_observer(motion_for("right")):
        raise Miss("observer_pose_unreachable")
    motion = motion_for("left")
    source_on_bench = True

    def apply(phase, held, capture_time, top, wrists):
        current = tracker.observe(top, wrists, source_held=held)
        entry = {"phase": phase, "capture_time_s": capture_time,
                 "processed_time_s": float(motion.sim.data.time)}
        for name, estimate in (("source", current.source), ("base", current.target)):
            if estimate is not None:
                entry[name] = dict(studs_xyz_m=estimate.studs_xyz.tolist(), views=estimate.views,
                                   error_px=estimate.reprojection_error_px)
        record["tracking"].append(entry)
        latest = tracker.latest
        if latest.source is not None and latest.target is not None:
            source_pixel = overhead.project(latest.source.xyz) if source_on_bench else None
            visible = "observed" if current.target is not None else "last observed"
            overlay(source_pixel, overhead.project(latest.target.xyz),
                    f"{phase} observation · base studs {visible} at {latest.target.xyz[2] * 1000:.0f} mm")

    observations = _StackObservations(see, detector, executor, apply,
                                      lambda: float(motion.sim.data.time))
    previous_step = motion.on_step

    def on_step(controller, phase):
        nonlocal source_on_bench
        if phase == "lift":
            source_on_bench = False
        observations.poll()
        if previous_step is not None:
            previous_step(controller, phase)

    motion.on_step = on_step
    try:
        _pick_and_stack(motion, observations, tracker, record, lifted)
        observations.finish()
    finally:
        motion.on_step = previous_step
        observations.cancel()


def _pick_and_stack(motion, observations, tracker, record, lifted):
    if survey(motion, observations.capture) is None:
        raise Miss("survey_unreachable")
    observations.poll()
    observations.wait_until(lambda: tracker.latest.source is not None and tracker.latest.target is not None)
    src, dst = tracker.latest.source, tracker.latest.target
    if src is None or dst is None:
        raise Miss("stack_studs_unobserved")
    yaw = motion.choose_stack_yaw(_center(src), src.studs_xyz, _center(dst), dst.studs_xyz,
                                  jaw_axes(src.studs_xyz))
    if yaw is None:
        raise Miss("unreachable_transfer")

    def refine_pick(_, yaw):
        nonlocal src
        observations.poll()
        src = tracker.latest.source
        return _center(src), yaw

    picked = motion.pick(_center(src), yaw, observe_approach=lambda: observations.capture("preshape"),
                         refine=refine_pick)
    lifted()
    if not picked:
        raise Miss("pick_failed")
    # Source is the resting geometry used for this grasp. A late frame can
    # update the base estimate, but the grasp transform is already fixed.
    observations.poll()
    dst = tracker.latest.target
    record.update(source_xy_m=_center(src)[:2].tolist(), destination_xy_m=_center(dst)[:2].tolist())

    def refine_release(plan):
        # The held block can hide the base during the final approach. Keep
        # the most recent triangulated base position through that occlusion.
        observations.poll()
        return plan.release + tracker.latest.target.xyz - dst.xyz

    if not motion.stack(_center(src), src.studs_xyz, _center(dst), dst.studs_xyz,
                        observe=lambda phase: observations.capture(phase, held=True),
                        refine=refine_release):
        raise Miss("stack_motion_failed")
