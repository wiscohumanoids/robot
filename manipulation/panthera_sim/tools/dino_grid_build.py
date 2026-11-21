"""Assemble observed blocks on a stud lattice anchored to a stationary row."""

import numpy as np

from sim.mega_bloks import BODY_HEIGHT_M, STUD_HEIGHT_M
from sim.mega_bloks_grid import grid_coordinates, plan_grid
from sim.mega_bloks_survey import position_observer, survey
from tools.dino_pick_stack import _StackObservations, _center
from tools.dino_pick_targets import Miss, clicked_block, jaw_axes, mark
from tools.dino_stack_track import BlockMap, StackFrame


def _select_blocks(image, blocks, args, select_pixels):
    def selection(pixel):
        return blocks.index(clicked_block(blocks, pixel))

    def source_ok(pixel):
        try:
            selection(pixel)
        except Miss:
            return "Click inside one block."

    def destination_ok(source, dest):
        try:
            if selection(source) != selection(dest):
                return
        except Miss:
            pass
        return "Choose a different block as the grid anchor."

    clicks = args.source_pixel, args.destination_pixel
    if args.interactive:
        clicks = select_pixels(image, blocks, (), source_ok, destination_ok)
        if clicks is None:
            raise Miss("selection_cancelled")
    first = selection(clicks[0]) if clicks[0] is not None else None
    anchor = selection(clicks[1]) if clicks[1] is not None else None
    if first is not None and first == anchor:
        raise Miss("choose_distinct_base")

    return clicks, first, anchor


def _layout(rows, anchor, first, base_count, motion, blueprint):
    # Favor a close anchor so the arm has room to lift above the upper layers.
    candidates = [anchor] if anchor is not None else sorted(
        (i for i in rows if i != first),
        key=lambda i: np.linalg.norm(rows[i].mean(axis=0)[:2] - [.20, .09]))
    def reachable(source, target):
        return _pick_yaw(motion, rows[source], target) is not None

    for candidate in candidates:
        try:
            if blueprint:
                from sim.mega_bloks_blueprint import plan_blueprint

                steps = plan_blueprint(rows, candidate, blueprint, first_source=first, reachable=reachable)
            else:
                steps = plan_grid(rows, candidate, base_count=base_count, first_source=first, reachable=reachable)
        except ValueError:
            continue
        return candidate, steps
    raise Miss("no_reachable_grid_layout")


def _body_center(row):
    return row.mean(axis=0) - np.array([0., 0., (BODY_HEIGHT_M + STUD_HEIGHT_M) / 2])


def _pick_yaw(motion, row, target):
    return motion.choose_placement_yaw(_body_center(row), row, _body_center(target), target, jaw_axes(row))


def _aligned(observed, target):
    if observed.shape != target.shape:
        return False
    # A row looks identical from either end. Compare sockets in order along
    # the intended heading, regardless of the detector's enumeration.
    axis = target[-1, :2] - target[0, :2]
    ordered = observed[np.argsort(observed[:, :2] @ axis)]
    delta = ordered - target
    return np.max(np.linalg.norm(delta[:, :2], axis=1)) <= .012 and np.max(np.abs(delta[:, 2])) <= .009


def build_grid(image, overhead, blocks, see, motion_for, detector, args, folder, record,
               lifted, select_pixels, overlay, executor, before_pick):
    clicks, first, anchor = _select_blocks(image, blocks, args, select_pixels)
    blueprint = getattr(args, "blueprint", None)
    tracker = BlockMap(StackFrame(overhead, tuple(blocks), image.shape))
    record.update(block_pixels=[list(b.xy) for b in blocks], tracking=[], moves=[], arm="left")
    if not position_observer(motion_for("right")):
        raise Miss("observer_pose_unreachable")
    motion = motion_for("left")
    buried = set()
    active = None

    def apply(phase, held, capture_time, top, wrists):
        excluded = buried | ({active} if held else set())
        current = tracker.observe(top, wrists, excluded=excluded)
        record["tracking"].append(dict(
            phase=phase, capture_time_s=capture_time, processed_time_s=float(motion.sim.data.time),
            blocks={i: dict(studs_xyz_m=e.studs_xyz.tolist(), views=e.views,
                            error_px=e.reprojection_error_px) for i, e in current.items()}))

    observations = _StackObservations(see, detector, executor, apply, lambda: float(motion.sim.data.time))
    previous = motion.on_step

    def on_step(controller, phase):
        observations.poll()
        if previous:
            previous(controller, phase)

    motion.on_step = on_step
    try:
        def scan():
            if survey(motion, observations.capture) is None:
                raise Miss("survey_unreachable")
            observations.finish()

        scan()
        if len(tracker.latest) != len(blocks):
            raise Miss("grid_studs_unobserved")
        rows = {i: e.studs_xyz for i, e in tracker.latest.items()}
        base_count = getattr(args, "base_blocks", None) or 2
        if blueprint:
            if sorted(map(len, rows.values())) != sorted(b["length"] for b in blueprint["blocks"]):
                raise Miss("observed_lengths_do_not_match_blueprint")
            tracker.confirm_lengths()
        anchor, steps = _layout(rows, anchor, first, base_count, motion, blueprint)
        if blueprint:
            from sim.mega_bloks_blueprint import blueprint_coordinates

            record["initial_grid"] = blueprint_coordinates(rows, anchor)
            plan = [dict(source=s.source, supports=list(s.supports), length=s.length,
                         cell=list(s.cell), turn=s.turn) for s in steps]
            # The first stud of the anchor is the blueprint origin.
            occupied = {anchor: {(x, 0, 0) for x in range(len(rows[anchor]))}}
            covered = set()
        else:
            plan = [dict(source=s.source, support=s.support, cell=list(s.cell)) for s in steps]
            record.update(base_blocks=base_count, initial_grid=grid_coordinates(rows, anchor))
        record.update(anchor=anchor, grid_plan=plan,
                      destination_pixel=list(clicks[1] if clicks[1] is not None else blocks[anchor].xy),
                      source_pixel=list(clicks[0] if clicks[0] is not None else blocks[steps[0].source].xy))
        mark(folder / "overhead.png", image, blocks, record["source_pixel"], record["destination_pixel"])
        for step in steps:
            active = step.source
            src = tracker.latest[active]

            def target():
                if blueprint:
                    support_rows = {i: tracker.latest[i].studs_xyz for i in step.supports}
                    return step.target_studs(tracker.latest[anchor].studs_xyz, support_rows)
                support = None if step.support is None else tracker.latest[step.support].studs_xyz
                return step.target_studs(tracker.latest[anchor].studs_xyz, support)

            target_row = target()
            yaw = _pick_yaw(motion, src.studs_xyz, target_row)
            if yaw is None:
                raise Miss("unreachable_transfer")
            move = dict(source=active, cell=list(step.cell))
            record["moves"].append(move)
            overlay(overhead.project(src.xyz), overhead.project(target_row.mean(axis=0)),
                    f"Move {len(record['moves'])} of {len(steps)} · grid cell {step.cell}")

            def refine_pick(_, heading):
                nonlocal src
                observations.poll()
                src = tracker.latest[active]
                return _center(src), heading

            if before_pick is not None:
                before_pick()
            picked = motion.pick(_center(src), yaw, observe_approach=lambda: observations.capture("preshape"),
                                 refine=refine_pick)
            lifted()
            if not picked:
                raise Miss("pick_failed")
            target_row = target()
            target_mid = target_row.mean(axis=0)

            def refine_release(plan):
                nonlocal target_row
                observations.poll()
                target_row = target()
                return plan.release + target_row.mean(axis=0) - target_mid

            if not motion.place_aligned(_center(src), src.studs_xyz, _body_center(target_row), target_row,
                                        observe=lambda phase: observations.capture(phase, held=True),
                                        refine=refine_release):
                raise Miss("stack_motion_failed")
            # Drain frames from this grasp before changing the source's identity gate.
            observations.finish()
            move["target_studs_xyz_m"] = target_row.tolist()
            tracker.expect_placement(active, target_row)
            if blueprint:
                cells = set(step.footprint())
                occupied[active] = cells
                covered.update((x, y, z - 1) for x, y, z in cells)
                buried.update(i for i, studs in occupied.items() if studs <= covered)
            elif step.support is not None:
                buried.add(step.support)
            observations.capture("released")
            observations.finish()
            if active not in tracker.latest or not _aligned(tracker.latest[active].studs_xyz, target_row):
                scan()
            if active not in tracker.latest:
                raise Miss("placed_block_unobserved")
            observed = tracker.latest[active].studs_xyz
            move["observed_studs_xyz_m"] = observed.tolist()
            if not _aligned(observed, target_row):
                raise Miss("placed_block_misaligned")
    finally:
        motion.on_step = previous
        observations.cancel()
