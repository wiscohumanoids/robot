"""Move a visually selected Mega Blok to a bench, platform, or another block."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
from itertools import count
import tempfile
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

from perception.pickup import DinoPickupPerception
from sim.mega_bloks import build_mega_bloks_scene
from sim.mega_bloks_motion import PickPlaceMotion
from sim.panthera_env import PantheraSim
from tools.dino_pick_camera import read_camera
from tools.dino_pick_score import PickObserver
from tools.dino_pick_targets import Miss, clicked_block, jaw_axes, mark
from tools.dino_pick_refine import overhead_pattern_ok, refine_grasp

# Fixed bench calibration and nominal dimensions for this family of blocks.
BENCH_Z = .050
BODY_CENTER_Z = BENCH_Z + .020
STUD_CENTER_Z = BENCH_Z + .050
WORKSPACE_X = (.16, .49)
WORKSPACE_Y = (-.23, .23)


def center(camera, block):
    point = camera.on_plane(block.xy, STUD_CENTER_Z if block.studs_xy else BODY_CENTER_Z)
    return point[:2]


def source_candidates(camera, blocks, pixel):
    if pixel is not None:
        return [clicked_block(blocks, pixel)]
    return sorted((b for b in blocks if b.color is not None),
                  key=lambda b: np.linalg.norm(center(camera, b) - [.30, .13]))


def yaw_candidates(camera, block, stud_z=STUD_CENTER_Z):
    return jaw_axes([camera.on_plane(s, stud_z) for s in block.studs_xy])


def footprint_radius(camera, block):
    if not block.studs_xy:
        return .055
    xy = center(camera, block)
    return .023 + max(np.linalg.norm(camera.on_plane(p, STUD_CENTER_Z)[:2] - xy)
                      for p in block.studs_xy)


def destinations(camera, blocks, block, xy, pixel):
    if pixel is not None:
        try:
            point = camera.on_plane(pixel, BENCH_Z)[:2]
        except ValueError as exc:
            raise Miss("destination_off_bench") from exc
        if not (WORKSPACE_X[0] <= point[0] <= WORKSPACE_X[1]
                and WORKSPACE_Y[0] <= point[1] <= WORKSPACE_Y[1]):
            raise Miss("destination_off_bench")
        if np.linalg.norm(point - xy) < .025:
            raise Miss("destination_at_source")
        radius = footprint_radius(camera, block)
        for other in blocks:
            if other is block:
                continue
            clearance = radius + footprint_radius(camera, other) + .01
            if np.linalg.norm(point - center(camera, other)) < clearance:
                raise Miss("destination_occupied")
        return [point]

    centers = [center(camera, b) for b in blocks]
    points = [np.array([x, y]) for x in np.linspace(.20, .42, 12)
              for y in np.linspace(-.04, .20, 13)]
    clear = [p for p in points if all(np.linalg.norm(p - c) > .105 for c in centers)]
    return sorted(clear, key=lambda p: abs(np.linalg.norm(p - xy) - .13))


CLICK_FEEDBACK = {
    "source_outside_detection": "Click inside a detected block's outline.",
    "source_overlaps_detections": "These outlines overlap. Click an unambiguous part of the block.",
    "source_pattern_ambiguous": "These stud detections span more than one block. Choose another outline.",
    "source_color_unknown": "The cameras cannot match this block by color. Choose another block.",
    "no_reachable_overhead_target": "Neither gripper can approach this block. Choose another block.",
    "destination_off_bench": "Choose a spot inside the calibrated bench workspace.",
    "destination_at_source": "Choose a spot at least 25 mm from the block's current center.",
    "destination_occupied": "This block needs more room. Click farther from neighboring blocks.",
    "unreachable_transfer": "Neither arm can reach both poses with this grasp. Try a closer spot.",
    "two_blocks_not_detected": "The overhead camera could not identify two blocks.",
    "choose_distinct_base": "Click the other block to use it as the base.",
    "stack_studs_unobserved": "The cameras could not locate the studs on both blocks.",
}


def run_policy(see, motion_for, detector, args, folder, record, lifted, select_pixels=None):
    rgb, overhead = see("overhead")
    blocks = detector.detect(rgb)
    mark(folder / "overhead.png", rgb, blocks)
    source_pixel, destination_pixel = args.source_pixel, args.destination_pixel

    def plan(source, destination):
        available_space = False
        targets = source_candidates(overhead, blocks, source)
        if not args.overhead_only:
            targets = [b for b in targets if overhead_pattern_ok(overhead, b, STUD_CENTER_Z)]
            if not targets:
                raise Miss("source_pattern_ambiguous")
        for block in targets:
            xy = center(overhead, block)
            yaws = yaw_candidates(overhead, block)
            goals = destinations(overhead, blocks, block, xy, destination)
            available_space |= bool(goals)
            # Try the closest alternatives to the requested transfer length.
            for goal in goals[:5]:
                for arm in ("left", "right"):
                    motion = motion_for(arm)
                    yaw = motion.choose_transfer_yaw(np.r_[xy, BODY_CENTER_Z],
                                                     np.r_[goal, BODY_CENTER_Z], yaws)
                    if yaw is not None:
                        return block, xy, goal, arm, motion, yaw
        raise Miss("unreachable_transfer" if available_space else "no_clear_destination")

    if args.interactive:
        selected_plan = None
        x0, x1 = WORKSPACE_X
        y0, y1 = WORKSPACE_Y
        boundary = [overhead.project([x, y, BENCH_Z]).tolist()
                    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]

        def rejected(pixel, exc):
            reason = str(exc)
            record.setdefault("rejected_clicks", []).append({"pixel": list(pixel), "reason": reason})
            return CLICK_FEEDBACK.get(reason, "Choose another empty bench location.")

        def validate_source(pixel):
            record["clicked_source_pixel"] = list(pixel)
            try:
                block, = source_candidates(overhead, blocks, pixel)
                if not args.overhead_only and not overhead_pattern_ok(overhead, block, STUD_CENTER_Z):
                    raise Miss("source_pattern_ambiguous")
                pose = np.r_[center(overhead, block), BODY_CENTER_Z]
                yaws = yaw_candidates(overhead, block)
                if all(motion_for(arm).choose_yaw(pose, yaws) is None for arm in ("left", "right")):
                    raise Miss("no_reachable_overhead_target")
            except Miss as exc:
                return rejected(pixel, exc)

        def validate_destination(source, destination):
            nonlocal selected_plan
            record["clicked_destination_pixel"] = list(destination)
            try:
                selected_plan = plan(source, destination)
            except Miss as exc:
                return rejected(destination, exc)

        if select_pixels(rgb, blocks, boundary, validate_source, validate_destination) is None:
            raise Miss("selection_cancelled")
        block, xy, dest, arm, motion, yaw = selected_plan
    else:
        if source_pixel is not None:
            record["clicked_source_pixel"] = list(source_pixel)
        if destination_pixel is not None:
            record["clicked_destination_pixel"] = list(destination_pixel)
        block, xy, dest, arm, motion, yaw = plan(source_pixel, destination_pixel)

    record.update(source_xy_m=xy.tolist(), source_pixel=list(block.xy), color=block.color,
                  destination_xy_m=dest.tolist(), arm=arm, jaw_yaw_rad=float(yaw),
                  destination_pixel=overhead.project(np.r_[dest, BENCH_Z]).tolist())
    mark(folder / "overhead.png", rgb, blocks, block.xy, record["destination_pixel"])
    pose = np.r_[xy, BODY_CENTER_Z]
    if not motion.approach(pose, yaw):
        raise Miss("approach_failed")
    if not args.overhead_only:
        rgb, wrist = see(f"{arm}_wrist")
        matches = detector.detect(rgb, color=block.color)
        neighbors = [(b, center(overhead, b)) for b in blocks if b is not block]
        result = refine_grasp(wrist, block, xy, matches, frame_shape=rgb.shape[:2],
                              stud_z=STUD_CENTER_Z, body_z=BODY_CENTER_Z, neighbors=neighbors, overhead=overhead)
        record.update(refinement=result.status, refinement_reason=result.reason)
        match = result.detection
        mark(folder / "wrist.png", rgb, matches, match.xy if match else None)
        if result.xy is None:
            raise Miss(result.status)
        if match is not None:
            record["wrist_xy_m"] = center(wrist, match).tolist()
        pose[:2] = result.xy
        record["grasp_xy_m"] = pose[:2].tolist()
    picked = motion.pick(pose, yaw)
    lifted()
    if not picked:
        raise Miss("pick_failed")
    if not motion.place(np.r_[dest, BODY_CENTER_Z], yaw):
        raise Miss("place_failed")


def outcome_text(record):
    status = record["status"]
    if status in ("pick_failed", "place_failed", "approach_failed", "stack_motion_failed"):
        events = record.get("motion", [])
        if events:
            phase = events[-1]["phase"].replace("_", " ")
            return f"The arm stopped during {phase}."
    if status != "completed":
        detail = record.get("refinement_reason", status) if status.startswith("wrist_") else status
        return "Stopped: " + CLICK_FEEDBACK.get(detail, detail.replace("_", " ")).rstrip(".") + "."
    score = record["score"]
    if "grid_plan" in record:
        return ("The build stayed aligned through the final hold." if score["success"]
                else f"Build failed. {score['reason'].capitalize()}.")
    if "stable_hold" in score and not score["distinct_selections"]:
        return "Stack failed: the evaluator could not identify two different selected blocks."
    if "stable_hold" in score and score["success"]:
        return f"Stacked successfully. Center offset {score['placement_error_m'] * 1000:.1f} mm."
    if score["success"]:
        return (f"Placed successfully. Lift {score['pickup_height_m'] * 1000:.0f} mm, "
                f"landing error {score['placement_error_m'] * 1000:.1f} mm.")
    if score["wrong_object"]:
        return "Placement failed: another block was lifted."
    if not score["pickup_success"]:
        return "Placement failed: the evaluator could not verify a lift of the selected block."
    if "stable_hold" in score:
        if not score["base_on_bench"] or score["base_displacement_m"] > .01 or score["base_rotation_deg"] > 15:
            return "Stack failed: the base moved away from its starting pose."
        if not score["stack_contact"] or not score["seated"]:
            return "Stack failed: the lifted block did not seat on the selected base."
        if not score["stable_hold"]:
            return "Stack failed: the blocks did not stay settled after release."
        return "Stack failed: the blocks fell outside the alignment limits."
    if "on_destination" in score and not score["on_destination"]:
        return "Placement failed: the block did not settle on the requested platform."
    if "on_table" in score and (not score["on_table"] or score["final_height_error_m"] > .006):
        return "Placement failed: the block did not settle at its original tabletop height."
    return f"Placement missed the destination by {score['placement_error_m'] * 1000:.1f} mm."


def run_trial(detector, args, seed, folder):
    folder.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as tmp:
        platforms = getattr(args, "platforms", False)
        stacking = getattr(args, "stack", False)
        blueprint = getattr(args, "blueprint", None)
        grid_build = stacking and (args.blocks > 2 or blueprint is not None)
        if stacking:
            from sim.mega_bloks_stack import build_stack_scene
            from sim.mega_bloks_stack_motion import StackMotion
            from tools.dino_stack_score import StackObserver

            lengths = [block["length"] for block in blueprint["blocks"]] if blueprint else None
            scene, metadata = build_stack_scene(Path(tmp) / "stack.xml", seed=seed, count=args.blocks, lengths=lengths)
        elif platforms:
            from sim.mega_bloks_platforms import build_mega_bloks_platform_scene
            from tools.dino_platform_score import PlatformObserver

            scene, metadata, supports = build_mega_bloks_platform_scene(Path(tmp) / "bench.xml",
                                                                       seed=seed, count=args.blocks)
        else:
            scene, metadata = build_mega_bloks_scene(Path(tmp) / "bench.xml", seed=seed, count=args.blocks)
        auto_calibrate = getattr(args, "auto_calibrate", False)
        if auto_calibrate:
            from tools.dino_passive_camera import PassiveCameras

        sim = PantheraSim(scene)
        sim.reset(randomize=False)
        sim.step(250)
        mujoco.mj_forward(sim.model, sim.data)
        if stacking:
            observer = StackObserver(sim.model, sim.data, metadata)
        elif platforms:
            observer = PlatformObserver(sim.model, sim.data, metadata, supports)
        else:
            observer = PickObserver(sim.model, sim.data, metadata)
        observer.capture("start")
        motion = None
        record = {"seed": seed, "blocks": args.blocks, "status": "completed",
                  "overhead_only": args.overhead_only}
        if stacking:
            record["stack"] = True
        if blueprint:
            record["blueprint"] = blueprint
        lift_captured = False

        with ExitStack() as stack:
            renderer = stack.enter_context(mujoco.Renderer(sim.model, height=512, width=512))
            view = None
            last_sim_frame = last_wall_frame = -float("inf")
            frame_period = 1 / (30 if stacking else 15)

            executor = (stack.enter_context(ThreadPoolExecutor(1, thread_name_prefix="stack-vision"))
                        if stacking else None)
            cameras = (PassiveCameras(sim, renderer, detector, flat_bench=not platforms,
                                      executor=executor) if auto_calibrate else None)
            if cameras:
                stack.callback(cameras.close)
                record["camera_calibration"] = cameras.report
            last_sample = {}
            selection_overhead = None

            def see(name):
                nonlocal selection_overhead
                frame = cameras.read(name) if cameras else read_camera(sim, renderer, name)
                if name == "overhead" and selection_overhead is None:
                    selection_overhead = frame[1]
                return frame

            def image_only(name):
                return cameras.image(name) if cameras else see(name)[0]

            def select_pixels(rgb, blocks, boundary, validate_source, validate_destination, platform_outlines=()):
                nonlocal view
                from tools.dino_pick_ui import InteractiveTrial

                view = InteractiveTrial(rgb, blocks, workspace_outline=boundary,
                                        platform_outlines=platform_outlines, stacking=stacking)
                stack.callback(view.close)
                return view.choose(validate_source, validate_destination)

            fixture_delta = None

            def show_motion(_motion, phase):
                nonlocal fixture_delta, last_sim_frame, last_wall_frame
                if (getattr(args, "nudge", False) and phase == "survey_2"
                        and fixture_delta is None):
                    from sim.mega_bloks_platforms import nudge_platform_block

                    fixture_delta = nudge_platform_block(sim).tolist()
                wrist = f"{sim.active_arm}_wrist"
                if cameras:
                    cameras.poll()
                    names = ("overhead",) if args.overhead_only else ("overhead", wrist)
                    for name in names:
                        if sim.data.time - last_sample.get(name, -float("inf")) >= .65:
                            # After the first lift, some studs belong to elevated
                            # or moved blocks. Only robot landmarks calibrate the mounts.
                            camera_phase = f"build_{phase}" if grid_build and lift_captured else phase
                            cameras.sample(name, camera_phase)
                            last_sample[name] = sim.data.time
                if view is None or not view.is_open:
                    return
                if sim.data.time - last_sim_frame < frame_period - .001:
                    return
                overhead_rgb = image_only("overhead")
                wrist_rgb = image_only(f"{sim.active_arm}_wrist")
                remaining = last_wall_frame + frame_period - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                last_sim_frame, last_wall_frame = sim.data.time, time.monotonic()
                label = f"{sim.active_arm.capitalize()} arm · {phase.capitalize()}"
                view.show(overhead_rgb, wrist_rgb, label)

            def motion_for(arm):
                nonlocal motion
                sim.select_arm(arm)
                controller = StackMotion if stacking else PickPlaceMotion
                on_step = show_motion if cameras or args.interactive or getattr(args, "nudge", False) else None
                motion = controller(sim, on_step=on_step)
                return motion

            def capture_lift():
                nonlocal lift_captured
                if grid_build:
                    observer.capture_build_lift()
                if not lift_captured:
                    observer.capture("lift")
                if motion and any(event["phase"] == "lift" for event in motion.events):
                    image = image_only("overhead")
                    filename = f"lift_{len(record['moves'])}.png" if lift_captured else "lift.png"
                    Image.fromarray(image).save(folder / filename)
                lift_captured = True

            def overlay(source, dest, message):
                if view is not None and view.is_open:
                    view.tracking(source, dest, message)

            try:
                if cameras:
                    cameras.sample("overhead", wait=True)
                    if "overhead" not in cameras.mounts:
                        from sim.mega_bloks_survey import survey

                        # At home the overhead camera may see only the tabletop.
                        # Move the open gripper into view to establish the robot frame.
                        probe = motion_for("left")
                        start_xyz, start_quat = sim.ee_pose()
                        def sample_overhead(phase):
                            cameras.sample("overhead", phase)

                        def calibrated():
                            cameras.sample("overhead", "camera_survey_entry", wait=True)
                            return "overhead" in cameras.mounts

                        if survey(probe, sample_overhead, phase_prefix="camera_survey",
                                  ready_at_entry=calibrated) is None:
                            raise Miss("camera_survey_unreachable")
                        if not probe.move("camera_return", start_xyz, start_quat, 1.):
                            raise Miss("camera_return_unreachable")
                        record["camera_survey"] = probe.events.copy()
                if stacking:
                    from tools.dino_pick_stack import stack_policy

                    stack_policy(see, motion_for, detector, args, folder, record,
                                 capture_lift, select_pixels, overlay, executor=executor,
                                 before_pick=observer.capture_build_start if grid_build else None)
                elif platforms:
                    from tools.dino_pick_elevated import elevated_policy

                    elevated_policy(see, motion_for, detector, args, folder, record,
                                    capture_lift, select_pixels, overlay)
                else:
                    run_policy(see, motion_for, detector, args, folder, record, capture_lift, select_pixels)
            except Miss as exc:
                record["status"] = str(exc)
            if not lift_captured:
                observer.capture("lift")
            sim.step(500)
            mujoco.mj_forward(sim.model, sim.data)
            if stacking:
                observer.capture("release")
                for sample in range(5):
                    sim.step(50)
                    mujoco.mj_forward(sim.model, sim.data)
                    if sample < 4:
                        observer.capture("hold")
                    show_motion(motion, "settled hold")
            observer.capture("final")
            if cameras and "overhead" not in cameras.mounts:
                rgb, overhead = image_only("overhead"), None
            else:
                rgb, overhead = see("overhead")
            mark(folder / "final.png", rgb, destination=record.get("destination_pixel"))
            if cameras:
                cameras.flush()
            record["motion"] = motion.events if motion else []
            if fixture_delta is not None:
                record["fixture_displacement_m"] = fixture_delta
            if stacking:
                def ray(pixel):
                    if pixel is None or selection_overhead is None:
                        return None
                    return selection_overhead.position, selection_overhead.ray(pixel)

                if grid_build:
                    from sim.mega_bloks import PITCH_M

                    placements = []
                    for i, step in enumerate(record.get("grid_plan", [])):
                        pixel = record["source_pixel"] if i == 0 else record["block_pixels"][step["source"]]
                        along, across, level = step["cell"]
                        if blueprint:
                            placements.append((ray(pixel), step["cell"], step["turn"], step["length"]))
                        else:
                            placements.append((ray(pixel), along * PITCH_M, across * PITCH_M, level))
                    score_layout = observer.score_blueprint if blueprint else observer.score_build
                    record["score"] = score_layout(anchor_ray=ray(record.get("destination_pixel")),
                                                   placements=placements)
                    if blueprint:
                        # Photograph the settled build after scoring, from outside the control cameras.
                        camera = mujoco.MjvCamera()
                        centers = np.array([sim.data.body(b["body_name"]).xpos.copy() for b in metadata])
                        camera.lookat[:] = (centers.min(axis=0) + centers.max(axis=0)) / 2
                        camera.distance = max(.34, np.linalg.norm(np.ptp(centers, axis=0)) * 2)
                        camera.azimuth, camera.elevation = 170, -17
                        visibility = mujoco.MjvOption()
                        visibility.sitegroup[:] = 0
                        renderer.update_scene(sim.data, camera=camera, scene_option=visibility)
                        Image.fromarray(renderer.render()).save(folder / "build.png")
                else:
                    record["score"] = observer.score(source_pixel_ray=ray(record.get("source_pixel")),
                                                     destination_ray=ray(record.get("destination_pixel")))
            elif platforms and "destination_pixel" in record:
                # The overhead camera is fixed in world coordinates across the trial.
                record["score"] = observer.score(
                    source_xy_m=record.get("source_xy_m"), destination_xy_m=record.get("destination_xy_m"),
                    source_surface_z_m=record.get("source_surface_z_m"),
                    destination_surface_z_m=record.get("destination_surface_z_m"),
                    source_pixel_ray=(selection_overhead.position, selection_overhead.ray(record["source_pixel"])),
                    destination_ray=(selection_overhead.position, selection_overhead.ray(record["destination_pixel"])),
                )
            elif "destination_xy_m" in record:
                record["score"] = observer.score(source_xy_m=record["source_xy_m"],
                                                 destination_xy_m=record["destination_xy_m"])
            record["outcome"] = outcome_text(record)
            (folder / "trial.json").write_text(json.dumps(record, indent=2) + "\n")
            if view is not None and view.is_open:
                wrist_rgb = image_only(f"{sim.active_arm}_wrist")
                view.finish(rgb, wrist_rgb, record["outcome"])
    return record


def create_run_directory(requested):
    """Reserve an unused sibling directory so retries preserve earlier runs."""
    for attempt in count(1):
        path = requested if attempt == 1 else requested.with_name(f"{requested.name}_{attempt}")
        try:
            path.mkdir(parents=True)
            return path
        except FileExistsError:
            if attempt == 1 and not path.is_dir():
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/megabloks_dinov3/adapter.pt"))
    parser.add_argument("--output", type=Path, required=True, help="Output directory; a numbered sibling is used if it already exists")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--blocks", type=int, choices=range(1, 13))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--platforms", action="store_true",
                      help="Track a block and an empty raised platform at unknown elevations")
    mode.add_argument("--stack", action="store_true",
                      help="Build on a stationary anchor with two to four socketed blocks")
    parser.add_argument("--blueprint", metavar="PRESET_OR_JSON",
                        help="Assemble castle, pyramid, one-stud, or a JSON stud layout (implies --stack)")
    parser.add_argument("--base-blocks", type=int, help="Number of blocks on the bench, including the anchor")
    parser.add_argument("--nudge", action="store_true",
                        help="Move the block 20 mm during the platform survey to test tracking")
    parser.add_argument("--single-estimate", action="store_true",
                        help="Freeze the first valid source and destination estimates for comparison")
    parser.add_argument("--interactive", action="store_true",
                        help="Click the block, then its destination, in the overhead image")
    parser.add_argument("--auto-calibrate", action="store_true",
                        help="Estimate camera mounts and radial distortion from robot motion and DINO stud observations")
    parser.add_argument("--overhead-only", action="store_true",
                        help="Use the overhead grasp estimate alone for a paired comparison")
    parser.add_argument("--source-pixel", type=int, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--destination-pixel", type=int, nargs=2, metavar=("X", "Y"))
    args = parser.parse_args()
    if args.blueprint is not None:
        from sim.mega_bloks_blueprint import load_blueprint

        if args.platforms or args.base_blocks is not None:
            parser.error("--blueprint defines the stack layout and cannot be combined with --platforms or --base-blocks")
        spec = args.blueprint
        try:
            if spec not in ("castle", "pyramid", "one-stud"):
                spec = json.loads(Path(spec).read_text())
            args.blueprint = load_blueprint(spec)
        except (OSError, ValueError) as exc:
            parser.error(f"invalid blueprint: {exc}")
        count = len(args.blueprint["blocks"])
        if args.blocks is not None and args.blocks != count:
            parser.error(f"this blueprint requires {count} blocks")
        args.stack, args.blocks = True, count
    if args.blocks is None:
        args.blocks = 2 if args.stack else 1 if args.platforms else 5
    if args.platforms and (args.blocks != 1 or args.overhead_only):
        parser.error("--platforms requires both cameras and one block for the cross-platform transfer")
    max_blocks = 6 if args.blueprint else 4
    if args.stack and (not 2 <= args.blocks <= max_blocks or args.overhead_only):
        parser.error(f"this stack mode requires the overhead and wrist cameras and 2 to {max_blocks} blocks")
    if args.base_blocks is not None and (not args.stack or not 1 <= args.base_blocks < args.blocks):
        parser.error("--base-blocks requires --stack and must leave at least one block for the tower")
    if (args.single_estimate or args.nudge) and not args.platforms:
        parser.error("--single-estimate and --nudge require --platforms")
    if args.interactive and (args.source_pixel or args.destination_pixel):
        parser.error("choose either interactive selection or pixel arguments")
    if args.trials < 1:
        parser.error("--trials must be positive")
    for pixel in (args.source_pixel, args.destination_pixel):
        if pixel is not None and any(p < 0 or p >= 512 for p in pixel):
            parser.error("overhead pixels must be in [0, 511]")
    detector = DinoPickupPerception(args.checkpoint)
    args.output = create_run_directory(args.output.resolve())
    print(f"Saving trial to {args.output}", flush=True)
    results = []
    for seed in range(args.seed, args.seed + args.trials):
        record = run_trial(detector, args, seed, args.output / f"seed_{seed:04d}")
        score = record.get("score", {})
        row = dict(seed=seed, status=record["status"], pickup=bool(score.get("pickup_success")),
                   success=record["status"] == "completed" and bool(score.get("success")),
                   placement_error_m=score.get("placement_error_m"))
        if args.platforms:
            row.update(direction=score.get("direction"),
                       source_height_error_m=score.get("source_height_error_m"),
                       destination_height_error_m=score.get("destination_height_error_m"),
                       held_offset_applied=record.get("held_offset_applied", False))
        if args.stack and (args.blocks > 2 or args.blueprint):
            block_scores = score.get("blocks", [])
            row.update(placed=sum(b["success"] for b in block_scores),
                       placement_error_m=max((b["placement_error_m"] for b in block_scores), default=None),
                       stable_hold=score.get("stable_hold"))
        elif args.stack:
            row.update(stack_contact=score.get("stack_contact"), seated=score.get("seated"),
                       stable_hold=score.get("stable_hold"))
        results.append(row)
        print(json.dumps(row), flush=True)
        summary = dict(successes=sum(t["success"] for t in results), trials=len(results), results=results)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
