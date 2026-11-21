"""Compare clean and perturbed RGB in closed-loop simulated stacking trials."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import torch

from perception.camera_augmentation import CONDITIONS, augment_rgb
from sim import mega_bloks_stack
from tools import dino_pick_place
from tools.dino_pick_camera import read_camera
from tools.dino_pick_stack import _StackObservations


CAMERAS = ("overhead", "left_wrist", "right_wrist")


def relighting(seed):
    """Sample a side light aimed at the blocks and a plain matte laminate worktop."""
    rng = np.random.default_rng([seed, 95279])
    azimuth = rng.uniform(0, 2 * np.pi)
    height = rng.uniform(.65, 1.0)
    offset = np.array([.65 * np.cos(azimuth), .65 * np.sin(azimuth), height])
    temperature = rng.uniform(-1, 1)
    light_rgb = np.array([1 + .12 * temperature, 1, 1 - .12 * temperature])
    gray = rng.uniform(.25, .70)
    return {
        "light_position_m": (np.array([.23, .125, .08]) + offset).tolist(),
        "light_direction": (-offset / np.linalg.norm(offset)).tolist(),
        "light_diffuse": (rng.uniform(.55, .85) * light_rgb).tolist(),
        "headlight_ambient": [.22] * 3, "headlight_diffuse": [.20] * 3,
        "table_rgba": [gray, gray, gray, 1.],
    }


def write_relit_scene(path, settings):
    tree = ET.parse(path)
    light = tree.find("worldbody/light")
    headlight = tree.find("visual/headlight")
    table = tree.find("asset/material[@name='photo_bench']")
    for element, attrs in ((light, {"pos": "light_position_m", "dir": "light_direction",
                                    "diffuse": "light_diffuse"}),
                           (headlight, {"ambient": "headlight_ambient", "diffuse": "headlight_diffuse"}),
                           (table, {"rgba": "table_rgba"})):
        for attribute, key in attrs.items():
            element.set(attribute, " ".join(map(str, settings[key])))
    table.attrib.pop("texture")
    table.set("specular", ".04")
    tree.write(path, encoding="utf-8")


def run_condition(detector, layout_seed, condition, latency_s, directory, settings):
    def read_exposure(sim, renderer, name):
        rgb, camera = read_camera(sim, renderer, name)
        # Lock photometry, blur and the sensor noise field to each camera for
        # the whole rollout. Image perturbations preserve the calibration rays.
        rng = np.random.default_rng([layout_seed, CAMERAS.index(name), 14963])
        return augment_rgb(rgb, condition, rng), camera

    def poll_at_sim_time(observations):
        # Inference still runs on its worker. Use the same capture-to-delivery
        # delay for both checkpoints regardless of host speed or GPU contention.
        while observations.pending and observations.sim_time() >= observations.pending[0][2] + latency_s:
            observations._receive()

    args = argparse.Namespace(stack=True, blocks=2, interactive=False, overhead_only=False,
                              auto_calibrate=False, source_pixel=None, destination_pixel=None)
    with ExitStack() as patches:
        if condition == "relit":
            original = mega_bloks_stack.build_stack_scene

            def build_relit(*args, **kwargs):
                path, metadata = original(*args, **kwargs)
                write_relit_scene(path, settings)
                return path, metadata

            patches.enter_context(patch.object(mega_bloks_stack, "build_stack_scene", build_relit))
        else:
            patches.enter_context(patch.object(dino_pick_place, "read_camera", read_exposure))
        patches.enter_context(patch.object(_StackObservations, "poll", poll_at_sim_time))
        return dino_pick_place.run_trial(detector, args, layout_seed, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2600, 2601, 2602, 2603])
    parser.add_argument("--conditions", choices=(*CONDITIONS, "relit"), nargs="+", default=["clean", "mixed"])
    parser.add_argument("--latency-s", type=float, default=.4,
                        help="Detection time in simulated seconds after each camera exposure")
    args = parser.parse_args()
    if not np.isfinite(args.latency_s) or args.latency_s < 0 or min(args.seeds) < 0:
        parser.error("Latency and layout seeds must be nonnegative and finite")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.conditions)) != len(args.conditions):
        parser.error("Each seed and condition must appear once")
    torch.set_num_threads(2)
    checkpoint = args.checkpoint.resolve()
    detector = dino_pick_place.DinoPickupPerception(checkpoint)
    output = dino_pick_place.create_run_directory(args.output.resolve())
    paths = [checkpoint, checkpoint.with_name("landmarks.pt")]
    summary = {
        "checkpoints": {p.name: sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()},
        "seeds": args.seeds, "conditions": args.conditions,
        "fixed_detection_latency_s": args.latency_s, "camera_calibration": "simulated intrinsics and pose",
        "results": [],
    }
    for seed in args.seeds:
        for condition in args.conditions:
            settings = relighting(seed) if condition == "relit" else None
            trial = run_condition(detector, seed, condition, args.latency_s,
                                  output / condition / f"seed_{seed:04d}", settings)
            score = trial["score"]
            row = dict(seed=seed, condition=condition, status=trial["status"], pickup=score["pickup_success"],
                       success=trial["status"] == "completed" and score["success"],
                       placement_error_m=score["placement_error_m"])
            if settings is not None:
                row["scene_appearance"] = settings
            summary["results"].append(row)
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
