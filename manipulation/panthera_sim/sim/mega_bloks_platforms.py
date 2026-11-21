"""Build two plain platforms at different heights on the Panthera bench."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .mega_bloks import BODY_HEIGHT_M, COLORS, SHAPES, TABLE_Z, _build_xml, _numbers, _write_scene


_PLATFORM_NAMES = ("transfer_platform_a", "transfer_platform_b")
_PLATFORM_HALF_SIZE_M = (.075, .095)


def build_mega_bloks_platform_scene(output_path, *, seed=0, count=1):
    """Return the scene path, evaluator block metadata, and evaluator support geom names.

    Each side can be higher. With one block, even seeds start on the lower
    platform and odd seeds start on the higher platform. Two blocks occupy one
    platform each. The plain matte platforms have visible edges in RGB. The
    returned geometry metadata is for the evaluator.
    """
    if not isinstance(count, int) or count not in (1, 2):
        raise ValueError("count must be 1 or 2")
    rng = np.random.default_rng(seed)
    # Offset the platforms sideways so both tops are visible from the overhead camera.
    centers = np.array([[.237, .078], [.414, .208]]) + rng.uniform(-.008, .008, size=(2, 2))
    low = TABLE_Z + rng.uniform(.023, .037)
    high = TABLE_Z + rng.uniform(.052, .070)
    lower_index = int(rng.integers(2))
    heights = [low, high] if lower_index == 0 else [high, low]
    first_index = lower_index if seed % 2 == 0 else 1 - lower_index
    colors = tuple(COLORS)
    placements = []
    for platform in (first_index, 1 - first_index)[:count]:
        xy = centers[platform] + rng.uniform([-.012, -.018], [.012, .018])
        z = heights[platform] + BODY_HEIGHT_M / 2 + .0002
        nx, ny = SHAPES[int(rng.integers(len(SHAPES)))]
        yaw = float(rng.uniform(-np.pi, np.pi))
        color = colors[int(rng.integers(len(colors)))]
        placements.append((nx, ny, [*xy, z], yaw, color))
    root, metadata = _build_xml(placements)
    root.set("model", "panthera_wood_mega_bloks_platforms")
    world = root.find("worldbody")
    finishes = rng.permutation([False, True])
    for center, surface_z, name, light_finish in zip(centers, heights, _PLATFORM_NAMES, finishes):
        half_height = (surface_z - TABLE_Z) / 2
        ET.SubElement(world, "geom", name=name, type="box",
                      pos=_numbers([*center, TABLE_Z + half_height]),
                      size=_numbers([*_PLATFORM_HALF_SIZE_M, half_height]),
                      rgba=".18 .185 .19 1", friction=".8 .02 .001",
                      contype="8", conaffinity="5", group="2")
        # The thin finish over the support box leaves its top and friction unchanged.
        ET.SubElement(world, "geom", type="plane", pos=_numbers([*center, surface_z + .0001]),
                      size=_numbers([*_PLATFORM_HALF_SIZE_M, .1]),
                      rgba=".57 .59 .61 1" if light_finish else ".29 .305 .32 1",
                      contype="0", conaffinity="0", group="2")
    return _write_scene(root, output_path), metadata, _PLATFORM_NAMES


def nudge_platform_block(sim, *, distance_m=.020):
    """Translate the single settled block across its pad and return its true XY shift.

    The benchmark calls this disturbance outside the policy. The shift is
    recorded by the evaluator and does not inform visual tracking or motion.
    """
    if not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("distance_m must be finite and positive")
    model, data = sim.model, sim.data
    joints = [model.joint(i) for i in range(model.njnt)
              if model.joint(i).name.startswith("mega_block_")]
    if len(joints) != 1:
        raise ValueError("The disturbance requires exactly one block")
    joint, = joints
    geom = model.geom(joint.name + "_body")
    mujoco.mj_forward(model, data)
    center = data.geom_xpos[geom.id]
    orientation = data.geom_xmat[geom.id].reshape(3, 3)
    extent = np.abs(orientation) @ geom.size
    for name in _PLATFORM_NAMES:
        pad = model.geom(name)
        pad_position = data.geom_xpos[pad.id]
        offset = center[:2] - pad_position[:2]
        room = pad.size[:2] - extent[:2]
        gap = center[2] - extent[2] - (pad_position[2] + pad.size[2])
        if abs(gap) <= .006 and np.all(np.abs(offset) <= room):
            break
    else:
        raise ValueError("The block must be resting within a platform")

    # Try axis-aligned shifts in order of landing clearance from the pad edges.
    # A small gap keeps blocks away from the side if they settle after the move.
    candidates = distance_m * np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    clearance = (room - np.abs(offset + candidates)).min(axis=1)
    best = int(np.argmax(clearance))
    if clearance[best] < .002:
        raise ValueError("The requested disturbance does not fit on the block's pad")
    shift = candidates[best]
    position_start = int(joint.qposadr[0])
    data.qpos[position_start:position_start + 2] += shift
    mujoco.mj_forward(model, data)
    return shift
