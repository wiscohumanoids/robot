"""Render disjoint synthetic scenes with visible body, stud, and instance labels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from PIL import Image

from sim.mega_bloks import build_mega_bloks_scene, SPAWN_XY_LOW, SPAWN_XY_HIGH, COLORS


WRIST_POSES = {
    'left_wrist': [.52789042, 1.74600666, 1.38345771, -1.53131903, .96792353, -.08564781],
    'right_wrist': [1.31764382, 1.17027725, .79932277, -.46694697, .63345598, -1.64842591],
}


def label_maps(model):
    """Map renderer geom IDs to semantic classes and block instances."""
    semantic = np.zeros(model.ngeom, dtype=np.uint8)
    instance = np.zeros(model.ngeom, dtype=np.uint16)
    names = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ''
        if name.startswith('mega_block_'):
            prefix = name.split('_body')[0].split('_stud')[0]
            if prefix not in names:
                names.append(prefix)
            semantic[geom_id] = 2 if '_stud_' in name else 1
            instance[geom_id] = names.index(prefix) + 1
    return semantic, instance


def stud_centers(model, segmentation, semantic_lut, instance_lut):
    """Return one centroid per visible physical stud, in zero-based pixel coordinates.

    Coordinates average the integer column and row indices of every visible
    pixel belonging to that stud geom. Occluded studs have no entry.
    """
    valid = segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
    result = []
    for geom_id in np.unique(segmentation[..., 0][valid]):
        if geom_id < 0 or semantic_lut[geom_id] != 2:
            continue
        y, x = np.nonzero(valid & (segmentation[..., 0] == geom_id))
        result.append({'instance_id': int(instance_lut[geom_id]),
                       'geom_name': model.geom(int(geom_id)).name,
                       'center_xy': [float(x.mean()), float(y.mean())],
                       'visible_pixels': int(x.size)})
    return result


def generate(args):
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f'{destination} exists; choose a new output directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name + '.generating-', dir=destination.parent) as temporary:
        _generate(args, Path(temporary), destination)


def _generate(args, output, destination):
    scene_path, blocks = build_mega_bloks_scene(output / 'scene.xml', seed=args.seed, count=12, randomize=False)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.vis.global_.offwidth = max(args.size, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(args.size, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    semantic_lut, instance_lut = label_maps(model)
    block_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b['body_name']) for b in blocks]
    joints = [int(model.jnt_qposadr[model.body_jntadr[b]]) for b in block_ids]
    initial = model.key_qpos[model.key('home').id].copy()
    base_rgba = model.geom_rgba.copy()
    base_light = model.light_diffuse.copy()
    palette = list(COLORS)
    materials = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, 'mega_plastic_' + color) for color in palette]
    manifest = {'classes': {'0': 'background', '1': 'body', '2': 'stud'},
                'seed': args.seed, 'size': [args.size, args.size], 'splits': {},
                'scene': 'wood dual Panthera', 'blocks': blocks}
    visibility = mujoco.MjvOption()
    visibility.geomgroup[5] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    with mujoco.Renderer(model, height=args.size, width=args.size) as renderer:
        for split_index, (split, count) in enumerate(zip(('train', 'val', 'test'), args.counts)):
            directory = output / split
            directory.mkdir(exist_ok=True)
            entries = []
            for index in range(count):
                scene_seed = args.seed + sum(args.counts[:split_index]) + index
                rng = np.random.default_rng(scene_seed)
                data.qpos[:] = initial
                data.qvel[:] = 0
                model.geom_group[instance_lut > 0] = 2
                model.geom_rgba[:] = base_rgba
                model.light_diffuse[:] = base_light * rng.uniform(.7, 1.2)
                active_count = 0 if index % 11 == 0 else int(rng.integers(4, 9))
                active = set(rng.choice(len(blocks), active_count, replace=False).tolist())
                radii = [float(np.linalg.norm(block['dimensions_m'][:2])) / 2 for block in blocks]
                # Circumscribed circles keep rotated rectangular bodies separated.
                for attempt in range(100):
                    positions = {}
                    for block_index in sorted(active, key=lambda i: -radii[i]):
                        for _ in range(1000):
                            xy = rng.uniform(SPAWN_XY_LOW, SPAWN_XY_HIGH)
                            if all(np.linalg.norm(xy - previous) > radii[block_index] + radii[other] + .004
                                   for other, previous in positions.items()):
                                positions[block_index] = xy
                                break
                        else:
                            break
                    if len(positions) == active_count:
                        break
                else:
                    raise RuntimeError('Cannot place blocks without overlap')
                frame_blocks = []
                for block_index, address in enumerate(joints):
                    if block_index not in active:
                        model.geom_group[instance_lut == block_index + 1] = 5
                        data.qpos[address:address + 3] = [0, 0, -2 - block_index]
                        continue
                    data.qpos[address:address + 2] = positions[block_index]
                    angle = rng.uniform(-np.pi, np.pi)
                    data.qpos[address + 3:address + 7] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
                    color_index = int(rng.integers(len(palette)))
                    geom_ids = np.flatnonzero(instance_lut == block_index + 1)
                    model.geom_matid[geom_ids] = materials[color_index]
                    frame_blocks.append({'instance_id': block_index + 1, 'color': palette[color_index],
                                         'qpos': data.qpos[address:address + 7].tolist()})
                camera_name = ('overhead', 'random_free', 'left_wrist', 'overhead', 'right_wrist')[index % 5]
                arm_pose = None
                if camera_name in WRIST_POSES:
                    prefix = '' if camera_name == 'left_wrist' else 'right_'
                    addresses = [model.jnt_qposadr[model.joint(prefix + 'joint' + str(i)).id] for i in range(1, 7)]
                    arm_pose = np.array(WRIST_POSES[camera_name]) + rng.uniform(-.025, .025, 6)
                    data.qpos[addresses] = arm_pose
                mujoco.mj_forward(model, data)
                if camera_name in WRIST_POSES and any(
                        contact.dist < -.001 and
                        all(instance_lut[g] == 0 or int(instance_lut[g]) - 1 in active
                            for g in (contact.geom1, contact.geom2))
                        for contact in data.contact):
                    data.qpos[addresses] = initial[addresses]
                    camera_name, arm_pose = 'overhead', None
                    mujoco.mj_forward(model, data)
                camera.lookat[:] = [.30 + rng.uniform(-.035, .035), rng.uniform(-.03, .03), .05]
                camera.distance = rng.uniform(.68, .92)
                camera.azimuth = rng.uniform(65, 115)
                camera.elevation = rng.uniform(-82, -48)
                render_camera = camera if camera_name == 'random_free' else camera_name
                renderer.disable_segmentation_rendering()
                renderer.update_scene(data, camera=render_camera, scene_option=visibility)
                rgb = renderer.render().copy()
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera=render_camera, scene_option=visibility)
                segmentation = renderer.render().copy()
                valid = (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (segmentation[..., 0] >= 0)
                ids = np.clip(segmentation[..., 0], 0, model.ngeom - 1)
                mask = np.where(valid, semantic_lut[ids], 0).astype(np.uint8)
                instance = np.where(valid, instance_lut[ids], 0).astype(np.uint16)
                visible_ids = set(np.unique(instance[instance > 0]).tolist())
                if not visible_ids <= {block_index + 1 for block_index in active}:
                    raise RuntimeError(f'Inactive block rendered in {split} scene {scene_seed}')
                stem = f'{index:05d}'
                paths = {key: f'{split}/{stem}_{suffix}.png' for key, suffix in
                         (('image', 'rgb'), ('mask', 'semantic'), ('instance', 'instance'))}
                for key, array in (('image', rgb), ('mask', mask), ('instance', instance)):
                    Image.fromarray(array).save(output / paths[key])
                entries.append({**paths, 'scene_seed': scene_seed, 'block_count': active_count, 'blocks': frame_blocks, 'camera_name': camera_name, 'arm_pose_rad': None if arm_pose is None else arm_pose.tolist(),
                                'visible_instances': sorted(visible_ids),
                                'studs': stud_centers(model, segmentation, semantic_lut, instance_lut),
                                'camera': ({'lookat': camera.lookat.tolist(), 'distance': camera.distance,
                                            'azimuth': camera.azimuth, 'elevation': camera.elevation}
                                           if camera_name == 'random_free' else
                                           {'name': camera_name, 'position_m': data.cam_xpos[model.camera(camera_name).id].tolist(),
                                            'rotation_matrix': data.cam_xmat[model.camera(camera_name).id].reshape(3, 3).tolist(),
                                            'fovy_degrees': float(model.cam_fovy[model.camera(camera_name).id])})})
                if (index + 1) % 20 == 0:
                    print(f'{split}: {index + 1}/{count}', flush=True)
            manifest['splits'][split] = entries
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    output.rename(destination)
    print(destination / 'manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/megabloks')
    parser.add_argument('--counts', type=int, nargs=3, default=(400, 80, 80), metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=73019)
    args = parser.parse_args()
    if args.size < 16 or args.size % 16:
        parser.error('--size must be a positive multiple of 16')
    if any(count < 1 for count in args.counts):
        parser.error('--counts must contain three positive integers')
    generate(args)


if __name__ == '__main__':
    main()
