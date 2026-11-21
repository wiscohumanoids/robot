"""Replay physical stud, body-corner, and visible-edge labels with exact mask checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import shutil

os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from generate_mega_bloks_segmentation import label_maps, stud_centers
from mega_bloks_landmarks import body_landmarks


def replay(dataset: Path, split: str, output: Path):
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(f'Output parent does not exist: {output.parent}')
    edge_directory = 'landmark_labels' if split == 'all' else f'landmark_labels_{split}'
    if (dataset / edge_directory).exists():
        raise FileExistsError(dataset / edge_directory)
    with tempfile.TemporaryDirectory(prefix='landmark-labels.generating-', dir=dataset) as temporary:
        _replay(dataset, split, output, Path(temporary), edge_directory)


def _replay(dataset, split, output, staging, edge_directory):
    manifest_path = dataset / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    model = mujoco.MjModel.from_xml_path(str(dataset / 'scene.xml'))
    height, width = manifest['size']
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    semantic_lut, instance_lut = label_maps(model)
    initial = model.key_qpos[model.key('home').id].copy()
    visibility = mujoco.MjvOption()
    visibility.geomgroup[5] = 0
    frames = []
    selected = [(name, entry) for name, entries in manifest['splits'].items() if split == 'all' or name == split for entry in entries]
    edge_root = staging / edge_directory
    if edge_root.exists():
        raise FileExistsError(edge_root)
    edge_root.mkdir()
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for split_name, entry in selected:
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            data.qpos[:] = initial
            data.qvel[:] = 0
            model.geom_group[instance_lut > 0] = 5
            active = {block['instance_id']: block for block in entry['blocks']}
            for index, block in enumerate(manifest['blocks'], 1):
                body = model.body(block['body_name']).id
                address = int(model.jnt_qposadr[model.body_jntadr[body]])
                if index in active:
                    data.qpos[address:address + 7] = active[index]['qpos']
                    model.geom_group[instance_lut == index] = 2
                else:
                    data.qpos[address:address + 3] = [0, 0, -1 - index]
            camera_name = entry['camera_name']
            if entry['arm_pose_rad'] is not None:
                prefix = '' if camera_name == 'left_wrist' else 'right_'
                for number, value in enumerate(entry['arm_pose_rad'], 1):
                    address = model.jnt_qposadr[model.joint(prefix + 'joint' + str(number)).id]
                    data.qpos[address] = value
            mujoco.mj_forward(model, data)
            if camera_name == 'random_free':
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                for key in ('lookat', 'distance', 'azimuth', 'elevation'):
                    if key == 'lookat':
                        camera.lookat[:] = entry['camera'][key]
                    else:
                        setattr(camera, key, entry['camera'][key])
            else:
                camera = camera_name
            renderer.update_scene(data, camera=camera, scene_option=visibility)
            segmentation = renderer.render().copy()
            valid = (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (segmentation[..., 0] >= 0)
            ids = np.clip(segmentation[..., 0], 0, model.ngeom - 1)
            semantic = np.where(valid, semantic_lut[ids], 0).astype(np.uint8)
            instance = np.where(valid, instance_lut[ids], 0).astype(np.uint16)
            for key, recovered in (('mask', semantic), ('instance', instance)):
                stored = np.asarray(Image.open(dataset / entry[key]))
                if not np.array_equal(stored, recovered):
                    difference = int(np.count_nonzero(stored != recovered))
                    raise RuntimeError(f'{entry[key]} replay differs at {difference} pixels')
            studs = stud_centers(model, segmentation, semantic_lut, instance_lut)
            renderer.disable_segmentation_rendering()
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=camera, scene_option=visibility)
            depth = renderer.render().copy()
            active_metadata = [{**manifest['blocks'][index - 1], 'instance_id': index} for index in active]
            landmarks = body_landmarks(model, data, renderer.scene, segmentation, depth,
                                       active_metadata, width, height)
            edge_path = Path(edge_directory) / split_name / (Path(entry['image']).stem.replace('_rgb', '_edges') + '.png')
            (staging / edge_path).parent.mkdir(exist_ok=True)
            edge_mask = Image.new('L', (width, height))
            draw = ImageDraw.Draw(edge_mask)
            for edge in landmarks['edge_details']:
                for fragment in edge['fragments']:
                    draw.line([tuple(np.rint(point).astype(int)) for point in fragment], fill=255, width=1)
            body_band = Image.fromarray((semantic == 1).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))
            edge_mask = Image.fromarray(np.where(np.asarray(body_band) > 0, np.asarray(edge_mask), 0).astype(np.uint8))
            edge_mask.save(staging / edge_path)
            frame = {key: entry[key] for key in ('image', 'mask', 'instance', 'scene_seed', 'camera_name')}
            frame.update(landmarks)
            frame.update({'split': split_name, 'studs': studs,
                          'stud_centers': [stud['center_xy'] for stud in studs], 'edges': str(edge_path),
                          'edges_sha256': hashlib.sha256((staging / edge_path).read_bytes()).hexdigest()})
            frames.append(frame)
            if len(frames) % 20 == 0:
                print(f'{len(frames)}/{len(selected)} frames verified and labeled', flush=True)
    result = {'schema_version': 1, 'label_protocol_version': 2, 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'split': split, 'image_size': [height, width],
              'scene_sha256': hashlib.sha256((dataset / 'scene.xml').read_bytes()).hexdigest(),
              'mujoco_version': mujoco.__version__,
              'corner_index_signs_xyz': [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
              'visibility': {'neighborhood_radius_px': 1, 'depth_tolerance': 'max(0.0006m, 1.5 * pixel footprint at query depth)',
                             'edges': 'true cuboid edges; fragments contain projected first/last visible samples',
                             'dense_edges': 'nearest-pixel rasterization clipped to a one-pixel rendered body neighborhood'},
              'coordinate_convention': 'zero-based [column, row] pixel indices; studs average visible pixels; corners and edges project 3-D geometry',
              'mask_verification': {'semantic_exact_frames': len(frames), 'instance_exact_frames': len(frames)},
              'frames': frames}
    if output.exists():
        raise FileExistsError(output)
    # Stage JSON beside its destination so publication also works across filesystems.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='.landmark-labels-',
                                     suffix='.json', dir=output.parent, delete=False) as stream:
        staged_output = Path(stream.name)
        json.dump(result, stream, indent=2)
        stream.write('\n')
    final_edges = dataset / edge_directory
    published_edges = False
    try:
        edge_root.rename(final_edges)
        published_edges = True
        staged_output.rename(output)
    except BaseException:
        if published_edges:
            shutil.rmtree(final_edges)
        raise
    finally:
        staged_output.unlink(missing_ok=True)
    print(f'{output}: {len(frames)} frames, {sum(len(frame["studs"]) for frame in frames)} visible physical studs; masks exact')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path(__file__).resolve().parents[1] / 'data/megabloks')
    parser.add_argument('--split', choices=('all', 'train', 'val', 'test'), default='all')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    replay(args.dataset, args.split, args.output or args.dataset / ('landmarks.json' if args.split == 'all' else f'{args.split}_landmarks.json'))


if __name__ == '__main__':
    main()
