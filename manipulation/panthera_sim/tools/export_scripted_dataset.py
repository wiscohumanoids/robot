#!/usr/bin/env python3
"""Render native demos and export ACT images without retaining duplicate image files.

Uses the existing renderer and LeRobot schema/action alignment. Original rendered
JPEG bytes are embedded in Parquet (no second lossy encoding or PNG expansion).
Only each episode's disposable camera files are removed after embedding; raw
recordings, rendered trajectories, source signatures and the manifest remain.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import lerobot.datasets.lerobot_dataset as lerobot_io
from lerobot.datasets.compute_stats import compute_episode_stats
from datasets.table import embed_table_storage
from teleop.build_lerobot_dataset import features
from teleop.dataset_contract import file_hash, render_signature
from teleop.render_vla_dataset import render_episode, shoulder_camera, wrist_camera
from sim.panthera_env import PantheraSim


def init_renderer(scene):
    global render_sim, renderers, cameras
    render_sim = PantheraSim(scene)
    renderers = [mujoco.Renderer(render_sim.model, height=256, width=256) for _ in range(2)]
    cameras = [shoulder_camera(render_sim.model), wrist_camera(render_sim.model)]


def render_job(source, dest, settings):
    count = render_episode(source, dest, render_sim, renderers, cameras, 30, 92, settings)
    image_features = {k:v for k,v in features(256,256).items() if v['dtype']=='image'}
    image_buffer = {key:[str(dest/key.rsplit('.',1)[-1]/f'{j:05d}.jpg')
                         for j in range(count-1)] for key in image_features}
    return count, compute_episode_stats(image_buffer, image_features)


def embed_batched(dataset):
    """Same upstream embedding operation, amortized over whole Arrow batches."""
    original_format = dataset.format
    return dataset.with_format('arrow').map(
        embed_table_storage, batched=True, batch_size=256).with_format(**original_format)


def save_prepared_episode(dataset, image_stats):
    """Reuse identical image statistics computed in the rendering workers.

    LeRobot 0.4.4 does not expose precomputed stats or batch embedding settings.
    Scope these two adapters to this synchronous save; restore library functions
    immediately afterward. Numeric statistics still use the upstream function.
    """
    def prepared_stats(buffer, schema):
        numeric = {k:v for k,v in schema.items() if v['dtype']!='image'}
        return {**compute_episode_stats({k:buffer[k] for k in numeric}, numeric), **image_stats}
    with patch.object(lerobot_io, 'compute_episode_stats', prepared_stats), \
         patch.object(lerobot_io, 'embed_images', embed_batched):
        dataset.save_episode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/scripted_stack')
    parser.add_argument('--rendered', type=Path, default=ROOT/'outputs/scripted_stack_rendered_30hz')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/lerobot/panthera_scripted_stack_30hz')
    parser.add_argument('--min-free-gb', type=float, default=5.)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.rendered.exists():
        raise SystemExit('Output/rendered directory already exists; use fresh paths to avoid overwriting data.')
    episodes = sorted(args.input.glob('episode_*/data.npz'))
    if not episodes:
        raise SystemExit('No source episodes')
    from sim.dynamics import recorded_dynamics
    metadata = [json.loads(p.with_name('meta.json').read_text()) for p in episodes]
    modes = {recorded_dynamics(m) for m in metadata}
    if len(modes) != 1:
        raise SystemExit('Export each simulation dynamics version separately.')
    dynamics = modes.pop()
    scenes = {m.get('scene', 'sim/panthera/scene.xml') for m in metadata}
    if len(scenes) != 1:
        raise SystemExit('Export one scene per dataset.')
    scene = ROOT / scenes.pop()
    status = args.input/'status.json'
    if status.exists() and not json.loads(status.read_text())['complete']:
        raise SystemExit('Wait for collection to complete before exporting.')
    args.rendered.mkdir(parents=True)
    settings = render_signature(30, 256, 92)
    dataset = LeRobotDataset.create(repo_id='local/panthera_scripted_stack', root=args.output,
        fps=30, robot_type='panthera_ht_sim', features=features(256,256), use_videos=False,
        image_writer_threads=0, metadata_buffer_size=20)
    total = 0
    sources = {}
    started = time.time()
    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=init_renderer,
        initargs=(scene,),
        mp_context=multiprocessing.get_context('spawn'))
    pending = {}
    def submit(i):
        source = episodes[i].parent
        pending[i] = pool.submit(render_job, source, args.rendered/source.name, settings)
    for i in range(min(len(episodes), args.workers*2)):
        submit(i)
    try:
        for i, source in enumerate(episodes):
            if shutil.disk_usage(args.output).free < args.min_free_gb*1024**3:
                raise RuntimeError('Low disk space; export stopped safely.')
            dest = args.rendered/source.parent.name
            count, image_stats = pending.pop(i).result()
            with np.load(dest/'trajectory.npz') as data:
                q, ctrl = data['q'], data['ctrl']
                if not np.allclose(np.diff(data['t']), 1/30, atol=1e-7):
                    raise ValueError('Nonuniform timestamps')
            # LeRobot's native episode buffer accepts image paths; its embedding
            # step reads the JPEG bytes directly. No decode/re-encode is needed.
            buffer = dataset.episode_buffer
            task = metadata[i].get('task', 'stack the three colored cubes')
            buffer.update(size=count-1, task=[task]*(count-1),
                timestamp=np.arange(count-1)/30, frame_index=np.arange(count-1))
            buffer['observation.state'] = np.c_[q[:-1],ctrl[:-1,6]].astype(np.float32)
            buffer['action'] = ctrl[1:].astype(np.float32)
            for camera in ('shoulder','wrist'):
                buffer[f'observation.images.{camera}'] = [
                    str(dest/camera/f'{j:05d}.jpg') for j in range(count-1)]
            save_prepared_episode(dataset, image_stats)
            # The standalone dataset now owns embedded image bytes. Keep the
            # compact rendered arrays/provenance for training validation/replay.
            for camera in ('shoulder','wrist'):
                shutil.rmtree(dest/camera)
            sources[source.parent.name] = json.loads((dest/'source.json').read_text())
            total += count
            progress = dict(episodes=i+1, requested=len(episodes), frames=total-i-1,
                elapsed_s=time.time()-started, free_gb=shutil.disk_usage(args.output).free/1024**3)
            (args.output/'export_status.json').write_text(json.dumps(progress,indent=2)+'\n')
            print(json.dumps(progress), flush=True)
            next_index = i + args.workers*2
            if next_index < len(episodes):
                submit(next_index)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        dataset.finalize()
        dataset.stop_image_writer()
    manifest = dict(format='robot-arm-learning-vla-render-v2', render_signature=settings,
        episodes=list(sources), num_episodes=len(episodes), num_frames=total, sample_hz=30,
        image_size=[256,256], cameras={'primary':'shoulder','wrist':'wrist'},
        image_storage='embedded in LeRobot Parquet; rerender raw episodes to restore JPEG cache')
    (args.rendered/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    provenance = dict(simulation_dynamics=dynamics, version=2, fps=30, rendered_root=str(args.rendered.resolve()),
        render_manifest_sha256=file_hash(args.rendered/'manifest.json'),
        converter_sha256=file_hash(ROOT/'teleop/build_lerobot_dataset.py'),
        scripted_exporter_sha256=file_hash(Path(__file__)),
        image_encoding='original renderer JPEG bytes', state_gripper='command',
        action_alignment='next_uniform_sample', sources=sources)
    (args.output/'meta/provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (args.output/'COMPLETE').write_text(f'{len(episodes)} episodes, {total-len(episodes)} frames\n')


if __name__ == '__main__':
    main()
