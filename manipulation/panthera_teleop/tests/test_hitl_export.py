"""A real LeRobot export/loader audit with two separate correction segments."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'app'))


@unittest.skipUnless(importlib.util.find_spec('lerobot'), 'Run with work/postprocess-venv/bin/python for LeRobot export')
class ExportTests(unittest.TestCase):
    def test_corrections_export_and_audit_without_crossing_policy_boundaries(self):
        self.export_fixture(False)

    def test_slowed_policy_exports_policy_time_and_preserves_host_timestamps(self):
        self.export_fixture(True)

    def export_fixture(self, slowed):
        from PIL import Image
        from prepare_hitl import prepare
        from postprocess_kept import export, validate_hitl_selection, source_path
        from verify_processed import verify
        from verify_full_rate import verify as verify_archives
        from remote_inference import CONTRACT
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'raw/kept'
            episode = source/'rollout1'
            episode.mkdir(parents=True)
            origin = 1000000000
            states = []
            for i in range(600):
                t=i/150
                owner='human' if 1 <= t < 2 or 2.6 <= t < 3.6 else 'policy'
                states.append(dict(sequence=i, monotonic_ns=origin+round(t*1e9*(3 if slowed else 1)),
                    episode_time_s=t*(3 if slowed else 1),
                    control_source='policy' if slowed else owner, arms={side:dict(position_rad=[.001*i]*6,
                    velocity_rad_s=[.05 if slowed else .15]*6, measured_torque_nm=[0.]*6,
                    gripper=dict(position=.5,velocity=0.,torque=0.)) for side in ('left','right')}))
                if slowed:
                    states[-1].update(data_monotonic_ns=origin+round(t*1e9), data_time_s=t,
                                      data_clock_rate=1/3, recording_segment=0)
            metadata=dict(id='rollout1',label='kept',complete=True,start_monotonic_ns=origin,
                end_monotonic_ns=states[-1]['monotonic_ns'],samples=len(states),
                context=dict(mode='human_in_the_loop',contract=CONTRACT,checkpoint='test-checkpoint',
                    rollout=dict(operator='tester',scene='fixture')), outcome=dict(success=True,quality=4))
            if slowed:
                metadata.update(schema_version=2, start_data_monotonic_ns=origin)
            image=io.BytesIO()
            Image.new('RGB',(640,480),'#234567').save(image,format='JPEG')
            frames=[]
            for camera in CONTRACT['image_crops_xyxy']:
                directory=episode/'images'/camera
                directory.mkdir(parents=True)
                for i in range(80):
                    relative=f'images/{camera}/{i:07d}.jpg'
                    (episode/relative).write_bytes(image.getvalue())
                    frames.append(dict(camera=camera,frame_index=i,sequence=i,path=relative,
                        monotonic_ns=origin+i*50000000*(3 if slowed else 1),
                        episode_time_s=i/20*(3 if slowed else 1),width=640,height=480))
                    if slowed:
                        frames[-1].update(data_monotonic_ns=origin+i*50000000, data_time_s=i/20,
                                          recording_segment=0)
            metadata['camera_frames']={c:80 for c in CONTRACT['image_crops_xyxy']}
            (episode/'metadata.json').write_text(json.dumps(metadata))
            for name, values in [('states',states),('frames',frames)]:
                (episode/f'{name}.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
            annotations=root/'annotations.json'
            selected=prepare(source,annotations,ROOT/'config/postprocessing-gear-pin-sept30.json')
            self.assertEqual(len(selected['episodes']),1 if slowed else 2)
            self.assertEqual({s['source_id'] for s in selected['episodes']},{'rollout1'})
            self.assertEqual({s['selection'] for s in selected['episodes']},
                             {'successful_unassisted' if slowed else 'human_correction'})
            with self.assertRaises(ValueError):
                validate_hitl_selection(metadata,dict(selection='human_correction'),states)
            with self.assertRaisesRegex(ValueError, 'recording pause'):
                validate_hitl_selection(metadata, dict(selection='human_correction'), [
                    dict(control_source='human', recording_segment=i) for i in (0,1)])
            with self.assertRaises(ValueError):
                source_path(source,dict(id='safe',source_id='../escape'))
            args=argparse.Namespace(source=source,annotations=annotations,output=root/'dataset',
                trimmed_output=root/'trimmed',repo_id='local/hitl-fixture')
            with contextlib.redirect_stdout(io.StringIO()):
                export(args)
                verify(args.output,source)
                verify_archives(args.output,source)
            info=json.loads((args.output/'meta/info.json').read_text())
            self.assertEqual(info['total_episodes'],1 if slowed else 2)
            self.assertGreater(info['total_frames'],20)
            self.assertTrue(json.loads((args.output/'verification.json').read_text())['episode_boundary_padding_verified'])
            if slowed:
                import pyarrow.parquet as pq
                manifest = json.loads((args.output/'processing/manifest.json').read_text())
                self.assertAlmostEqual(manifest['episodes'][0]['original_duration_s'], 4., delta=.01)
                self.assertAlmostEqual(manifest['episodes'][0]['original_wall_duration_s'], 12., delta=.03)
                self.assertLess(info['total_frames'], 81)
                row = pq.read_table(next((args.output/'data').rglob('*.parquet'))).to_pylist()[0]
                self.assertAlmostEqual(row['observation.velocity'][0], .15, places=6)
