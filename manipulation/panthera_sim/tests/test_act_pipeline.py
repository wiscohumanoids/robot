import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from teleop.dataset_contract import PhysicsClock, resolve_control_hz, source_signature
from teleop.render_vla_dataset import interpolate, recording_times, reconstruct_fingers, cache_valid
from sim.panthera_env import PantheraSim


class PipelineTest(unittest.TestCase):
    def test_fractional_physics_rate_does_not_drift(self):
        clock = PhysicsClock(30, .002)
        counts = [clock.next_steps() for _ in range(300)]
        self.assertEqual(sum(counts), 5000)
        self.assertEqual(set(counts), {16, 17})

    def test_recorded_clock_and_legacy_floor(self):
        t, method = recording_times({'sim_time': np.array([9., 9.02, 9.04])}, .002)
        np.testing.assert_allclose(t, [0, .02, .04])
        t, method = recording_times({'t': np.array([0., .027, .054])}, .002)
        np.testing.assert_allclose(t, [0, .026, .052])

    def test_uniform_interpolation_and_quaternion_sign(self):
        t = np.array([0., .026, .052, .078])
        grid = np.arange(3) / 30
        np.testing.assert_allclose(interpolate(t, 2*t, grid), 2*grid)
        q = np.array([[1., 0, 0, 0], [-1., 0, 0, 0]])
        out = interpolate([0, 1], q, [.5], True)
        self.assertAlmostEqual(abs(out[0, 0]), 1.)

    def test_open_legacy_fingers_do_not_render_closed(self):
        sim = PantheraSim()
        positions, quat = sim.object_poses()
        source = {'q': np.tile(sim.q, (3, 1)), 'ctrl': np.tile(sim.data.ctrl, (3, 1)),
                  'obj_pos': np.tile(positions, (3, 1, 1)), 'obj_quat': np.tile(quat, (3, 1, 1))}
        source['ctrl'][:, 6] = .04
        fingers, method = reconstruct_fingers(source, [0., .03, .06], sim)
        self.assertEqual(method, 'reconstructed_dynamics')
        self.assertTrue(np.all(fingers[:, 0] > .035))
        self.assertTrue(np.all(fingers[:, 1] < -.035))
        source['finger_q'] = np.array([[.01, -.01]] * 3)
        np.testing.assert_array_equal(reconstruct_fingers(source, [0., .03, .06], sim)[0], source['finger_q'])

    def test_source_mutation_changes_fingerprint_and_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'data.npz').write_bytes(b'first')
            (root / 'meta.json').write_text('{}')
            before = source_signature(root)
            (root / 'data.npz').write_bytes(b'other')
            self.assertNotEqual(before, source_signature(root))
            self.assertFalse(cache_valid(root, before, {}))

    def test_cached_episode_requires_matching_settings_and_complete_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(root / 'trajectory.npz', q=np.zeros((2, 6)))
            for camera in ('shoulder', 'wrist'):
                (root / camera).mkdir()
                for i in range(2):
                    (root / camera / f'{i:05d}.jpg').write_bytes(b'image')
            signature, settings = {'data.npz': 'hash'}, {'sample_hz': 30}
            (root / 'source.json').write_text(json.dumps({
                'frames': 2, 'source_signature': signature, 'render_signature': settings}))
            self.assertTrue(cache_valid(root, signature, settings))
            self.assertFalse(cache_valid(root, {'data.npz': 'changed'}, settings))
            self.assertFalse(cache_valid(root, signature, {'sample_hz': 10}))
            (root / 'wrist/00001.jpg').unlink()
            self.assertFalse(cache_valid(root, signature, settings))

    def test_recorder_preserves_actual_fingers_and_simulation_time(self):
        from teleop.episode import Episode, FIELDS
        episode = Episode()
        row = {key: 0. for key in FIELDS}
        row.update(sim_time=12., finger_q=np.array([.02, -.02]),
                   finger_dq=np.array([.1, -.1]), physics_steps=13)
        episode.add(row)
        with tempfile.TemporaryDirectory() as folder:
            episode.save(Path(folder), {}, 30., False)
            with np.load(Path(folder) / 'data.npz') as data:
                np.testing.assert_array_equal(data['finger_q'], [[.02, -.02]])
                self.assertEqual(data['sim_time'][0], 12.)
                self.assertEqual(data['physics_steps'][0], 13)

    def test_checkpoint_rate_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'deployment.json').write_text(json.dumps({'fps': 30}))
            self.assertEqual(resolve_control_hz(root, None, 30), 30)
            with self.assertRaises(ValueError):
                resolve_control_hz(root, 10, 30)
            with self.assertRaises(ValueError):
                resolve_control_hz(root, None, 10)


if __name__ == '__main__':
    unittest.main()
