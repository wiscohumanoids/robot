"""Deterministic checks of saved state/image clocks and omitted hold boundaries."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from hitl_recording import RolloutStore


class RecordingClockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        config = json.loads((ROOT / 'config/recording.json').read_text())
        config.update(minimum_free_bytes=0, control_hz=100)
        self.store = RolloutStore(self.temp.name, config, {})
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.store.close, 'test')
        self.send('rollout_begin', dict(sample=dict(monotonic_ns=1_000_000_000), details={}))

    def send(self, kind, payload):
        self.store.submit(kind, payload)
        self.store.events.join()
        self.assertIsNone(self.store.failure)

    def sample(self, i, source='policy', origin=1_000_000_000):
        raw = origin + i*10_000_000
        frame = dict(camera='overhead', sequence=i, monotonic_ns=raw-5_000_000, jpeg=b'jpeg')
        self.send('sample', dict(monotonic_ns=raw, sequence=i, control_source=source,
            policy_time_s=i*.005, policy_clock_rate=.5, inference_sequence=0,
            camera_frames={'overhead': frame}, arms={}))

    def saved(self):
        self.send('rollout_end', dict(label='kept'))
        directory = next((Path(self.temp.name) / 'kept').iterdir())
        return {kind: [json.loads(l) for l in (directory / f'{kind}.jsonl').read_text().splitlines()]
                for kind in ('states', 'frames')}

    def test_frames_preserve_receive_time_and_map_to_the_policy_clock(self):
        for i in range(101):
            self.sample(i)
        rows = self.saved()
        self.assertEqual(len(rows['states']), 51)
        self.assertGreater(len(rows['frames']), 5)
        for frame in rows['frames']:
            elapsed = frame['monotonic_ns']-1_000_000_000
            self.assertEqual(frame['data_monotonic_ns'], 1_000_000_000 + elapsed//2)
        for a, b in zip(rows['frames'], rows['frames'][1:]):
            self.assertGreaterEqual(b['data_monotonic_ns'] - a['data_monotonic_ns'], 50_000_000)

    def test_holds_write_neither_states_nor_images_and_do_not_advance_time(self):
        for i in range(101):
            self.sample(i, 'hold')
        self.assertEqual(self.store.snapshot()['duration'], 0)
        self.assertFalse(self.store.snapshot()['collecting'])
        self.assertEqual(self.saved(), dict(states=[], frames=[]))

    def test_human_guidance_resumes_in_a_separate_segment_without_wait_time(self):
        for i in range(11):
            self.sample(i, 'human')
        for i in range(11, 101):
            self.sample(i, 'hold')
        for i in range(101, 112):
            self.sample(i, 'human')
        rows = self.saved()
        self.assertEqual({s['recording_segment'] for s in rows['states']}, {0, 1})
        self.assertAlmostEqual(rows['states'][-1]['data_time_s'], .21)
        self.assertAlmostEqual(rows['states'][-1]['episode_time_s'], 1.11)
        self.assertTrue(all(s['control_source'] == 'human' for s in rows['states']))
