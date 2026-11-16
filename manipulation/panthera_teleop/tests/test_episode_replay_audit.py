"""Saved-action replay boundaries and interruption through the real web controls."""
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'app'))
import episode_replay_audit as replay
import inference_dashboard as dashboard
from hitl_demo import DemoHardware


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        actions = np.zeros((55, 14))
        actions[:, 0] = np.linspace(0, .025, 55)
        np.savez(self.root / 'episode.npz', states=np.zeros_like(actions), actions=actions)
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps(dict(fps=20, episodes=[dict(episode=7, frames=55,
                                                                     file='episode.npz')])))

    def test_validation_rejects_wrong_start_limits_and_exhaustion(self):
        source = replay.ReplaySource(self.manifest)
        source.select(7)
        with self.assertRaisesRegex(RuntimeError, 'starting pose'):
            source.infer(dict(sequence=0, state=[.2]*14))
        first = source.infer(dict(sequence=0, state=[0.]*14))
        second = source.infer(dict(sequence=1, state=[0.]*14))
        self.assertEqual(len(first['actions']), 50)
        self.assertEqual(len(second['actions']), 5)
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            source.infer(dict(sequence=2, state=[0.]*14))
        with self.assertRaisesRegex(RuntimeError, 'Repeated'):
            source.infer(dict(sequence=0, state=[0.]*14))
        bad = np.zeros((2,14)); bad[0,0] = 3
        np.savez(self.root/'episode.npz', states=np.zeros_like(bad), actions=bad)
        with self.assertRaisesRegex(RuntimeError, 'outside'):
            replay.ReplaySource(self.manifest)

    def setup_web(self):
        port = free_port()
        threading.Thread(target=replay.serve, args=(self.manifest, port), daemon=True).start()
        source_url = f'http://127.0.0.1:{port}'
        for _ in range(100):
            try:
                health = replay.call(source_url, '/health'); break
            except OSError:
                time.sleep(.01)
        else:
            self.fail('Source did not start')
        controller = dashboard.Controller(source_url, DemoHardware,
            log_path=self.root/'steps.jsonl', data_dir=self.root/'raw')
        controller.connect(False)
        for _ in range(200):
            if controller.snapshot()['mode'] == 'holding': break
            time.sleep(.01)
        server = dashboard.make_server(controller, free_port())
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        def cleanup():
            controller.shutdown(); controller.worker.join(3)
            server.shutdown(); server.server_close()
        self.addCleanup(cleanup)
        return source_url, f'http://127.0.0.1:{server.server_port}', controller

    def test_complete_replay_records_final_partial_chunk_and_keeps_hold(self):
        source, web, controller = self.setup_web()
        replay.run_episode(source, web, 7, self.root/'results')
        self.assertEqual(controller.snapshot()['steps_completed'], 2)
        self.assertTrue(controller.snapshot()['connected'])
        np.testing.assert_allclose(controller.hardware.position[0], .025, atol=1e-6)
        result=json.loads((self.root/'results/episode_000007-result.json').read_text())
        self.assertTrue(result['completed'])
        for _ in range(100):
            paths=list((self.root/'raw/review').glob('*/metadata.json'))
            if paths: break
            time.sleep(.01)
        metadata=json.loads(paths[0].read_text())
        self.assertEqual(metadata['context']['checkpoint'], replay.call(source,'/health')['checkpoint'])
        self.assertFalse(metadata['outcome']['success'])

    def test_operator_stop_prevents_a_second_chunk(self):
        source, web, controller = self.setup_web()
        def stop_after_motion():
            for _ in range(500):
                if controller.snapshot()['mode']=='executing':
                    controller.stop(); return
                time.sleep(.005)
        thread=threading.Thread(target=stop_after_motion);thread.start()
        with self.assertRaisesRegex(RuntimeError, 'interrupted|canceled'):
            replay.run_episode(source, web, 7, self.root/'results')
        thread.join(3)
        self.assertEqual(controller.snapshot()['steps_completed'], 0)
        self.assertTrue(controller.snapshot()['connected'])
        self.assertEqual(replay.call(source,'/status')['cursor'], 50)


if __name__ == '__main__':
    unittest.main()
