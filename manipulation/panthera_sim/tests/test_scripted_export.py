"""Ensure the compact exporter preserves LeRobot's pixels, actions and stats."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.compute_stats import compute_episode_stats
from teleop.build_lerobot_dataset import features
from tools.export_scripted_dataset import save_prepared_episode


class ScriptedExportTest(unittest.TestCase):
    def test_prepared_save_matches_upstream_and_preserves_jpeg_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for i in range(4):
                path = root/f'{i}.jpg'
                pixels = np.random.default_rng(i).integers(0,256,(16,16,3),dtype=np.uint8)
                Image.fromarray(pixels).save(path, quality=92)
                paths.append(str(path))
            schema = features(16,16)
            cameras = {k:v for k,v in schema.items() if v['dtype']=='image'}
            image_stats = compute_episode_stats({k:paths for k in cameras}, cameras)
            results = []
            for mode in ('upstream','prepared'):
                dataset = LeRobotDataset.create(repo_id='local/test', root=root/mode,
                    fps=30, features=schema, use_videos=False, image_writer_threads=0)
                buffer = dataset.episode_buffer
                buffer.update(size=4, task=['stack']*4, timestamp=np.arange(4)/30,
                    frame_index=np.arange(4))
                buffer['observation.state'] = np.arange(28,dtype=np.float32).reshape(4,7)/30
                buffer['action'] = buffer['observation.state']+.01
                for key in cameras:
                    buffer[key] = paths.copy()
                if mode=='prepared':
                    save_prepared_episode(dataset,copy.deepcopy(image_stats))
                else:
                    dataset.save_episode()
                dataset.finalize()
                table = pq.read_table(sorted((root/mode).glob('data/*/*.parquet')))
                results.append((table,json.loads((root/mode/'meta/stats.json').read_text())))
                for key in cameras:
                    for i, cell in enumerate(table[key].to_pylist()):
                        self.assertEqual(cell['bytes'], Path(paths[i]).read_bytes())
            self.assertTrue(results[0][0].equals(results[1][0]))
            self.assertEqual(results[0][1],results[1][1])


if __name__ == '__main__':
    unittest.main()
