import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.act_scene import fixed_environment, reset_fixed_arm, rollout_metrics


class FixedSceneTest(unittest.TestCase):
    def test_fixed_pose_survives_random_layout_resets(self):
        from sim.panthera_env import PantheraSim
        root = Path(__file__).resolve().parents[1]
        sim = PantheraSim(root / 'sim/panthera/scene_two_blocks.xml')
        q = [0., 1.5423167027421607, 1.6388978774456795, -1.0564857076996368, 0., 0.]
        positions = []
        for seed in (10, 20):
            sim.reset(rng=np.random.default_rng(seed))
            reset_fixed_arm(sim, dict(initial_arm_q=q, gripper_open_m=.04))
            np.testing.assert_array_equal(sim.q, q)
            np.testing.assert_array_equal(sim.data.qvel, 0.)
            np.testing.assert_allclose(sim.data.qpos[sim.finger_qadr], [.04, -.04])
            np.testing.assert_allclose(sim.ee_pos(), [.38, 0., .24], atol=.001)
            positions.append(sim.object_poses()[0])
        self.assertFalse(np.array_equal(*positions))

    def test_two_block_success_requires_red_on_green_on_table(self):
        valid = np.array([[.4, .1, .1175], [.4, .1, .0725]])
        self.assertTrue(rollout_metrics(valid)['task_stack'])
        self.assertFalse(rollout_metrics(valid)['three_stack'])
        self.assertFalse(rollout_metrics(valid[::-1])['task_stack'])
        misaligned = valid.copy()
        misaligned[0, 0] += .02
        self.assertFalse(rollout_metrics(misaligned)['task_stack'])
        raised = valid.copy()
        raised[:, 2] += .05
        self.assertFalse(rollout_metrics(raised)['task_stack'])
        three = np.vstack((valid, [.4, .1, .1625]))
        self.assertTrue(rollout_metrics(three)['three_stack'])
        self.assertTrue(rollout_metrics(three)['task_stack'])

    def test_provenance_rejects_inconsistent_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / str(i) for i in range(2)]
            meta = dict(scene='scene.xml', task='stack', objects=['red', 'green'],
                        arm_start='fixed', initial_arm_q=[0.] * 6, gripper_open_m=.04)
            for path in paths:
                path.mkdir()
                (path / 'meta.json').write_text(json.dumps(meta))
            provenance = {'sources': {str(i): {'source': str(path)} for i, path in enumerate(paths)}}
            self.assertEqual(fixed_environment(provenance)['initial_arm_q'], [0.] * 6)
            meta['initial_arm_q'][0] = 1.
            (paths[1] / 'meta.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'inconsistent'):
                fixed_environment(provenance)


if __name__ == '__main__':
    unittest.main()
