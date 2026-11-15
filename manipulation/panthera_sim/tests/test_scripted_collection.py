"""Integration checks for physical scripted demos and native dataset compatibility."""
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from sim.panthera_env import PantheraSim
from sim.stack_task import stack_metrics
from teleop.episode import FIELDS
from teleop.render_vla_dataset import recording_times, interpolate
from teleop.dataset_contract import PhysicsClock
from tools.collect_scripted import Planner
from tools.validate_scripted_dataset import replay
from sim.stack_task import ordered_two_stack_metrics


class ScriptedCollectionTest(unittest.TestCase):
    def test_two_blocks_fixed_start_and_replay(self):
        initial = []
        layouts = []
        for seed in (230924, 230925, 230926):
            planner = Planner(seed, blocks=2, arm_start='fixed')
            self.assertTrue(planner.run()['two_stack'])
            self.assertEqual(planner.sim.object_names, ['cube_red', 'cube_green'])
            self.assertFalse(any(s['name'].startswith('2_') for s in planner.stages))
            initial.append(planner.episode.rows[0]['q'])
            layouts.append(planner.episode.rows[0]['obj_pos'])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                planner.episode.save(path, {'scene': 'sim/panthera/scene_two_blocks.xml'}, 30, False)
                result = replay(path)
                self.assertTrue(result['success'], result)
                self.assertEqual(result['grasped_blocks'], [0])
        for q in initial[1:]:
            np.testing.assert_array_equal(q, initial[0])
        self.assertFalse(np.allclose(layouts[0], layouts[1]))

    def test_two_stack_requires_correct_order_alignment_and_table_support(self):
        positions = np.array([[.4, 0., .1175], [.4, 0., .0725]])
        self.assertTrue(ordered_two_stack_metrics(positions)['two_stack'])
        self.assertFalse(ordered_two_stack_metrics(positions[::-1])['two_stack'])
        self.assertFalse(ordered_two_stack_metrics(positions + [0, 0, .1])['two_stack'])
        positions[0, 0] += .02
        self.assertFalse(ordered_two_stack_metrics(positions)['two_stack'])

    def test_successful_demo_survives_native_save_and_act_clock_replay(self):
        planner = Planner(230927)
        self.assertTrue(planner.run()['three_stack'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            planner.episode.save(path, {'seed': planner.seed}, 30, False)
            with np.load(path/'data.npz') as raw:
                self.assertTrue(set(FIELDS).issubset(raw.files))
                self.assertTrue({'sim_time','finger_q','finger_dq','physics_steps'}.issubset(raw.files))
                for value in raw.values():
                    self.assertTrue(np.isfinite(value).all())
                self.assertLessEqual(np.abs(np.diff(raw['ctrl'][:,:6],axis=0)).max(), .0800001)
                sim = PantheraSim()
                t, method = recording_times(raw, sim.dt)
                self.assertEqual(method, 'recorded_simulation_time')
                grid = np.arange(0, t[-1]+1e-9, 1/30)
                actions = interpolate(t, raw['ctrl'], grid)
                sim.reset(randomize=False)
                sim.data.qpos[sim.arm_qadr] = raw['q'][0]
                sim.data.qvel[sim.arm_dofadr] = raw['dq'][0]
                sim.data.qpos[sim.finger_qadr] = raw['finger_q'][0]
                sim.data.qvel[sim.finger_dofadr] = raw['finger_dq'][0]
                sim.set_object_poses(raw['obj_pos'][0], raw['obj_quat'][0])
                sim.data.ctrl[:] = actions[0]
                sim.sync_control_state()
                mujoco.mj_forward(sim.model, sim.data)
                clock = PhysicsClock(30, sim.dt)
                stable = 0
                for action in actions[1:]:
                    sim.set_arm_ctrl(action[:6])
                    sim.set_gripper(action[6]/.04)
                    sim.step(clock.next_steps())
                    success = stack_metrics(sim.object_poses()[0])['three_stack']
                    released = not sim.grasped
                    stable = stable+1 if success and released else 0
                self.assertGreaterEqual(stable, 30)


if __name__ == '__main__':
    unittest.main()
