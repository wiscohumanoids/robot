import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from neutral_hold import NeutralHold
import test_recording as helpers


class HoldTests(unittest.TestCase):
    def setUp(self):
        self.settings=json.loads((ROOT/'config/recording.json').read_text())['neutral_return']
        self.targets={s:np.array([0,.25,.25,0,0,0]) for s in ('left','right')}
        self.hold=NeutralHold(self.targets,self.settings,0)
        self.arms={s:dict(position_rad=q.tolist(),measured_torque_nm=[1,3,4,.4,.1,.1])
                   for s,q in self.targets.items()}
        for t in np.arange(0,.7,.01):self.hold.tick(float(t),self.arms)

    def test_stays_held_without_operator_load(self):
        for t in np.arange(.7,8,.01):
            self.assertFalse(any(self.hold.tick(float(t),self.arms).values()))
        np.testing.assert_allclose(self.hold.targets['left'],self.targets['left'])

    def test_one_handle_releases_only_that_arm_and_latches(self):
        self.arms['left']['measured_torque_nm'][1]+=1.5
        for t in np.arange(.7,1.0,.01):ready=self.hold.tick(float(t),self.arms)
        self.assertEqual(ready,dict(left=True,right=False))
        self.arms['left']['position_rad'][1]=1.5  # Released arm can move far from neutral.
        self.arms['left']['measured_torque_nm'][1]=3
        self.assertEqual(self.hold.tick(1.1,self.arms),dict(left=True,right=False))
        self.arms['right']['measured_torque_nm'][4]+=.4
        for t in np.arange(1.2,1.5,.01):ready=self.hold.tick(float(t),self.arms)
        self.assertTrue(all(ready.values()))

    def test_long_idle_hold_still_allows_independent_handle_release(self):
        for t in (60, 600, 3600, 86400):
            self.assertEqual(self.hold.tick(t,self.arms),dict(left=False,right=False))
        self.arms['left']['measured_torque_nm'][1]+=1.5
        self.hold.tick(86400.1,self.arms)
        self.assertEqual(self.hold.tick(86400.3,self.arms),dict(left=True,right=False))

    def test_short_load_spike_does_not_release(self):
        self.arms['left']['measured_torque_nm'][1]+=2
        self.hold.tick(.7,self.arms)
        self.hold.tick(.8,self.arms)
        self.arms['left']['measured_torque_nm'][1]-=2
        self.assertFalse(any(self.hold.tick(.81,self.arms).values()))
        self.assertFalse(any(self.hold.tick(1.5,self.arms).values()))

    def test_gravity_load_and_small_noise_do_not_release(self):
        for i,t in enumerate(np.arange(.7,3,.01)):
            self.arms['left']['measured_torque_nm'][1]=3+(.15 if i%2 else -.15)
            self.assertFalse(any(self.hold.tick(float(t),self.arms).values()))

    def test_deviation_and_nonfinite_feedback_rejected(self):
        self.arms['left']['position_rad'][0]=.1
        with self.assertRaisesRegex(RuntimeError,'tracking deviation'):self.hold.tick(1,self.arms)
        self.arms['left']['position_rad'][0]=0
        self.arms['left']['measured_torque_nm'][0]=float('nan')
        with self.assertRaisesRegex(RuntimeError,'invalid hold torque'):self.hold.tick(1,self.arms)


class HoldRecordingTests(unittest.TestCase):
    setUp=helpers.EpisodeTests.setUp
    tearDown=helpers.EpisodeTests.tearDown
    sample=helpers.EpisodeTests.sample
    trigger=helpers.EpisodeTests.trigger
    decision=helpers.EpisodeTests.decision

    def test_hold_is_excluded_and_first_release_rearms_recording(self):
        identifier=self.trigger()
        self.config['neutral_return']['enabled']=True
        self.decision('kept',identifier)
        self.store.submit('return_complete',dict(hold_for_ready=True))
        self.store.events.join()
        for _ in range(30):self.sample(.25,.0)
        self.assertEqual(self.store.snapshot()['phase'],'holding_neutral')
        self.assertIsNone(self.store.snapshot()['active'])
        self.assertEqual(len(self.store.pre_roll),0)
        sample=dict(monotonic_ns=self.t,arms={s:dict(position_rad=[.25]*6,
                    velocity_rad_s=[0]*6,gripper=dict(position=0,velocity=0)) for s in ('left','right')})
        self.store.submit('handoff_ready',dict(sample=sample,ready=dict(left=True,right=False)))
        self.store.events.join()
        self.assertEqual(self.store.snapshot()['phase'],'armed')
        for _ in range(5):self.sample(.30,.1)
        self.assertEqual(self.store.snapshot()['phase'],'recording')
        self.assertEqual(self.store.snapshot()['handoff_ready'],dict(left=True,right=False))


if __name__=='__main__':unittest.main()
