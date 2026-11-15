import unittest
import numpy as np
from tools.audit_act_observability import collision_report


class ObservabilityTest(unittest.TestCase):
    def setUp(self):
        self.states=np.array([[0]*7,[1]*7,[0]*7],dtype=np.float32)
        self.actions=np.array([[0]*7,[0]*7,[2]*7],dtype=np.float32)
        self.images=[[{'bytes':b'a'},{'bytes':b'b'},{'bytes':b'a'}]]

    def test_conflicting_identical_observations_have_positive_lower_bound(self):
        report=collision_report(self.states,self.actions,self.images,np.ones(7),chunk=1)
        self.assertEqual(report['contradictory_frames'],2)
        self.assertAlmostEqual(report['whole_episode_l1_lower_bound_from_exact_duplicates'],2/3)
        self.assertEqual(report['contradictory_groups'][0]['frames'],[0,2])

    def test_distinct_images_disambiguate_identical_states(self):
        self.images[0][2]={'bytes':b'c'}
        report=collision_report(self.states,self.actions,self.images,np.ones(7),chunk=1)
        self.assertEqual(report['contradictory_frames'],0)

    def test_padding_does_not_create_extra_constraints(self):
        report=collision_report(self.states,self.actions,self.images,np.ones(7),chunk=2)
        # Five valid actions total. Only horizon zero has both duplicate inputs.
        self.assertAlmostEqual(report['whole_episode_l1_lower_bound_from_exact_duplicates'],2/5)


if __name__=='__main__':unittest.main()
