"""Exercise controller-to-IK target mapping without opening robot devices."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
root = Path(__file__).resolve().parents[1]
scripts=root/'work/Panthera-HT_SDK/panthera_python/scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('dual',scripts/'7_vr_dual_cartesian_control.py')
dual=importlib.util.module_from_spec(spec)
spec.loader.exec_module(dual)

class FakeRobot:
    def gripper_control_MIT(self,*args): pass
    def inverse_kinematics(self,**kwargs):
        self.target=kwargs
        return None


def simulate(rotation, displacement=np.zeros(3)):
    robot=FakeRobot()
    start=np.array([0.2,1.0,-0.3])
    initial_hand={'pos':start,'rot':[0,0,0,1],'pose_valid':1,'squeeze':1.0,'trigger':0.0,'primary':0}
    state=dict(robot=robot,label='test',control_activated=False,last_hand_pos=None,last_hand_rot=None,
               target_position=np.array([0.3,0.0,0.3]),target_rotation=np.eye(3),
               last_primary_pressed=0,last_gripper_pos=0.0,last_valid_joint_pos=np.zeros(6))
    dual.process_arm_control(state,initial_hand)
    assert np.allclose(state['target_rotation'],np.eye(3)), 'Engagement must not rotate'
    moved=dict(initial_hand,pos=start+displacement,rot=rotation.as_quat())
    dual.process_arm_control(state,moved)
    return state

# XR X-axis wrist pitch maps to robot Y. Reversing pitch changes its sign only.
for degrees in (-10,10):
    state=simulate(R.from_euler('x',degrees,degrees=True))
    assert np.allclose(R.from_matrix(state['target_rotation']).as_rotvec(),[0,np.deg2rad(degrees)*dual.rot_scale,0])
# Other wrist axes preserve the original mapping.
for axis, expected in [('y',np.array([0,0,1.])),('z',np.array([-1.,0,0]))]:
    state=simulate(R.from_euler(axis,10,degrees=True))
    assert np.allclose(R.from_matrix(state['target_rotation']).as_rotvec(),expected*np.deg2rad(10)*dual.rot_scale)
# Pure translation is unchanged and generates no rotation.
delta=np.array([0.01,0.02,0.03])
state=simulate(R.identity(),delta)
assert np.allclose(state['target_position'],[0.27,-0.01,0.32])
assert np.allclose(state['target_rotation'],np.eye(3))
print('PASS: pitch reversed both directions; roll/yaw and translation unchanged; no clutch-engagement rotation.')
