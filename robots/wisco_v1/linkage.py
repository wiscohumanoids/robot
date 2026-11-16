"""Joint <-> motor mapping for the two parallel mechanisms (ankles, waist).

The URDF/MJCF model these as plain two-axis joints at the real pivots.  The
real robot drives each with two motors through cranks and ball-ended rods.
This file converts joint torque / velocity at a given joint angle pair into
the torque / velocity of each motor, using the rod geometry measured from CAD
(linkage.json, zero pose, base frame, metres).

    from linkage import Mechanism
    m = Mechanism('left_ankle')
    tau_motor = m.motor_torque(q=(0.1, 0.0), tau_joint=(20.0, 0.0))   # N*m, [motor A, motor B]
    amps = tau_motor / KT                                             # your torque constant
"""
import json, os, numpy as np
def _rot(axis, ang):
    a = np.asarray(axis, float); a = a/np.linalg.norm(a)
    K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return np.eye(3)+np.sin(ang)*K+(1-np.cos(ang))*K@K
def _about(p, pt, axis, ang):
    pt = np.asarray(pt, float); return _rot(axis, ang)@(np.asarray(p, float)-pt)+pt
class Mechanism:
    def __init__(self, name, path=None):
        d = json.load(open(path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'linkage.json')))
        self.m = d[name]; self.joints = self.m['joints']
    def _move(self, p, q):
        m = self.m
        return _about(_about(p, m['axis2_pt'], m['axis2_dir'], q[1]), m['axis1_pt'], m['axis1_dir'], q[0])
    def _err(self, c, th, q):
        b = _about(c['crank_ball'], c['axis_pt'], c['axis_dir'], th)
        if self.m['crank_on'] == 'fixed': f = self._move(c['far_ball'], q)
        else: b = self._move(b, q); f = np.asarray(c['far_ball'], float)
        return np.linalg.norm(b-f)-c['rod_len']
    def motor_angles(self, q):
        """Crank angle of each motor (rad, relative to the crank position drawn in CAD)."""
        out = []
        for c in self.m['cranks']:
            t0, t1 = 0.0, 0.05
            e0, e1 = self._err(c, t0, q), self._err(c, t1, q)
            for _ in range(60):
                if abs(e1) < 1e-10 or e1 == e0: break
                t0, t1, e0 = t1, t1-e1*(t1-t0)/(e1-e0), e1
                e1 = self._err(c, t1, q)
            out.append(t1)
        return np.array(out)
    def jacobian(self, q, h=1e-5):
        """d(motor angle)/d(joint angle), 2x2.  Rows: motors.  Columns: joints (order of self.joints)."""
        q = np.asarray(q, float); J = np.zeros((2, 2))
        for k in range(2):
            d = np.zeros(2); d[k] = h
            J[:, k] = (self.motor_angles(q+d)-self.motor_angles(q-d))/(2*h)
        return J
    def motor_torque(self, q, tau_joint):
        return np.linalg.solve(self.jacobian(q).T, np.asarray(tau_joint, float))
    def motor_velocity(self, q, qd):
        return self.jacobian(q)@np.asarray(qd, float)
    def joint_torque_limit(self, q, motor_torque_limit):
        """Largest pure torque on each joint when both motors are at +/- the given limit."""
        return np.abs(self.jacobian(q)).sum(0)*motor_torque_limit
if __name__ == '__main__':
    for n in ('left_ankle', 'right_ankle', 'waist'):
        m = Mechanism(n); J = m.jacobian((0, 0))
        print(n, m.joints, '\n  d(motor)/d(joint) at zero pose:\n ', np.round(J, 3).tolist())
