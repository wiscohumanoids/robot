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
    """Rotation matrix of `ang` about `axis` (Rodrigues)."""
    a = np.asarray(axis, float); a = a/np.linalg.norm(a)
    K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return np.eye(3)+np.sin(ang)*K+(1-np.cos(ang))*K@K
def _about(p, pt, axis, ang):
    """Rotate point `p` by `ang` about the axis through `pt`."""
    pt = np.asarray(pt, float); return _rot(axis, ang)@(np.asarray(p, float)-pt)+pt
class Mechanism:
    """One rod linkage (an ankle or the waist) from linkage.json: joint angles <-> crank angles."""
    def __init__(self, name, path=None):
        """Load mechanism `name` from linkage.json."""
        d = json.load(open(path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'linkage.json')))
        self.m = d[name]; self.joints = self.m['joints']
    def _move(self, p, q):
        """Move a point with the joint angles `q`."""
        m = self.m
        return _about(_about(p, m['axis2_pt'], m['axis2_dir'], q[1]), m['axis1_pt'], m['axis1_dir'], q[0])
    def _err(self, c, th, q):
        """Rod length error for crank angle `th` at joint angles `q` (zero when the linkage closes)."""
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
                if abs(e1) < 1e-14 or e1 == e0: break        # rod length error, m (~1e-12 rad at the crank)
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
    def joint_angles(self, theta, guess=(0.0, 0.0), iters=40, tol=1e-12, max_step=0.2):
        """Joint angles for measured motor (crank) angles `theta`: Newton's method on motor_angles from `guess`, each
        step limited to `max_step` rad so a far guess cannot overshoot. Raises ValueError when it does not converge."""
        q = np.asarray(guess, float).copy(); theta = np.asarray(theta, float)
        for _ in range(iters):
            e = self.motor_angles(q)-theta
            if np.max(np.abs(e)) < tol: return q
            dq = np.linalg.solve(self.jacobian(q), e)
            q = q-dq*min(1.0, max_step/max(np.max(np.abs(dq)), 1e-300))
        if np.max(np.abs(self.motor_angles(q)-theta)) < 1e-9: return q
        raise ValueError(f'no joint angles for motor angles {theta.tolist()}')
    def motor_torque(self, q, tau_joint):
        """Motor torques that produce joint torque `tau_joint` at `q`."""
        return np.linalg.solve(self.jacobian(q).T, np.asarray(tau_joint, float))
    def motor_velocity(self, q, qd):
        """Motor speeds for joint speeds `qd` at `q`."""
        return self.jacobian(q)@np.asarray(qd, float)
    def joint_torque_limit(self, q, motor_torque_limit):
        """Largest pure torque on each joint when both motors are at +/- the given limit."""
        return np.abs(self.jacobian(q)).sum(0)*motor_torque_limit
if __name__ == '__main__':
    for n in ('left_ankle', 'right_ankle', 'waist'):
        m = Mechanism(n); J = m.jacobian((0, 0))
        print(n, m.joints, '\n  d(motor)/d(joint) at zero pose:\n ', np.round(J, 3).tolist())
