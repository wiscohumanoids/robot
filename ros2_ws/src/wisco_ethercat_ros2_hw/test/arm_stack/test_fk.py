"""Forward kinematics of arm01: URDF/Pinocchio against the CAD export, and
self-consistency of the FK, Jacobian and kinematic overlay.

The reference is robots/arm01/cad/arm01_capture.json, written by
robots/arm01/cad/import_full_assembly.py from the CAD team's `full_assembly_urdf`
export: its capture pose (all nine joints, the passive wrist ones at the
export's exact four-bar solution), every link's captured CAD placement in
this repo's base frame, the four-bar pins, and the operating ranges.
"""

import importlib.util
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pytest

from arm_stack.calib.common import safe_home
from arm_stack.config import load_arm
from arm_stack.model.robot import ArmModel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARM01 = os.path.normpath(os.path.join(ROOT, '..', '..', '..', 'robots', 'arm01'))
URDF = os.path.join(ARM01, 'urdf', 'arm01.urdf')
with open(os.path.join(ARM01, 'cad', 'arm01_capture.json')) as _f:
    CAP = json.load(_f)
CAPTURE = CAP['capture_q_rad']

POS_TOL = 1e-6                    # m: the export's numbers, re-based (1 um)
ANG_TOL = 1e-6                    # rad
LINK_FRAMES = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'flange', 'tool0']
DRIVEN_LINKS = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7']


def unit(v):
    """Unit vector."""
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def angle(u, v):
    """Angle between two directions (sign ignored)."""
    return math.acos(min(1.0, abs(float(unit(u) @ unit(v)))))


def line_distance(p, point, direction):
    """Distance from point `p` to the line through `point` along `direction`."""
    d = unit(direction)
    off = np.asarray(p, float) - point
    return float(np.linalg.norm(off - d * (off @ d)))


def placement_error(M, T):
    """Position and rotation error between a pin.SE3 and a 4x4 transform."""
    T = np.asarray(T, float)
    return (float(np.linalg.norm(M.translation - T[:3, 3])),
            float(np.linalg.norm(pin.log3(M.rotation.T @ T[:3, :3]))))


# --------------------------------------------------------------------------- models
@pytest.fixture(scope='module')
def full():
    """The whole URDF tree: the parallelogram's crank and rod are free joints here."""
    return pin.buildModelFromUrdf(URDF)


@pytest.fixture(scope='module')
def arm():
    """arm01 joint config."""
    return load_arm('arm01')


@pytest.fixture()
def model(arm):
    """arm01 model."""
    return ArmModel(arm.urdf, arm.names)


def full_q(m, **values):
    """Neutral configuration with the named joints set."""
    q = pin.neutral(m)
    for name, v in values.items():
        q[m.joints[m.getJointId(name)].idx_q] = v
    return q


def _mimics(path):
    """Mimic joints of a URDF: name -> (leader, multiplier, offset)."""
    out = {}
    for j in ET.parse(path).getroot().findall('joint'):
        m = j.find('mimic')
        if m is not None:
            out[j.get('name')] = (m.get('joint'), float(m.get('multiplier', 1)), float(m.get('offset', 0)))
    return out


MIMIC = _mimics(URDF)


def wrist(q6, **values):
    """Full-model joint values with the URDF's mimic joints following wrist_pitch."""
    out = dict(values, wrist_pitch=q6)
    for name, (leader, mult, offset) in MIMIC.items():
        assert leader == 'wrist_pitch', name
        out[name] = mult * q6 + offset
    return out


def test_parallelogram_mimics_are_the_crank_and_rod():
    assert {n: m[1:] for n, m in MIMIC.items()} == {'wrist_crank_joint': (1.0, 0.0),
                                                    'wrist_rod_joint': (-1.0, 0.0)}


def joint_axis_world(d, j):
    """World axis of joint `j`."""
    S = np.array(d.joints[j].S).reshape(6, -1)
    return d.oMi[j].rotation @ S[3:, 0]


def kinematics(m, q):
    """Pinocchio data after forward kinematics and frame placements at `q`."""
    d = m.createData()
    pin.forwardKinematics(m, d, q)
    pin.updateFramePlacements(m, d)
    return d


def random_poses(model, count, seed):
    """Random configurations inside the limits."""
    rng = np.random.default_rng(seed)
    return [rng.uniform(model.lower, model.upper) for _ in range(count)]



# =========================================================================== 1. FK vs the CAD export
@pytest.mark.parametrize('link', sorted(set(CAP['links_in_base']) - {'base_link'}))
def test_capture_pose_reproduces_the_export(full, link):
    """The whole tree at the CAD capture (the crank, coupler and output pivot at the
    export's exact loop solution) puts every link where the export captured it."""
    d = kinematics(full, full_q(full, **CAPTURE))
    dp, dr = placement_error(d.oMf[full.getFrameId(link)], CAP['links_in_base'][link])
    assert dp < POS_TOL and dr < ANG_TOL, (link, dp, dr)


def test_base_mesh_is_rebased_with_the_arm(full):
    """base_link is the post top (CAD (0, 0, 541) mm, x = CAD -y): its mesh carries the
    same transform the export's base frame gets."""
    vis = ET.parse(URDF).getroot().find("link[@name='base_link']/visual/origin")
    T = np.eye(4)
    T[:3, :3] = _rpy(*_floats(vis.get('rpy'), '0 0 0'))
    T[:3, 3] = _floats(vis.get('xyz'), '0 0 0')
    np.testing.assert_allclose(T, CAP['links_in_base']['base_link'], atol=1e-8)


def test_driven_model_matches_the_capture(model):
    """The reduced model the sim and controller use (mimics locked, wrist_pitch is
    the output pivot) at the capture: every driven link lands on the export."""
    q = np.array([CAPTURE[n] for n in model.joint_names])
    for link in DRIVEN_LINKS:
        dp, dr = placement_error(model.fk(q, link), CAP['links_in_base'][link])
        assert dp < POS_TOL and dr < ANG_TOL, (link, dp, dr)


def test_limits_are_the_export_operating_ranges(model):
    for k, name in enumerate(model.joint_names):
        lo, hi = CAP['operating_limits_deg'][name]
        assert model.lower[k] == pytest.approx(math.radians(lo), abs=1e-9), name
        assert model.upper[k] == pytest.approx(math.radians(hi), abs=1e-9), name


def test_home_is_low_gravity_and_zero_is_stretched_out(arm):
    """q = 0 is the arm held out forward (worst shoulder load); `home_pose_rad` hangs the
    upper arm with the elbow bent, the peg tip clear of the stand's base plate (z -0.541)."""
    m = ArmModel(arm.urdf, arm.names, home=arm.home)
    clamp = np.array([j.max_torque_nm for j in arm.joints])
    home = safe_home(m)
    assert np.all(np.abs(m.gravity(home)) < 0.3 * clamp), m.gravity(home)
    zero = np.clip(np.zeros(m.n), m.lower, m.upper)
    # Compared with home rather than the clamp: the clamps are now the motors' peak torque.
    assert abs(m.gravity(zero)[0]) > 2 * abs(m.gravity(home)[0])
    tip0, tiph = m.fk(zero).translation, m.fk(home).translation
    assert tip0[0] > 0.4 and abs(tip0[2]) < 0.2           # out in front, near shoulder height
    assert -0.49 < tiph[2] < -0.35 and tiph[0] < 0.35      # below the shoulder, peg clear of the base
    # the capture (the arm hanging in CAD) is within 15 deg of home on the shoulder
    assert abs(home[0] - CAPTURE['shoulder_pitch']) < math.radians(15)


# --------------------------------------------------------------------------- wrist four-bar
def four_bar():
    """Crank pivot C, crank pin A, output pivot P, output pin B (link5 frame, at the CAD
    capture: the export gives the pins where they sit in the capture), and the pin axis."""
    fb = CAP['four_bar_in_link5']
    return (np.array(fb['input_origin']), np.array(fb['input_pin']), np.array(fb['output_origin']),
            np.array(fb['output_pin']), unit(fb['axis']))


def capture_full(full):
    """Kinematics at the CAD capture pose."""
    return kinematics(full, full_q(full, **CAPTURE))


def in_link5(full, d, point):
    """Point expressed through link5."""
    return d.oMf[full.getFrameId('link5')].act(point)


def test_four_bar_pins_are_the_urdf_joints(full):
    """At the capture the crank pivot, crank pin and output pivot are the URDF's crank,
    coupler and wrist_pitch joints, all turning about the pin axis."""
    C, A, P, B, a = four_bar()
    d = capture_full(full)
    R5 = d.oMf[full.getFrameId('link5')].rotation
    for name, point in (('wrist_crank_joint', C), ('wrist_rod_joint', A), ('wrist_pitch', P)):
        j = full.getJointId(name)
        assert np.linalg.norm(d.oMi[j].translation - in_link5(full, d, point)) < POS_TOL, name
        assert angle(R5.T @ joint_axis_world(d, j), a) < ANG_TOL, name


def solve_output(phi):
    """Exact four-bar: the output pivot angle for crank angle phi, both in the URDF's
    coordinates (the capture pins sit at the captured crank and output angles)."""
    C, A, P, B, a = four_bar()
    phi_c, psi_c = CAPTURE['wrist_crank_joint'], CAPTURE['wrist_pitch']
    perp = lambda v: v - a * (v @ a)   # noqa: E731
    c0, l0, L = perp(A - C), perp(B - P), np.linalg.norm(perp(B - A))
    d = perp(C + pin.exp3(a * (phi - phi_c)) @ c0 - P)
    k = (l0 @ l0 + d @ d - L * L) / 2
    x, y = l0 @ d, np.cross(a, l0) @ d
    cands = [psi_c + math.atan2(y, x) + s * math.acos(k / math.hypot(x, y)) for s in (1, -1)]
    return min(cands, key=lambda c: abs(math.remainder(c - phi, 2 * math.pi)))


def test_exact_four_bar_closes_at_zero():
    """The pin geometry alone puts the output at 0 when the crank is at 0: the
    export's q = 0 closes the loop, which the 1:1 mimic relies on."""
    assert solve_output(0.0) == pytest.approx(0.0, abs=1e-6)


def test_one_to_one_mimic_is_within_a_quarter_degree(full):
    """Inside the wrist_pitch limits the output pivot follows the crank 1:1 within
    0.25 deg (export: ratio 0.996..1.004), so driving wrist_pitch with the J6 encoder
    is good to that."""
    j6 = full.getJointId('wrist_pitch')
    lo, hi = full.lowerPositionLimit[full.joints[j6].idx_q], full.upperPositionLimit[full.joints[j6].idx_q]
    worst = max(abs(math.remainder(solve_output(phi) - phi, 2 * math.pi)) for phi in np.linspace(lo, hi, 71))
    assert worst < math.radians(0.25), math.degrees(worst)
    assert worst > math.radians(0.005)         # really not a parallelogram: the check is not vacuous


def test_mimic_tree_keeps_the_loop_nearly_closed(full):
    """With the URDF mimics (crank +q6, coupler -q6) the coupler keeps link5's
    orientation and its far pin stays within 0.25 mm of the output link's pin."""
    _, _, _, B, _ = four_bar()
    dc = capture_full(full)
    jr, j6 = full.getJointId('wrist_rod_joint'), full.getJointId('wrist_pitch')
    Bw = in_link5(full, dc, B)
    B_rod, B_out = dc.oMi[jr].actInv(Bw), dc.oMi[j6].actInv(Bw)    # body-fixed pin, both sides
    d0 = kinematics(full, pin.neutral(full))
    rel0 = d0.oMf[full.getFrameId('link5')].rotation.T @ d0.oMi[jr].rotation
    lo, hi = full.lowerPositionLimit[full.joints[j6].idx_q], full.upperPositionLimit[full.joints[j6].idx_q]
    for q6 in np.linspace(lo, hi, 29):
        d = kinematics(full, full_q(full, **wrist(q6)))
        rel = d.oMf[full.getFrameId('link5')].rotation.T @ d.oMi[jr].rotation
        assert np.linalg.norm(pin.log3(rel0.T @ rel)) < 1e-9, 'coupler turned with the crank'
        gap = np.linalg.norm(d.oMi[jr].act(B_rod) - d.oMi[j6].act(B_out))
        assert gap < 0.25e-3, 'loop opens by %.3f mm at %.1f deg' % (gap * 1e3, math.degrees(q6))


# --------------------------------------------------------------------------- flange and tool0
def _importer():
    """Load robots/arm01/cad/import_full_assembly.py as a module."""
    spec = importlib.util.spec_from_file_location(
        'import_full_assembly', os.path.join(ARM01, 'cad', 'import_full_assembly.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flange_on_the_j7_axis_and_tool0_on_the_peg(model):
    q = np.array([CAPTURE[n] for n in model.joint_names])
    l7 = model.fk(q, 'link7')
    flange = l7.actInv(model.fk(q, 'flange'))
    tool0 = l7.actInv(model.fk(q, 'tool0'))
    # flange: on the J7 axis (link7 z), z along it toward the tool side
    assert np.linalg.norm(flange.translation[:2]) < 1e-9
    assert flange.rotation[2, 2] > math.cos(1e-9)
    # tool0: the peg sticks out perpendicular to J7, and z points out of its tip
    assert abs(tool0.rotation[2, 2]) < math.sin(math.radians(1.0))
    assert float(tool0.rotation[:, 2] @ tool0.translation) > 0.15
    # an independent re-fit of the peg's end face on the shipped mesh lands on tool0
    tip, axis, radius = _importer().peg_tip(os.path.join(ARM01, 'meshes', 'link7.stl'))
    assert radius == pytest.approx(0.010, abs=0.0005)
    assert np.linalg.norm(tip - tool0.translation) < 0.5e-3
    assert angle(axis, tool0.rotation[:, 2]) < math.radians(0.5)


def test_tool0_is_rigid_on_the_flange(model):
    ref = None
    for q in random_poses(model, 20, seed=3):
        T = model.fk(q, 'flange').actInv(model.fk(q, 'tool0'))
        if ref is None:
            ref = T
        assert np.allclose(T.homogeneous, ref.homogeneous, atol=1e-12)


# =========================================================================== 2. self-consistency
def overlay_model(arm, overlay):
    """Arm model with a kinematic overlay applied."""
    m = ArmModel(arm.urdf, arm.names)
    m.apply_overlay(overlay)
    return m


@pytest.mark.parametrize('frame', ['tool0', 'flange', 'link4'])
def test_jacobian_matches_finite_differences(model, arm, frame):
    calibrated = overlay_model(arm, {
        'shoulder_roll': {'dxyz': [0.001, -0.002, 0.0005], 'drot': [0.01, 0, -0.02]},
        'wrist_pitch': {'dxyz': [0.0, 0.001, 0.0], 'drot': [0.0, 0.015, 0.0]}})
    h = 1e-6
    for m in (model, calibrated):
        for q in random_poses(m, 10, seed=1):
            J = m.jacobian(q, frame)
            fd = np.zeros_like(J)
            for k in range(m.n):
                dq = np.zeros(m.n)
                dq[k] = h
                Mp, Mm = m.fk(q + dq, frame), m.fk(q - dq, frame)
                fd[:3, k] = (Mp.translation - Mm.translation) / (2 * h)
                fd[3:, k] = pin.log3(Mp.rotation @ Mm.rotation.T) / (2 * h)
            assert np.allclose(J, fd, atol=1e-7), np.abs(J - fd).max()


def subtree(m, root):
    """Joint ids in the subtree rooted at `root`."""
    out = {root}
    for j in range(root + 1, m.njoints):
        if m.parents[j] in out:
            out.add(j)
    return out


@pytest.mark.parametrize('name', ['shoulder_pitch', 'shoulder_yaw', 'elbow', 'wrist_pitch', 'wrist_yaw'])
def test_overlay_translation_moves_downstream_frames_by_that_amount(model, arm, name):
    dxyz = np.array([0.0012, -0.0007, 0.0021])
    shifted = overlay_model(arm, {name: {'dxyz': dxyz.tolist()}})
    k = model.joint_names.index(name)
    j = model.jid[k]
    down = subtree(model.model, j)
    frames = [f for f in model.model.frames]
    for q in [np.zeros(model.n)] + random_poses(model, 10, seed=2):
        a = kinematics(model.model, model._q(q))
        b = kinematics(shifted.model, shifted._q(q))
        # the shift is along dxyz in the parent joint's frame
        R_parent = a.oMi[model.model.parents[j]].rotation
        expect = R_parent @ dxyz
        for fid, f in enumerate(frames):
            Ma, Mb = a.oMf[fid], b.oMf[fid]
            assert np.allclose(Ma.rotation, Mb.rotation, atol=1e-12), f.name
            moved = Mb.translation - Ma.translation
            want = expect if f.parentJoint in down else np.zeros(3)
            assert np.allclose(moved, want, atol=1e-12), f.name


def test_overlay_rotation_pivots_on_the_joint_origin(model, arm):
    name = 'elbow'
    tilted = overlay_model(arm, {name: {'drot': [0.02, -0.01, 0.015]}})
    j = model.jid[model.joint_names.index(name)]
    down = subtree(model.model, j)
    for q in random_poses(model, 10, seed=4):
        a = kinematics(model.model, model._q(q))
        b = kinematics(tilted.model, tilted._q(q))
        o = a.oMi[j].translation
        assert np.allclose(b.oMi[j].translation, o, atol=1e-12)
        for fid, f in enumerate(model.model.frames):
            pa, pb = a.oMf[fid].translation, b.oMf[fid].translation
            if f.parentJoint in down:
                assert abs(np.linalg.norm(pb - o) - np.linalg.norm(pa - o)) < 1e-12, f.name
            else:
                assert np.allclose(pa, pb, atol=1e-12), f.name


# --------------------------------------------------------------------------- independent FK
def _floats(text, default):
    """Floats from a space-separated attribute."""
    return np.array([float(v) for v in (text or default).split()])


def _rpy(r, p, y):
    """URDF fixed-axis roll, pitch, yaw rotation matrix."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx                    # URDF: fixed-axis roll, pitch, yaw


def _hat(w):
    """Skew-symmetric matrix."""
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def _exp_twist(w, v, theta):
    """exp([S] theta) for a unit-rotation screw S = (w, v)."""
    K = _hat(w)
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K
    T[:3, 3] = (np.eye(3) * theta + (1 - math.cos(theta)) * K + (theta - math.sin(theta)) * K @ K) @ v
    return T


class ProductOfExponentials:
    """Space-frame PoE FK built straight from the URDF XML (no Pinocchio):
    T(q) = exp([S1] q1) ... exp([Sn] qn) M, with M the zero-pose link transform
    and S the joint screws in base_link at the zero pose."""

    def __init__(self, path):
        """Independent product-of-exponentials model read straight from the URDF."""
        root = ET.parse(path).getroot()
        self.joint = {}
        for j in root.findall('joint'):
            o = j.find('origin')
            ax = j.find('axis')
            mimic = j.find('mimic')
            T = np.eye(4)
            T[:3, :3] = _rpy(*_floats(o.get('rpy') if o is not None else None, '0 0 0'))
            T[:3, 3] = _floats(o.get('xyz') if o is not None else None, '0 0 0')
            self.joint[j.get('name')] = dict(
                type=j.get('type'), parent=j.find('parent').get('link'), child=j.find('child').get('link'),
                origin=T, axis=unit(_floats(ax.get('xyz') if ax is not None else None, '1 0 0')),
                mimic=None if mimic is None else (mimic.get('joint'), float(mimic.get('multiplier', 1)),
                                                  float(mimic.get('offset', 0))))
        self.by_child = {d['child']: n for n, d in self.joint.items()}
        self.M = {}
        self.screw = {}
        for link in [d['child'] for d in self.joint.values()]:
            self._zero(link)
        for n, d in self.joint.items():
            if d['type'] in ('revolute', 'continuous'):
                Mc = self.M[d['child']]
                w = Mc[:3, :3] @ d['axis']
                self.screw[n] = (w, -np.cross(w, Mc[:3, 3]))

    def _zero(self, link):
        """Zero-pose transform of `link`."""
        if link not in self.M:
            if link not in self.by_child:
                self.M[link] = np.eye(4)            # root
            else:
                d = self.joint[self.by_child[link]]
                self.M[link] = self._zero(d['parent']) @ d['origin']
        return self.M[link]

    def chain(self, link):
        """Joints from `link` back to the root."""
        out = []
        while link in self.by_child:
            n = self.by_child[link]
            out.append(n)
            link = self.joint[n]['parent']
        return out[::-1]

    def fk(self, q: dict, link):
        """Pose of `link` for joint values `q` (mimic joints follow their leaders)."""
        T = np.eye(4)
        for n in self.chain(link):
            if n not in self.screw:
                continue
            m = self.joint[n]['mimic']
            theta = m[1] * q[m[0]] + m[2] if m else q[n]
            T = T @ _exp_twist(*self.screw[n], theta)
        return T @ self.M[link]


def test_independent_poe_fk_matches_pinocchio(model, full):
    poe = ProductOfExponentials(URDF)
    for q in random_poses(model, 100, seed=5):
        qmap = dict(zip(model.joint_names, q))
        for frame in LINK_FRAMES:
            err = np.abs(model.fk(q, frame).homogeneous - poe.fk(qmap, frame)).max()
            assert err < 1e-9, (frame, err)
        # the parallelogram's moving parts, with the mimic joints following
        d = kinematics(full, full_q(full, **wrist(qmap['wrist_pitch'],
                                                   **{n: v for n, v in qmap.items() if n != 'wrist_pitch'})))
        for frame in ('wrist_crank', 'wrist_rod'):
            err = np.abs(d.oMf[full.getFrameId(frame)].homogeneous - poe.fk(qmap, frame)).max()
            assert err < 1e-9, (frame, err)


def test_overlay_applied_twice_does_not_compound():
    arm = load_arm()
    m = ArmModel(arm.urdf, arm.names)
    overlay = {arm.names[3]: {'dxyz': [0.001, -0.0005, 0.0007], 'drot': [0.002, 0.0, -0.001]}}
    q = np.linspace(-0.3, 0.4, m.n)
    m.apply_overlay(overlay)
    once = m.fk(q).homogeneous.copy()
    m.apply_overlay(overlay)
    np.testing.assert_allclose(m.fk(q).homogeneous, once, atol=1e-12)
    # and the nominal model is untouched
    plain = ArmModel(arm.urdf, arm.names)
    np.testing.assert_allclose(plain.fk(q).homogeneous,
                               ArmModel(arm.urdf, arm.names).fk(q).homogeneous, atol=1e-12)
    assert not np.allclose(plain.fk(q).homogeneous, once, atol=1e-6)
