"""Camera and footprint geometry against MuJoCo's own cameras and the sim's true block poses."""
import numpy as np
import pytest

from core.build import load_config
from core.camera import Camera, pose_error
from core.geometry import fold_yaw, footprint, intersect_plane
from robots.sim_robot import SimRobot


@pytest.fixture(scope='module')
def robot():
    return SimRobot(load_config(), seed=3)


def test_projection_hits_the_right_body(robot):
    """A block's top-centre projects onto pixels of that block in both cameras."""
    for cam in ('overhead', 'wrist'):
        obs = robot.observe(cam)
        names, seg = obs.extra['body_names'], obs.extra['segmentation']
        for name, (p, _) in robot.truth_objects().items():
            u, v = obs.camera.project((p + [0, 0, 0.0225])[None])[0]
            if 0 <= u < obs.camera.width and 0 <= v < obs.camera.height:
                assert names[seg[int(round(v)), int(round(u))]] == name, (cam, name)


def test_ray_plane_round_trip(robot):
    """project, then the ray through that pixel, lands back on the point (pinhole and with distortion)."""
    cam = robot.camera('overhead')
    pts = np.array([[0.3, 0.1, 0.095], [0.5, -0.2, 0.095], [0.4, 0.0, 0.095]])
    back = intersect_plane(cam.position, cam.rays(cam.project(pts)), 0.095)
    assert np.allclose(back, pts, atol=1e-6)
    cam.dist = np.array([-0.2, 0.05, 0.001, -0.001, 0.0])
    back = intersect_plane(cam.position, cam.rays(cam.project(pts)), 0.095)
    assert np.allclose(back, pts, atol=1e-4)


def test_footprint_from_the_overhead_camera(robot):
    """Unoccluded blocks: centre within 1 mm, size 45 mm, yaw modulo 90 deg within 1 deg."""
    obs = robot.observe('overhead')
    names, seg = obs.extra['body_names'], obs.extra['segmentation']
    for name, (p, R) in robot.truth_objects().items():
        fp = footprint(seg == names.index(name), obs.camera, 0.095, 0.05)
        assert np.linalg.norm(fp.center[:2] - p[:2]) < 1e-3
        assert np.allclose(fp.size, 0.045, atol=1.5e-3)
        dyaw = (np.rad2deg(fp.yaw - np.arctan2(R[1, 0], R[0, 0])) + 45) % 90 - 45
        assert abs(dyaw) < 1.0


def test_yaw_folding():
    assert fold_yaw(np.deg2rad(100), np.deg2rad(90), 0.0) == pytest.approx(np.deg2rad(10))
    assert fold_yaw(np.deg2rad(-80), np.deg2rad(90), np.deg2rad(45)) == pytest.approx(np.deg2rad(10))


def test_camera_dict_round_trip_and_pose_error(robot):
    cam = robot.camera('wrist')
    again = Camera.from_dict(cam.to_dict())
    assert np.allclose(again.K, cam.K) and np.allclose(again.T_base_cam, cam.T_base_cam)
    E = pose_error(0.005, 1.0, np.random.default_rng(0))
    assert np.linalg.norm(E[:3, 3]) == pytest.approx(0.005)
    angle = np.rad2deg(np.arccos((np.trace(E[:3, :3]) - 1) / 2))
    assert angle == pytest.approx(1.0, abs=1e-6)
