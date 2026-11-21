"""Exercise physical camera calibration against known 3-D points without arm commands."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('cv2')
pytest.importorskip('mujoco')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'panthera_teleop/app'))
import stack_vision
from perception.pickup import BlockDetection


def camera_config(x):
    return {
        'resolution': [640, 480],
        'camera_matrix': [[575, 0, 328], [0, 529, 243], [0, 0, 1]],
        'distortion': [.12, -.04, .005, -.004, 0],
        'world_from_camera': [[1, 0, 0, x], [0, -1, 0, 0],
                              [0, 0, -1, .62], [0, 0, 0, 1]],
        'reference_q_rad': [0] * 6,
    }


STUDS = np.array([[[-.075, -.035, .1], [-.075, -.003, .1]],
                  [[.055, .018, .1], [.055, .05, .1]]])


def project_blocks(config, *, extra_peak=False):
    cv2 = stack_vision.cv2
    camera_from_world = np.linalg.inv(config['world_from_camera'])
    rotation = cv2.Rodrigues(camera_from_world[:3, :3])[0]
    detections = []
    for studs, color in zip(STUDS, ['blue', 'red']):
        pixels = cv2.projectPoints(studs, rotation, camera_from_world[:3, 3],
                                   np.array(config['camera_matrix'], float),
                                   np.array(config['distortion']))[0][:, 0]
        box = (*np.floor(pixels.min(axis=0) - 6), *np.ceil(pixels.max(axis=0) + 6))
        if extra_peak:
            pixels = np.vstack([pixels, pixels[0] + [6., -11.]])
        detections.append(BlockDetection(tuple(pixels.mean(axis=0)), color, 1., box,
                                         tuple(map(tuple, pixels)), 800))
    return detections


def test_calibrated_real_camera_conventions_and_wrist_reference(monkeypatch):
    overhead, wrist = camera_config(-.13), camera_config(.11)
    views = iter([project_blocks(overhead), project_blocks(wrist, extra_peak=True)] * 2)
    mock_detector = SimpleNamespace(detect=lambda rgb: next(views))
    monkeypatch.setattr(stack_vision, 'DinoPickupPerception', lambda *a, **kw: mock_detector)
    vision = stack_vision.StackVision(
        dict(overhead=overhead, wrist=wrist, checkpoint='test', table_z_m=.05075)
    )
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match='survey pose'):
        vision.locate(rgb, rgb, [.02] * 6)
    with pytest.raises(ValueError, match='640×480'):
        vision.locate(rgb[:256], rgb, [0] * 6)
    blocks = vision.locate(rgb, rgb, [0] * 6)
    np.testing.assert_allclose([b['studs_xyz'] for b in blocks], STUDS, atol=1e-6)
    assert blocks[0]['bbox_xyxy'] == list(project_blocks(overhead)[0].bbox_xyxy)
    vision.table_z += .03
    with pytest.raises(ValueError, match='calibrated table'):
        vision.locate(rgb, rgb, [0] * 6)


def test_auto_session_requires_independent_verification_and_updates_wrist_from_feedback(monkeypatch):
    import base64
    from stack_detector_worker import process_frame

    top, wrist = camera_config(-.13), camera_config(.11)
    cv2 = stack_vision.cv2
    image = np.zeros((480, 640, 3), np.uint8)
    image[:] = [255, 0, 0]  # Blue in OpenCV's BGR order.
    _, encoded = cv2.imencode('.jpg', image)
    jpeg = base64.b64encode(encoded.tobytes()).decode()
    packet = dict(overhead=jpeg, wrist=jpeg, q=[0.] * 6, gripper=.35)
    views = iter([project_blocks(top), project_blocks(wrist)])
    detector = SimpleNamespace(detect=lambda rgb: next(views))
    monkeypatch.setattr(stack_vision, 'DinoPickupPerception', lambda *a, **kw: detector)

    class Estimator:
        def __init__(self, config, supplied_detector):
            assert supplied_detector is detector
            self.passed = False
            self.queries = []
            self.closed = False

        def sample(self, overhead_rgb, wrist_rgb, q, opening):
            np.testing.assert_array_equal(q, packet['q'])
            assert opening == .35
            assert overhead_rgb.shape == wrist_rgb.shape == (480, 640, 3)
            assert overhead_rgb[0, 0, 2] >= 250 and overhead_rgb[0, 0, 0] <= 2
            return dict(ready=True, samples=2, message='Fit available')

        def verify(self, overhead_rgb, wrist_rgb, q, opening):
            return dict(ready=self.passed, samples=2, message='Validation')

        def camera_configs(self, q):
            self.queries.append(q.tolist())
            return {'overhead': top, 'wrist': wrist}

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, 'stack_autocal', SimpleNamespace(StackAutoCalibration=Estimator))
    vision = stack_vision.StackVision(dict(auto_calibrate=True, checkpoint='test', table_z_m=.05075))
    assert 'cameras' not in process_frame(vision, dict(packet, op='calibrate'))
    for kwargs in ({}, {'op': 'locate'}):
        with pytest.raises(ValueError, match='verify'):
            process_frame(vision, dict(packet, **kwargs))
    failed = process_frame(vision, dict(packet, op='finish'))
    assert not failed['ready'] and 'cameras' not in failed
    assert not vision.calibrator.queries
    vision.calibrator.passed = True
    report = process_frame(vision, dict(packet, op='finish'))
    assert report['cameras']['overhead'] == top
    measured = [.1] * 6
    blocks = process_frame(vision, dict(packet, q=measured))
    np.testing.assert_allclose([b['studs_xyz'] for b in blocks], STUDS, atol=1e-6)
    assert vision.calibrator.queries == [packet['q'], measured]
    process_frame(vision, dict(packet, op='calibrate'))
    with pytest.raises(ValueError, match='verify'):
        process_frame(vision, packet)
    with pytest.raises(ValueError, match='Unknown vision operation'):
        process_frame(vision, {'op': 'run'})
    vision.close()
    assert vision.calibrator.closed
