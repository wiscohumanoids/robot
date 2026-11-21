"""Check that motion can continue while DINO works on captured stack views."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident

import numpy as np
import pytest

from tools.dino_pick_stack import _StackObservations


def test_detection_uses_snapshots_and_delivers_on_control_thread():
    control_thread = get_ident()
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    pose = np.zeros(1)
    tick = 0
    delivered = []
    release_detector = Event()

    def see(name):
        assert get_ident() == control_thread
        index = ("overhead", "left_wrist", "right_wrist").index(name)
        pose[:] = tick + index
        rgb[:] = tick + index
        return rgb, pose

    class Detector:
        def detect(self, image):
            assert get_ident() != control_thread
            assert release_detector.wait(timeout=5)
            return (int(image[0, 0, 0]),)

    def apply(phase, held, captured_at, overhead, wrists):
        assert get_ident() == control_thread
        frames = (overhead, *wrists)
        delivered.append((phase, held, captured_at, [f.detections[0] for f in frames],
                          [int(f.camera[0]) for f in frames]))

    with ThreadPoolExecutor(max_workers=1) as worker:
        observations = _StackObservations(see, Detector(), worker, apply, lambda: tick)
        try:
            observations.capture("survey")
            tick = 10
            observations.capture("lift", held=True)
            pose[:] = 99
            rgb[:] = 99
            observations.poll()
            assert delivered == []
        finally:
            release_detector.set()
        observations.finish()

    assert delivered == [("survey", False, 0, [0, 1, 2], [0, 1, 2]),
                         ("lift", True, 10, [10, 11, 12], [10, 11, 12])]


def test_worker_error_reaches_decision():
    class FailedDetector:
        def detect(self, image):
            raise RuntimeError("detector failed")

    with ThreadPoolExecutor(max_workers=1) as worker:
        observations = _StackObservations(lambda _: (np.zeros((1, 1, 3)), None),
                                          FailedDetector(), worker, None, lambda: 0.)
        observations.capture("survey")
        with pytest.raises(RuntimeError, match="detector failed"):
            observations.finish()
        observations.cancel()


def test_late_detections_do_not_pause_a_grasp_or_change_its_source(monkeypatch):
    from concurrent.futures import Future
    from types import SimpleNamespace

    from tools import dino_pick_stack as policy

    def estimate(x):
        xyz = np.array([x, .1, .13])
        return SimpleNamespace(xyz=xyz, studs_xyz=np.array([xyz, xyz + [.032, 0., 0.]]))

    tracker = SimpleNamespace(latest=SimpleNamespace(source=None, target=None))
    first_source, later_source = estimate(.20), estimate(.202)
    first_base, later_base = estimate(.30), estimate(.303)
    futures = []

    class FrameFuture(Future):
        def result(self, timeout=None):
            assert self.done(), "the moving arm waited for an optional camera frame"
            return super().result(timeout)

    class Executor:
        def submit(self, fn, views):
            futures.append(FrameFuture())
            return futures[-1]

    def deliver(phase, held, time, overhead, wrists):
        tracker.latest.source = first_source if phase == "survey_0" else later_source
        tracker.latest.target = first_base if phase == "survey_0" else later_base

    observations = _StackObservations(lambda _: (np.zeros((1, 1, 3)), None),
                                      None, Executor(), deliver, lambda: 0.)

    def survey(motion, capture):
        capture("survey_0")
        capture("survey_1")
        futures[0].set_result([None, None, None])
        return []

    monkeypatch.setattr(policy, "survey", survey)

    class Motion:
        def choose_stack_yaw(self, *args):
            return 0.

        def pick(self, center, yaw, *, observe_approach, refine):
            observe_approach()
            grasp, _ = refine(center, yaw)
            np.testing.assert_array_equal(grasp, policy._center(first_source))
            # Both detections arrive after the arm has committed to this grasp.
            for future in futures[1:]:
                future.set_result([None, None, None])
            return True

        def stack(self, source, studs, base, base_studs, *, observe, refine):
            np.testing.assert_array_equal(source, policy._center(first_source))
            np.testing.assert_array_equal(studs, first_source.studs_xyz)
            np.testing.assert_array_equal(base, policy._center(later_base))
            observe("transfer_mid")
            plan = SimpleNamespace(release=np.array([.3, .1, .2]))
            np.testing.assert_allclose(refine(plan), plan.release)
            return True

    record = {}
    policy._pick_and_stack(Motion(), observations, tracker, record, lambda: None)
    np.testing.assert_array_equal(record["source_xy_m"], first_source.xyz[:2])
    observations.cancel()
    assert futures[-1].cancelled()


def test_released_support_must_match_both_target_studs():
    from tools.dino_grid_build import _aligned

    target = np.array([[.20, .15, .14], [.232, .15, .14]])
    assert _aligned(target[::-1] + [.002, .001, -.003], target)
    assert not _aligned(target + [0., 0., .016], target)
    across = target.mean(axis=0) + np.array([[0., -.016, 0.], [0., .016, 0.]])
    assert not _aligned(across, target)
