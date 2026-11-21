"""Check box identity and stud localization metrics on known detector outputs."""

from types import SimpleNamespace

import numpy as np
import pytest

from perception import evaluate_camera_robustness as benchmark


@pytest.mark.parametrize(('color', 'color_f1'), [('red', 1.), ('blue', 0.)])
def test_stud_credit_requires_a_matching_block_color(monkeypatch, color, color_f1):
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[2:10, 2:10] = 1
    mask[4:6, 4:6] = 2
    center = (4.5, 4.5)
    box = (2, 2, 10, 10)
    probabilities = np.eye(3)[mask].transpose(2, 0, 1)
    detector = SimpleNamespace(_predict_frame=lambda rgb, include_landmarks: (probabilities, [], [center]))
    detection = SimpleNamespace(bbox_xyxy=box, color=color, studs_xy=[center])
    monkeypatch.setattr(benchmark, 'detect_blocks', lambda *args, **kwargs: [detection])
    counts = benchmark.score_image(detector, np.zeros((12, 12, 3), dtype=np.uint8), mask,
                                   [('red', box, [center])], {'stud_centers': [center], 'corners': []})
    metrics = benchmark.totals([{'counts': counts}])
    assert metrics['foreground_miou'] == 1.
    assert metrics['blocks_f1'] == metrics['raw_studs_f1'] == 1.
    assert metrics['color_blocks_f1'] == metrics['studs_on_color_blocks_f1'] == color_f1


def test_empty_frame_counts_unmatched_predictions(monkeypatch):
    mask = np.zeros((12, 12), dtype=np.uint8)
    probabilities = np.eye(3)[mask].transpose(2, 0, 1)
    detector = SimpleNamespace(_predict_frame=lambda rgb, include_landmarks: (probabilities, [], []))
    detection = SimpleNamespace(bbox_xyxy=(0, 0, 4, 4), color='red', studs_xy=[(2., 2.)])
    monkeypatch.setattr(benchmark, 'detect_blocks', lambda *args, **kwargs: [detection])
    counts = benchmark.score_image(detector, np.zeros((12, 12, 3), dtype=np.uint8), mask,
                                   [], {'stud_centers': [], 'corners': []})
    assert counts['blocks'] == counts['color_blocks'] == counts['studs_on_color_blocks'] == [0, 1, 0]
