"""Pixel coordinate and connectivity contracts for inference measurements."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    'instances', Path(__file__).parents[1] / 'perception/instances.py')
instances = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(instances)


def test_separate_blocks_and_multiple_studs():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[2:10, 3:15] = 1
    mask[3:5, 4:6] = 2
    mask[6:8, 10:13] = 2
    mask[12:18, 20:27] = 1
    mask[13:15, 22:24] = 2
    objects = instances.extract_instances(mask)
    assert len(objects) == 2
    assert objects[0]['bbox_xyxy'] == [3, 2, 15, 10]
    assert objects[0]['area_pixels'] == 96
    assert [s['centroid_xy'] for s in objects[0]['studs']] == [[4.5, 3.5], [11., 6.5]]
    assert objects[1]['studs'][0]['centroid_xy'] == [22.5, 13.5]
    annotated = instances.annotate_instances(Image.new('RGB', (30, 20)), objects)
    assert annotated.size == (30, 20)


def test_empty_border_and_diagonal_connectivity():
    mask = np.zeros((4, 4), dtype=np.uint8)
    assert instances.extract_instances(mask) == []
    mask[0, 0] = 1
    mask[1, 1] = 2
    result = instances.extract_instances(mask)
    assert len(result) == 1
    assert result[0]['bbox_xyxy'] == [0, 0, 2, 2]
    assert result[0]['studs'][0]['centroid_xy'] == [1., 1.]


@pytest.mark.parametrize('mask', [np.zeros((2, 2, 3)), np.array([[3]])])
def test_invalid_labels(mask):
    with pytest.raises(ValueError):
        instances.extract_instances(mask)


def test_rgb_separates_touching_colors_and_keeps_unsaturated_pixels():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[20:80, 10:110] = 1
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[20:80, 10:60] = [220, 10, 5]
    rgb[20:80, 60:110] = [5, 40, 220]
    rgb[35:40, 30:35] = 240  # Specular pixels inherit a neighboring color seed.
    assert len(instances.extract_instances(mask)) == 1
    objects = instances.extract_instances(mask, rgb)
    assert len(objects) == 2
    assert sorted(obj['bbox_xyxy'] for obj in objects) == [[10, 20, 60, 80], [60, 20, 110, 80]]
    labels = instances.instance_regions(mask, rgb)
    assert np.array_equal(labels > 0, mask > 0)
    assert sum(obj['area_pixels'] for obj in objects) == int((mask > 0).sum())


def test_rgb_same_color_touching_remains_one_region():
    mask = np.ones((32, 32), dtype=np.uint8)
    rgb = np.full((32, 32, 3), [250, 70, 5], dtype=np.uint8)
    assert len(instances.extract_instances(mask, rgb)) == 1
