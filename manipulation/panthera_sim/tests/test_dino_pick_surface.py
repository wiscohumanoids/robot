import cv2
import numpy as np

from tools.dino_pick_camera import CameraCalibration
from tools.dino_pick_surface import SurfaceTracker, find_platforms


def _camera(position):
    return CameraCalibration(np.array(position), np.array([1., 0., 0.]),
                             np.array([0., 1., 0.]), np.array([0., 0., -1.]),
                             780., np.array([399.5, 299.5]))


def _natural_texture(seed, light):
    random = np.random.default_rng(seed)
    coarse = cv2.resize(random.normal(size=(16, 22)), (150, 190), interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(random.normal(size=(35, 40)), (150, 190), interpolation=cv2.INTER_CUBIC)
    return np.clip(light + 14 * coarse + 4 * fine, 56, 215).astype(np.uint8)


def _frame(camera, heights, visible=(0, 1), textured=True):
    rgb = np.full((600, 800, 3), (125, 102, 51), dtype=np.uint8)
    pattern_corners = np.array([[0, 0], [149, 0], [149, 189], [0, 189]], np.float32)
    for platform in visible:
        offsets = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * [.075, .095]
        corners = offsets + [.18 * platform - .09, 0.]
        world = [np.r_[corner, heights[platform] + .025 * corner[1]] for corner in corners]
        pixels = np.array([camera.project(corner) for corner in world], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(pattern_corners, pixels)
        light = (77, 145)[platform]
        texture = (_natural_texture(platform, light) if textured
                   else np.full((190, 150), light, np.uint8))
        warped = cv2.warpPerspective(texture, transform, (800, 600))
        rgb[warped > 0] = warped[warped > 0, None]
    return rgb


def _describe(crops):
    """Provide deterministic spatial descriptors for the geometry unit tests."""
    gray = crops.mean(axis=-1) / 255. - 104 / 255.
    tiles = gray.reshape(-1, 6, 16, 6, 16).transpose(0, 1, 3, 2, 4)
    detail = tiles[..., ::4, ::4].reshape(-1, 6, 6, 16)
    features = np.concatenate([detail, np.full((*detail.shape[:-1], 1), .2)], axis=-1)
    return features / np.linalg.norm(features, axis=-1, keepdims=True)


def test_click_intersects_the_selected_surface_from_rgb_views():
    heights = [.071, .136]
    overhead = _camera([0., 0., .61])
    rgb = _frame(overhead, heights)
    platforms = find_platforms(rgb)
    assert len(platforms) == 2
    for platform, height in enumerate(heights):
        xy = np.array([.18 * platform - .081, .013])
        location = np.r_[xy, height + .025 * xy[1]]
        pixel = overhead.project(location)
        tracker = SurfaceTracker(rgb, overhead, pixel, describe=_describe)
        assert platforms[tracker.platform_id].contains(pixel, margin=25.)
        assert tracker.estimate() is None
        for pose in ([.06, -.05, .48], [-.05, .06, .52], [.01, .075, .50]):
            camera = _camera(pose)
            tracker.add_frame(_frame(camera, heights), camera)
        estimate = tracker.estimate()
        assert tracker.view_count == 4
        assert estimate.view_count >= 3
        assert estimate.feature_count > 10
        assert estimate.reprojection_rmse_px < 1.
        assert estimate.plane_rmse_m < .003
        np.testing.assert_allclose(estimate.point, location, atol=.002)
        np.testing.assert_allclose(estimate.normal, [0., -.025, 1.], atol=.035)


def test_duplicate_views_and_occluded_surfaces_cannot_confirm_depth():
    heights = [.071, .136]
    overhead = _camera([0., 0., .61])
    rgb = _frame(overhead, heights, textured=False)
    tracker = SurfaceTracker(rgb, overhead, find_platforms(rgb)[0].center, describe=_describe)
    assert tracker.add_frame(rgb, overhead) == 0
    for pose in ([.06, -.05, .48], [-.05, .06, .52]):
        camera = _camera(pose)
        assert tracker.add_frame(_frame(camera, heights, visible=()), camera) == 1
    assert tracker.estimate() is None

    outside = SurfaceTracker(rgb, overhead, (0, 0), describe=_describe)
    assert outside.platform_id is None
    assert outside.add_frame(rgb, overhead) == 0
    assert outside.estimate() is None


def test_overlapping_supports_with_similar_brightness_have_separate_centers():
    rgb = np.full((400, 400, 3), (125, 102, 51), dtype=np.uint8)
    rear = np.array([[84, 100], [190, 100], [177, 157], [44, 157]])
    front = np.array([[137, 153], [274, 153], [279, 253], [103, 253]])
    cv2.fillConvexPoly(rgb, rear, (78, 78, 78))
    cv2.fillConvexPoly(rgb, front, (81, 81, 81))
    cv2.circle(rgb, (165, 166), 24, (180, 0, 0), -1)
    platforms = find_platforms(rgb)
    assert len(platforms) == 2
    assert platforms[0].contains((125, 125), margin=10)
    assert platforms[1].contains((190, 215), margin=10)
