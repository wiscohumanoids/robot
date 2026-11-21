"""Projection and occlusion contracts for cuboid landmark labels."""
import importlib.util
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import numpy as np
import pytest

TOOLS = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location('landmarks_replay', TOOLS / 'mega_bloks_landmarks.py')
landmarks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(landmarks)


@pytest.mark.parametrize('camera_kind', ['fixed', 'free'])
def test_projected_centers_match_rendered_pixel_centers(camera_kind):
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <visual><global offwidth="320" offheight="240"/></visual>
      <worldbody><camera name="known" pos="0 0 3" fovy="45"/>
        <geom name="origin" type="sphere" pos="0 0 0" size=".035"/>
        <geom name="right" type="sphere" pos=".5 0 0" size=".035"/>
        <geom name="up" type="sphere" pos="0 .4 0" size=".035"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = 'known'
    if camera_kind == 'free':
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0, 0, 0]
        camera.distance = 3
        camera.azimuth = 77
        camera.elevation = -65
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera)
        segmentation = renderer.render().copy()
        points = data.geom_xpos.copy()
        xy, depth = landmarks.project_points(points, renderer.scene, 320, 240)
        assert np.all(depth > 0)
        for geom_id in range(model.ngeom):
            y, x = np.nonzero((segmentation[..., 0] == geom_id) &
                              (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM))
            assert x.size > 5
            np.testing.assert_allclose(xy[geom_id], [x.mean(), y.mean()], atol=.4)
        if camera_kind == 'fixed':
            np.testing.assert_allclose(xy[0], [159.5, 119.5], atol=1e-6)
            np.testing.assert_allclose(depth, 3., atol=1e-6)
            focal = 120 / np.tan(np.deg2rad(45) / 2)
            np.testing.assert_allclose(xy[1], [159.5 + focal * .5 / 3, 119.5], atol=1e-5)
            np.testing.assert_allclose(xy[2], [159.5, 119.5 - focal * .4 / 3], atol=1e-5)
            assert xy[1, 0] > xy[0, 0]
            assert xy[2, 1] < xy[0, 1]


@pytest.mark.parametrize('occluded', [False, True])
def test_body_hides_back_corners_and_foreground_occluder_hides_left_corners(occluded):
    blocker = ('<geom name="occluder" type="box" pos="-.25 0 .6" '
               'size=".24 .6 .1"/>' if occluded else '')
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <visual><global offwidth="320" offheight="240"/></visual>
      <worldbody><camera name="known" pos="0 0 3" fovy="45"/>
        <geom name="block" type="box" size=".4 .3 .2"/>{blocker}
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera='known')
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera='known')
        depth = renderer.render().copy()
        result = landmarks.body_landmarks(model, data, renderer.scene, segmentation,
                                          depth, [{'instance_id': 1, 'body_geom': 'block'}], 320, 240)
    indices = {corner['corner_index'] for corner in result['corner_details']}
    assert indices == ({5, 7} if occluded else {1, 3, 5, 7})
    assert all(edge['type'] == 'top' for edge in result['edge_details'])
    if occluded:
        # The blocker covers the middle of the top surface's left edge.
        assert not any(edge['corner_indices'] == [1, 3] for edge in result['edge_details'])


def test_edge_fragments_split_at_occlusion_without_losing_visible_endpoints():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <visual><global offwidth="320" offheight="240"/></visual>
      <worldbody><camera name="known" pos="0 0 3" fovy="45"/>
        <geom name="block" type="box" size=".4 .3 .2"/>
        <geom name="occluder" type="box" pos="0 0 .6" size=".05 .6 .1"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera='known')
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera='known')
        depth = renderer.render().copy()
        result = landmarks.body_landmarks(model, data, renderer.scene, segmentation,
                                          depth, [{'instance_id': 1, 'body_geom': 'block'}], 320, 240)
    assert {corner['corner_index'] for corner in result['corner_details']} == {1, 3, 5, 7}
    edges = {tuple(edge['corner_indices']): edge['fragments'] for edge in result['edge_details']}
    for key in ((1, 5), (3, 7)):
        assert len(edges[key]) == 2
        assert max(point[0] for point in edges[key][0]) < 159.5
        assert min(point[0] for point in edges[key][1]) > 159.5
