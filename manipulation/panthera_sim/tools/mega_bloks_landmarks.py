"""Project visible physical cuboid corners and edge fragments into rendered images."""
from __future__ import annotations

from itertools import product
import numpy as np
import mujoco


def camera_basis(scene):
    cameras = scene.camera
    position = (np.array(cameras[0].pos, dtype=float) + cameras[1].pos) / 2
    forward = np.array(cameras[0].forward, dtype=float)
    forward /= np.linalg.norm(forward)
    up = np.array(cameras[0].up, dtype=float)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    return position, right, up, forward


def project_points(points, scene, width, height):
    """Project world points through MuJoCo's mono center GL camera.

    Return zero-based pixel-index XY coordinates and positive camera-axis depth.
    The midpoint of the two GL eye positions matches mono rendering.
    """
    camera = scene.camera[0]
    position, right, up, forward = camera_basis(scene)
    relative = np.asarray(points) - position
    z = relative @ forward
    near = float(camera.frustum_near)
    top, bottom = float(camera.frustum_top), float(camera.frustum_bottom)
    half_width = (top - bottom) * width / height / 2
    center_x = float(camera.frustum_center)
    if camera.orthographic:
        projected_x, projected_y = relative @ right, relative @ up
    else:
        projected_x = (relative @ right) * near / z
        projected_y = (relative @ up) * near / z
    x = (projected_x - center_x) / (2 * half_width) * width + (width - 1) / 2
    y = (top - projected_y) / (top - bottom) * height - .5
    return np.stack([x, y], axis=-1), z


def visible_samples(xy, z, body_geom, segmentation, depth, scene):
    """Depth-test samples against their own body within a one-pixel boundary band."""
    height, width = depth.shape
    nearest = np.rint(xy).astype(int)
    best = np.full(len(xy), np.inf)
    # One rendered pixel subtends this much world distance at the query depth.
    camera = scene.camera[0]
    footprint = z * (camera.frustum_top - camera.frustum_bottom) / camera.frustum_near / height
    tolerance = np.maximum(.0006, 1.5 * footprint)
    inside = (z > 0) & (xy[:, 0] >= -.5) & (xy[:, 0] < width - .5) & (xy[:, 1] >= -.5) & (xy[:, 1] < height - .5)
    for dy, dx in product((-1, 0, 1), repeat=2):
        x = np.clip(nearest[:, 0] + dx, 0, width - 1)
        y = np.clip(nearest[:, 1] + dy, 0, height - 1)
        owner = (segmentation[y, x, 0] == body_geom) & (segmentation[y, x, 1] == mujoco.mjtObj.mjOBJ_GEOM)
        difference = np.abs(depth[y, x] - z)
        best = np.minimum(best, np.where(owner, difference, np.inf))
    return inside & (best <= tolerance), best


def body_landmarks(model, data, scene, segmentation, depth, blocks, width, height):
    """Return visible body corners and dense visible fragments of its twelve edges.

    ``blocks`` contains the scene's block metadata for active instances, with an
    ``instance_id`` field. Fragments stop where a different object, the body
    itself, or a stud occludes the edge. Backside surfaces fail the depth test.
    """
    result = {'corners': [], 'corner_details': [], 'edge_details': []}
    position, _, _, _ = camera_basis(scene)
    signs = np.array(list(product((-1, 1), repeat=3)), dtype=float)
    for block in blocks:
        instance_id = block['instance_id']
        geom = model.geom(block['body_geom']).id
        rotation = data.geom_xmat[geom].reshape(3, 3)
        center = data.geom_xpos[geom]
        world = (signs * model.geom_size[geom]) @ rotation.T + center
        xy, z = project_points(world, scene, width, height)
        visible, residual = visible_samples(xy, z, geom, segmentation, depth, scene)
        camera_local = (position - center) @ rotation
        visible &= np.any(signs * camera_local > model.geom_size[geom], axis=1)
        for index in np.flatnonzero(visible):
            point = xy[index].tolist()
            result['corners'].append(point)
            result['corner_details'].append({'instance_id': instance_id, 'corner_index': int(index),
                                             'xy': point, 'xyz_m': world[index].tolist(),
                                             'depth_residual_m': float(residual[index])})
        for a in range(8):
            for b in range(a + 1, 8):
                changed = np.flatnonzero(signs[a] != signs[b])
                if len(changed) != 1:
                    continue
                axis = int(changed[0])
                adjacent_axes = [i for i in range(3) if i != axis]
                # An edge with both adjacent faces facing away is hidden by its own solid.
                midpoint = (world[a] + world[b]) / 2
                if not any(np.dot(rotation[:, i] * signs[a, i], position - midpoint) > 0 for i in adjacent_axes):
                    continue
                length = np.linalg.norm(xy[a] - xy[b])
                count = min(4096, max(16, int(np.ceil(length * 2)) + 1))
                parameter = np.linspace(0, 1, count)
                samples = world[a] + parameter[:, None] * (world[b] - world[a])
                projected, sample_z = project_points(samples, scene, width, height)
                valid, _ = visible_samples(projected, sample_z, geom, segmentation, depth, scene)
                padded = np.r_[False, valid, False].astype(int)
                starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
                fragments = [projected[[start, end - 1]].tolist() for start, end in zip(starts, ends) if end - start >= 2]
                if fragments:
                    edge_type = 'vertical' if axis == 2 else ('top' if signs[a, 2] > 0 else 'bottom')
                    result['edge_details'].append({'instance_id': instance_id, 'corner_indices': [a, b],
                                                   'type': edge_type, 'fragments': fragments})
    return result
