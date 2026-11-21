"""Locate known Panthera CAD features in an RGB frame using measured joint angles."""

from dataclasses import replace
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from sim.mega_bloks import BASE_SCENE


def _robot_model():
    root = ET.parse(BASE_SCENE).getroot()
    root.find('compiler').set('meshdir', str(BASE_SCENE.parent))
    for asset in root.findall('asset/*'):
        if asset.tag in ('material', 'texture'):
            root.find('asset').remove(asset)
    world = root.find('worldbody')
    for element in list(world):
        if element.tag != 'light' and element.get('name') not in ('right_base', 'left_base'):
            world.remove(element)
    for parent in root.iter('body'):
        for camera in parent.findall('camera'):
            parent.remove(camera)
    ET.SubElement(world, 'camera', name='cad_view')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def _gradients(gray):
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return np.dstack((np.sqrt(abs(dx)), np.sqrt(abs(dy))))


def _correlation(image, template, mask):
    scores = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED, mask=mask)
    # Masked normalization can have a zero denominator on blank image regions.
    scores[~np.isfinite(scores) | (scores > 1.001)] = -1
    return scores


def _match_corner(src_edges, dst_edges, component, u, v, shift):
    height, width = component.shape
    patch, radius = 9, 24
    if not (patch <= u < width - patch and patch <= v < height - patch):
        return None
    cx, cy = (u, v) + shift
    x0, y0 = max(0, cx - radius - patch), max(0, cy - radius - patch)
    x1, y1 = min(width, cx + radius + patch + 1), min(height, cy + radius + patch + 1)
    if min(x1 - x0, y1 - y0) < 2 * patch + 1:
        return None
    template = np.s_[v-patch:v+patch+1, u-patch:u+patch+1]
    sample_mask = component[template]
    match = _correlation(dst_edges[y0:y1, x0:x1], src_edges[template], sample_mask)
    _, peak, _, (dx, dy) = cv2.minMaxLoc(match)
    if peak < .76:
        return None
    match[max(0, dy-4):dy+5, max(0, dx-4):dx+5] = -1
    if peak - match.max() < .015:
        return None
    return (float(x0 + dx + patch), float(y0 + dy + patch))


def _template_pairs(src, dst, mask):
    # Search each connected arm region together first. A large silhouette and its
    # internal edges disambiguate the repetitive ribs on the fingers.
    src_edges, dst_edges = _gradients(src), _gradients(dst)
    count, labels, regions, _ = cv2.connectedComponentsWithStats(mask)
    sources, targets = [], []
    height = mask.shape[0]
    for i in range(1, count):
        x, y, w, h, area = regions[i]
        if area < 100:
            continue
        component = np.uint8(labels == i)
        keypoints = cv2.goodFeaturesToTrack(src, 160, .001, 5, mask=component)
        if keypoints is None:
            continue
        # Ignore the unknown background while looking for this robot component.
        roi = np.s_[y:y+h, x:x+w]
        scores = _correlation(dst_edges, src_edges[roi], component[roi])
        _, _, _, (px, py) = cv2.minMaxLoc(scores)
        shift = np.array([px - x, py - y])
        if np.linalg.norm(shift) > height * .40:
            shift = np.zeros(2, dtype=int)
        for u, v in np.int32(keypoints.reshape(-1, 2)):
            match = _match_corner(src_edges, dst_edges, component, u, v, shift)
            if match is not None:
                sources.append((u, v))
                targets.append(match)
    return sources, targets


class RobotFeatures:
    """Render both arms at encoder positions to identify metric image features."""

    def __init__(self, sim=None):
        self.sim = sim
        self.model = _robot_model()
        self.data = mujoco.MjData(self.model)
        self.joints = [sim.model.joint(self.model.joint(i).name).qposadr.item()
                       for i in range(self.model.njnt)] if sim is not None else None
        self.renderer = None
        self.visibility = mujoco.MjvOption()
        self.visibility.sitegroup[:] = 0
        self.sift = cv2.SIFT_create(contrastThreshold=.012, edgeThreshold=12)

    def encoder_positions(self):
        """Copy the live joint angles in CAD model order at exposure time."""
        if self.sim is None:
            raise ValueError('Supply measured encoder positions for this camera exposure')
        return self.sim.data.qpos[self.joints].copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _reference(self, camera, shape):
        height, width = shape[:2]
        if self.renderer is None or (self.renderer.height, self.renderer.width) != (height, width):
            self.close()
            self.renderer = mujoco.Renderer(self.model, height, width)
        cad = self.model.camera('cad_view')
        cad.pos[:] = camera.position
        axes = np.column_stack((camera.right, camera.up, -camera.forward))
        mujoco.mju_mat2Quat(cad.quat, axes.ravel())
        cad.fovy[:] = np.rad2deg(2 * np.arctan(height / (2 * camera.focal)))
        mujoco.mj_forward(self.model, self.data)
        renderer = self.renderer
        renderer.disable_depth_rendering()
        renderer.update_scene(self.data, camera='cad_view', scene_option=self.visibility)
        reference = renderer.render()
        renderer.enable_depth_rendering()
        depth = renderer.render()
        renderer.disable_depth_rendering()
        return reference, depth

    def _pairs(self, rgb, camera):
        reference, depth = self._reference(camera, rgb.shape)
        near = depth < self.model.stat.extent * self.model.vis.map.zfar * .999
        # Interior CAD pixels have a defined surface even where links occlude one another.
        mask = cv2.erode(near.astype(np.uint8), np.ones((3, 3), np.uint8))
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
        image_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        src, descriptors = self.sift.detectAndCompute(ref_gray, mask)
        dst, observed = self.sift.detectAndCompute(image_gray, None) if descriptors is not None else ([], None)
        src_points, dst_points = _template_pairs(ref_gray, image_gray, mask)
        if observed is not None and len(observed) >= 2:
            for one, two in cv2.BFMatcher().knnMatch(descriptors, observed, k=2):
                s, t = src[one.queryIdx].pt, dst[one.trainIdx].pt
                if one.distance < .82 * two.distance and np.linalg.norm(np.subtract(t, s)) < rgb.shape[0] * .32:
                    src_points.append(s)
                    dst_points.append(t)
        world = [camera.position + depth[int(round(v)), int(round(u))] * camera.ray((u, v))
                 for u, v in src_points]
        return np.array(world).reshape(-1, 3), np.array(dst_points).reshape(-1, 2)

    def match(self, rgb, nominal_camera, encoder_positions=None):
        """Match robot surface points using encoder positions from the RGB exposure."""
        self.data.qpos[:] = self.encoder_positions() if encoder_positions is None else encoder_positions
        world, pixels = self._pairs(rgb, nominal_camera)
        if len(world) < 10:
            return world, pixels
        # Re-render once near the observed pose: silhouettes move across curved
        # links and the jaw ribs change apparent spacing with camera pitch.
        ok, rot, trans, inliers = cv2.solvePnPRansac(
            world, pixels, nominal_camera.matrix, None, flags=cv2.SOLVEPNP_AP3P,
            iterationsCount=1200, reprojectionError=3.)
        if not ok or len(inliers) < 9:
            return world, pixels
        rotation, _ = cv2.Rodrigues(rot)
        position = -rotation.T @ trans.ravel()
        if (np.linalg.norm(position - nominal_camera.position) > .07
                or rotation[2] @ nominal_camera.forward < np.cos(np.deg2rad(17))):
            return world, pixels
        camera = replace(nominal_camera, position=position, right=rotation[0],
                         up=-rotation[1], forward=rotation[2])
        return self._pairs(rgb, camera)
