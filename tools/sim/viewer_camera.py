"""Viewer camera helpers: a high follow camera, a see-through cutaway, and a safe starting angle.

FollowCam   sits above the room (so it sees over walls and counters), keeps the robot centered as it
            walks, and slowly swings round behind it as it turns. Orbit/zoom with the mouse whenever
            you like: it keeps your angle and distance and just follows the position. F toggles
            whether it also swings round with the robot's heading.
Cutaway     hides tall things (walls, fridges, upper cabinets) that stand between the camera and the
            robot, the way games do, so the robot never disappears behind a wall. Works in any env:
            it only uses each shape's bounding box.
clear_view  starting angle that can actually see the robot (kept for scripts that want a low camera).
"""

import math

import mujoco
import numpy as np

ROBOT_PREFIX = "robot/"
HIDDEN_GROUP = 5


def robot_yaw(d: mujoco.MjData, qpos_quat_adr: int = 3) -> float:
    """Robot heading, degrees."""
    w, x, y, z = d.qpos[qpos_quat_adr:qpos_quat_adr + 4]
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap180(a):
    """Angle wrapped to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


class FollowCam:
    """Camera that follows the robot, optionally turning with its heading."""
    def __init__(self, elevation=-55.0, distance=5.0):
        """Elevation and distance; azimuth and target set on the first update."""
        self.el, self.dist = elevation, distance
        self.az = None
        self.lookat = None
        self.follow_heading = True
        self.last_set = None

    def update(self, cam: mujoco.MjvCamera, d: mujoco.MjData, body: int, dt: float, shift=0.0, aspect=1.6):
        """shift: put the robot this fraction of the window width left of center (side-by-side layout)."""
        target = d.xpos[body] + np.array([0.0, 0.0, 0.2])
        yaw = robot_yaw(d)
        if self.lookat is None:
            self.lookat, self.az = target.copy(), yaw
        elif self.last_set is not None:
            # the person orbited or zoomed with the mouse: keep their view (position still follows)
            if abs(wrap180(cam.azimuth - self.last_set[0])) > 0.2:
                self.az = cam.azimuth
                self.follow_heading = False
            if abs(cam.elevation - self.last_set[1]) > 0.2:
                self.el = cam.elevation
            if abs(cam.distance - self.last_set[2]) > 0.01:
                self.dist = cam.distance
        self.lookat += (target - self.lookat) * (1.0 - math.exp(-dt / 0.35))
        if self.follow_heading:                      # swing round behind the robot, slowly
            self.az += wrap180(yaw - self.az) * (1.0 - math.exp(-dt / 2.5))
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        a = math.radians(self.az)
        width = 2 * self.dist * math.tan(math.radians(45.0 / 2)) * aspect    # visible width at the robot (fovy 45)
        cam.lookat[:] = self.lookat + np.array([math.sin(a), -math.cos(a), 0.0]) * shift * width
        cam.azimuth, cam.elevation, cam.distance = self.az, self.el, self.dist
        self.last_set = (cam.azimuth, cam.elevation, cam.distance)

    def toggle_heading(self, d):
        """Toggle turning with the robot."""
        self.follow_heading = not self.follow_heading
        return self.follow_heading


class Cutaway:
    """Hides room geoms between the camera and the robot so it stays visible."""
    def __init__(self, m: mujoco.MjModel, min_top=1.2, linger_s=0.6):
        """Candidate geoms: drawn, not the robot, not a plane."""
        self.m = m
        robot = np.array([m.body(m.geom_bodyid[g]).name.startswith(ROBOT_PREFIX) for g in range(m.ngeom)])
        drawn = m.geom_group <= 2
        not_plane = m.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
        self.cand = np.flatnonzero(~robot & drawn & not_plane)
        self.orig_group = m.geom_group.copy()
        self.min_top, self.linger = min_top, linger_s
        self.last_hit = {}                           # geom -> time it last blocked the view

    def update(self, d: mujoco.MjData, cam: mujoco.MjvCamera, targets, now: float):
        """Hide geoms that block the view of the targets; they reappear after `linger_s`."""
        m, g = self.m, self.cand
        if len(g) == 0:
            return
        el, az = math.radians(cam.elevation), math.radians(cam.azimuth)
        fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        cam_pos = np.array(cam.lookat) - fwd * cam.distance
        R = d.geom_xmat[g].reshape(-1, 3, 3)
        c, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:] + 0.02
        center = d.geom_xpos[g] + np.einsum("nij,nj->ni", R, c)
        top = center[:, 2] + np.einsum("nj,nj->n", np.abs(R[:, 2, :]), half)
        tall = top > self.min_top
        hit = np.zeros(len(g), bool)
        for p in targets:                            # segment camera -> point, slab test in each box's frame
            a = np.einsum("nji,nj->ni", R, cam_pos - center)
            b = np.einsum("nji,nj->ni", R, np.asarray(p) - center)
            dvec = b - a
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (-half - a) / dvec, (half - a) / dvec
            tmin = np.nanmax(np.where(dvec == 0, np.where(np.abs(a) <= half, -np.inf, np.inf), np.minimum(t1, t2)), 1)
            tmax = np.nanmin(np.where(dvec == 0, np.where(np.abs(a) <= half, np.inf, -np.inf), np.maximum(t1, t2)), 1)
            hit |= (tmax >= np.maximum(tmin, 0.0)) & (tmin <= 0.97)
        for gi in g[hit & tall]:
            self.last_hit[gi] = now
        for gi, t in list(self.last_hit.items()):
            if now - t < self.linger:
                m.geom_group[gi] = HIDDEN_GROUP
            else:
                m.geom_group[gi] = self.orig_group[gi]
                del self.last_hit[gi]


def clear_view(m, d, target_body: int, distance=(2.5, 2.0, 1.5), elevation=-20.0):
    """Returns (azimuth_deg, distance, elevation) for a low camera that sees target_body."""
    target = d.xpos[target_body] + np.array([0.0, 0.0, 0.2])
    preferred = robot_yaw(d) + 180.0
    gid = np.zeros(1, np.int32)
    el = math.radians(elevation)
    for dist in distance:
        best = None
        for k in range(24):                                  # every 15 degrees
            az = preferred + (k + 1) // 2 * 15.0 * (1 if k % 2 else -1)
            a = math.radians(az)
            look = np.array([math.cos(a) * math.cos(el), math.sin(a) * math.cos(el), math.sin(el)])
            cam = target - look * dist

            def sees_robot(point):
                v = point - cam
                hit = mujoco.mj_ray(m, d, cam, v / np.linalg.norm(v), None, 1, -1, gid)
                return hit >= 0 and m.body(m.geom_bodyid[gid[0]]).name.startswith(ROBOT_PREFIX)
            if sees_robot(target) and sees_robot(d.xpos[target_body] - np.array([0.0, 0.0, 0.45])):
                best = az
                break
        if best is not None:
            return best, dist, elevation
    return preferred, distance[-1], elevation
