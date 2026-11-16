"""Teleoperate the Panthera arm from the keyboard and record demonstrations.

Keys drive a commanded end-effector pose at a fixed rate while IK makes the
simulated arm chase it. No webcam, printed markers, or calibration is required.

The target is a *pose you drive*, not a pose you jump to. Keys move a commanded
end-effector frame at a fixed rate for as long as they are held, and the IK
chases it every tick. The commanded
frame can be somewhere the arm cannot reach -- it is clamped to a box, not to
the workspace -- and when it is, you see the orange target ball part company
with the gripper and the IK residual climb. That residual is logged per step.

Episodes contain simulator state, targets, object poses, gripper commands, and
IK residuals. Replay them with:

    python teleop/keyboard.py
    python teleop/replay.py data/episode_000

Four views, all on screen at once, one per quadrant:

    shoulder | wrist        shoulder: fixed over-the-shoulder view
    ---------+---------
    overhead | chase        wrist:    first person, from a camera on link6 --
                             are the jaws lined up with the cube?
                            overhead: straight down at the table -- where is
                             everything?
                            chase:    the shoulder view, following the wrist
                             out of the fixed frame

Teleoperating from a single view means inferring the two things it cannot show
-- depth along the view axis, and anything behind the arm -- from the one it
can. Four at once turns both into something you read directly: the wrist view
answers left-right and the overhead answers how far forward, and neither has to
be inferred from the other.

The two fixed cameras live in the model rather than in this script. The teleop
wrist preview is roll-stabilized to keep the operator's horizon steady; dataset
and policy renderers use the true wrist-camera pose, including roll. A number
key blows one view up to fill the window and the same key again brings the
other three back; `g` always returns to the grid, and `--view` picks what to
start with.

`sim.mp4` records the window as it stands, so by default it is a 640x480
four-up mosaic -- about 320x240 a tile. Focus
a single view before recording if the clip is meant to be looked at closely.

`f` (or `--minecraft`) switches to a first-person scheme. The mouse yaws and
pitches the gripper, WASD slides on the heading -- W goes forward in yaw,
staying level, even if you are looking down -- and SPACE / SHIFT still lift
and drop along world +z. Heading is clamped to ±60 deg of robot +x so the
wrist cannot be asked to look behind the base. The cursor is captured while
the mode is on; `f` again gives it back and restores the world-frame keys.
The wrist view (2) is the one that looks where you are aiming.

Controls
    w / s    end-effector forward / back      (robot +x / -x)
    a / d    left / right                     (robot +y / -y)
    SPACE    up                               (robot +z)
    SHIFT    down                             (robot -z)
    u / j    pitch the gripper up / down      (about base y)
    i / k    yaw left / right                 (about base z)
    n / m    roll                             (about base x)
    side-scroll  roll                         (right / left = + / -)
    o / l    open / close the gripper
    [ / ]    slower / faster
    c        re-centre the target on the arm (after reaching out of range)
    r        reset the arm near home and discard the unsaved take
    e        save the current take (recording starts automatically on movement)
    x        discard the episode in progress
    f        Minecraft look on / off
    q / ESC  quit (ESC first releases Minecraft look)

    1 2 3 4  fill the window with shoulder / wrist / overhead / chase
    g        back to all four
    drag / vertical scroll  orbit and zoom -- acts on whichever view the
                            pointer is over (the wrist and overhead cameras
                            are fixed and do not move)

    Minecraft look (`f` / `--minecraft`)
    w / s    forward / back on the heading (level, not along the look)
    a / d    strafe left / right on that heading
    SPACE    up                               (world +z, always)
    SHIFT    down                             (world -z, always)
    mouse    yaw / pitch the gripper
    side-scroll  roll about the look axis
    l-click  close the gripper
    r-click  open the gripper
    n / m    roll about the look axis
    o / l    open / close (keys still work)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sim"))

# We render into our own GLFW window rather than offscreen, so the window's
# context is the one MuJoCo draws through. Set before mujoco is imported.
os.environ.setdefault("MUJOCO_GL", "glfw")

import cv2  # noqa: E402
import glfw  # noqa: E402
import mujoco  # noqa: E402

from panthera_env import PantheraSim, mat_to_quat, quat_mul  # noqa: E402
from episode import Episode, next_episode_dir  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data"
VIEW_AZIMUTH, VIEW_ELEVATION, VIEW_DISTANCE = 14.0, -34.0, 1.20
VIEW_LOOKAT = (0.44, 0.0, 0.12)
DEFAULT_ARM_START_RANGE = (0.30, 0.48, -0.16, 0.16, 0.10, 0.40)
SCROLL_ROLL_STEP = np.deg2rad(5.0)


def randomize_arm_start(sim: PantheraSim, start_range, rng=None) -> None:
    """Place the arm at a random reachable point in a Cartesian box.

    A downward grasp orientation makes low tabletop-adjacent starts reachable;
    the home orientation cannot reach much below 17 cm. This is a teleport
    before recording, so the first command has no artificial slew.
    """
    rng = np.random.default_rng() if rng is None else rng
    bounds = np.asarray(start_range, dtype=float)
    lo = bounds[[0, 2, 4]]
    hi = bounds[[1, 3, 5]]
    pitch = np.deg2rad(30.0)
    grasp_quat = mat_to_quat(np.array([
        [np.cos(pitch), 0.0, np.sin(pitch)],
        [0.0, 1.0, 0.0],
        [-np.sin(pitch), 0.0, np.cos(pitch)],
    ]))

    # Rejection keeps this usable with custom ranges that partly extend beyond
    # the workspace. The default range almost always succeeds on its first try.
    for _ in range(100):
        target = rng.uniform(lo, hi)
        q, pos_err, rot_err = sim.ik(
            target, grasp_quat, q_init=sim.q, max_joint_step=None)
        if pos_err <= 1e-3 and rot_err <= 1e-3:
            break
    else:
        return
    sim.data.qpos[sim.arm_qadr] = q
    sim.data.qvel[sim.arm_dofadr] = 0.0
    sim.set_arm_ctrl(q, immediate=True)
    mujoco.mj_forward(sim.model, sim.data)


def _add_geom(scene, geom_type, size, pos, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, geom_type, np.asarray(size, dtype=np.float64),
                        np.asarray(pos, dtype=np.float64), np.eye(3).ravel(),
                        np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def _add_box(scene, lo, hi, rgba) -> None:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    _add_geom(scene, mujoco.mjtGeom.mjGEOM_BOX, (hi - lo) / 2,
              (hi + lo) / 2, rgba)


def _add_sphere(scene, pos, radius, rgba) -> None:
    _add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE,
              (radius, radius, radius), pos, rgba)

# Held-key -> end-effector velocity, in the robot base frame. Rates are per
# second and scaled by `--speed`; the loop multiplies by the real elapsed time,
# so the arm covers the same distance per second whatever the frame rate.
TRANSLATE = {
    glfw.KEY_W: (0, +1.0), glfw.KEY_S: (0, -1.0),
    glfw.KEY_A: (1, +1.0), glfw.KEY_D: (1, -1.0),
    glfw.KEY_SPACE: (2, +1.0),
    glfw.KEY_LEFT_SHIFT: (2, -1.0), glfw.KEY_RIGHT_SHIFT: (2, -1.0),
}

# Held-key -> angular velocity about a base-frame axis. Base frame, not the
# gripper's own: "pitch down" should mean the same thing wherever the wrist
# happens to be pointing, and a body-frame rotation would instead mean
# something different after every yaw.
ROTATE = {
    glfw.KEY_U: (1, +1.0), glfw.KEY_J: (1, -1.0),   # pitch, about +y
    glfw.KEY_I: (2, +1.0), glfw.KEY_K: (2, -1.0),   # yaw,   about +z
    glfw.KEY_N: (0, +1.0), glfw.KEY_M: (0, -1.0),   # roll,  about +x
}

GRIP = {glfw.KEY_O: +1.0, glfw.KEY_L: -1.0}

# The commanded frame is clamped to this box. Deliberately larger than the
# arm's reach -- clamping to the workspace would hide the out-of-reach case
# that the residual is there to show -- but small enough that a key left held
# down does not send the target to the next postcode.
DEFAULT_REGION = (0.10, 0.70, -0.40, 0.40, 0.01, 0.55)

# Minecraft look stores yaw/pitch as Euler angles so the gripper cannot roll
# just from looking around. ±89 deg, not ±90: at the pole the no-roll "left"
# axis vanishes and a pixel of mouse motion would spin the jaws.
PITCH_LIMIT = np.deg2rad(89.0)
# Heading stays in a forward cone. The workspace sits in front of the base;
# past about 60 deg the wrist has to unwind through poses that flip joint1
# and the IK residual climbs for no useful view.
YAW_LIMIT = np.deg2rad(60.0)


# All four are on screen at once, one per quadrant, in this order. `shoulder`
# and `chase` drive a free camera (orbitable with the mouse); `wrist` and
# `overhead` are cameras defined in the model by sim/make_mjcf.py.
VIEWS = ("shoulder", "wrist", "overhead", "chase")
VIEW_KEYS = {glfw.KEY_1: 0, glfw.KEY_2: 1, glfw.KEY_3: 2, glfw.KEY_4: 3}


def quadrants(w: int, h: int) -> list["mujoco.MjrRect"]:
    """The four cells of the grid, in `VIEWS` order, bottom-left origin.

    The split is `w // 2` with the remainder given to the right and top cells
    rather than `w / 2` rounded twice, so an odd window width leaves no
    one-pixel column of stale framebuffer down the seam.
    """
    hw, hh = w // 2, h // 2
    return [mujoco.MjrRect(0, hh, hw, h - hh),        # top left
            mujoco.MjrRect(hw, hh, w - hw, h - hh),   # top right
            mujoco.MjrRect(0, 0, hw, hh),             # bottom left
            mujoco.MjrRect(hw, 0, w - hw, hh)]        # bottom right


def tiles(views, focus: str, w: int, h: int) -> list[tuple[str, "mujoco.MjrRect"]]:
    """(view, viewport) pairs to render this frame.

    One entry filling the window when a view is focused, otherwise one per
    quadrant. Views are dropped from `views` when their camera is missing from
    the model, so this has to cope with fewer than four: the leftover cells are
    blanked by the caller rather than left showing the last frame rendered
    into them.
    """
    if focus:
        return [(focus, mujoco.MjrRect(0, 0, w, h))]
    return list(zip(views, quadrants(w, h)))


def set_view(cam, model, name: str) -> str:
    """Point `cam` at one of `VIEWS`. Returns the view actually selected.

    A model-defined camera that is missing means the XML predates it rather
    than that anything is wrong with the request -- panthera.xml and scene.xml
    are build products -- so say which command puts it back and leave the
    current view alone instead of dying mid-episode.
    """
    if name in ("shoulder", "chase"):
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.azimuth, cam.elevation = VIEW_AZIMUTH, VIEW_ELEVATION
        cam.distance = VIEW_DISTANCE
        cam.lookat[:] = VIEW_LOOKAT
        if name == "chase":
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = body
            cam.distance, cam.elevation = 0.75, -25.0
        return name
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cid < 0:
        print(f"no {name!r} camera in the scene -- "
              f"re-run `python sim/make_mjcf.py` to add it")
        return ""
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = cid
    return name


class Input:
    """GLFW keyboard and mouse state for the teleop window.

    Continuous motion polls `held()`: GLFW reports a key as down for as long as
    it is down, which is what a velocity command wants. Key *events* (record,
    reset, quit) come through the callback into `taps`, because polling them
    would fire once per frame for as long as the finger stayed on the key.
    """

    def __init__(self, window):
        self.window = window
        self.taps: list[int] = []
        self._last = (0.0, 0.0)
        self._buttons = {"left": False, "middle": False, "right": False}
        # Mouse-look is opt-in. While it is on, motion accumulates here instead
        # of orbiting a camera, and the OS cursor is hidden and captured.
        self.fps_look = False
        self.look_delta = np.zeros(2)
        self.roll_delta = 0.0
        self._look_armed = False
        glfw.set_key_callback(window, self._on_key)
        glfw.set_cursor_pos_callback(window, self._on_move)
        glfw.set_mouse_button_callback(window, self._on_button)
        glfw.set_scroll_callback(window, self._on_scroll)
        # Set by the caller before the first frame. `tiles` is rewritten every
        # frame with the current layout, so the mouse always acts on the view
        # the pointer is actually over.
        self.model = None
        self.scene = None
        self.cams: dict = {}
        self.tiles: list = []

    def _on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.taps.append(key)

    def _on_button(self, window, button, action, mods):
        self._buttons["left"] = glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._buttons["middle"] = glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self._buttons["right"] = glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self._last = glfw.get_cursor_pos(window)

    def under_cursor(self):
        """The camera of the tile the pointer is over, if it can be moved.

        Fixed cameras are excluded: they have no lookat or azimuth to move, so
        dragging one would silently wind up a pose that never shows up.
        """
        x, y = glfw.get_cursor_pos(self.window)
        _, wh = glfw.get_window_size(self.window)
        fw, fh = glfw.get_framebuffer_size(self.window)
        # Cursor coordinates are in window units from the top left; viewports
        # are in framebuffer pixels from the bottom left. On a HiDPI screen
        # those are not the same units.
        k = fh / max(wh, 1)
        x, y = x * k, fh - y * k
        for name, r in self.tiles:
            if r.left <= x < r.left + r.width and r.bottom <= y < r.bottom + r.height:
                cam = self.cams.get(name)
                if cam is not None and cam.type != mujoco.mjtCamera.mjCAMERA_FIXED:
                    return cam
                return None
        return None

    def _on_move(self, window, x, y):
        dx, dy = x - self._last[0], y - self._last[1]
        self._last = (x, y)
        if self.fps_look:
            # The first event after capture is a jump to the virtual origin
            # on some platforms; swallow it so the gripper does not twitch.
            if not self._look_armed:
                self._look_armed = True
                return
            self.look_delta[0] += dx
            self.look_delta[1] += dy
            return
        if not any(self._buttons.values()):
            return
        cam = self.under_cursor()
        if cam is None:
            return
        if self._buttons["right"]:
            act = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._buttons["left"]:
            act = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            act = mujoco.mjtMouse.mjMOUSE_ZOOM
        _, h = glfw.get_window_size(window)
        mujoco.mjv_moveCamera(self.model, act, dx / h, dy / h, self.scene, cam)

    def _on_scroll(self, window, dx, dy):
        # GLFW reports horizontal and vertical scrolling independently. Keep
        # vertical scrolling for camera zoom, and make a horizontal wheel or
        # two-finger side-scroll roll the commanded gripper in either mode.
        self.roll_delta += dx
        if self.fps_look:
            return
        cam = self.under_cursor()
        if cam is not None:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                                  0.0, -0.05 * dy, self.scene, cam)

    def set_fps_look(self, on: bool) -> None:
        """Capture or release the cursor for Minecraft-style mouse look."""
        if on == self.fps_look:
            return
        self.fps_look = on
        self.look_delta[:] = 0
        self._look_armed = False
        self._last = glfw.get_cursor_pos(self.window)
        if on:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
            if (hasattr(glfw, "raw_mouse_motion_supported")
                    and glfw.raw_mouse_motion_supported()):
                glfw.set_input_mode(self.window, glfw.RAW_MOUSE_MOTION,
                                    glfw.TRUE)
        else:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def drain_look(self) -> np.ndarray:
        d = self.look_delta.copy()
        self.look_delta[:] = 0
        return d

    def drain_roll(self) -> float:
        d = self.roll_delta
        self.roll_delta = 0.0
        return d

    def held(self, key) -> bool:
        return glfw.get_key(self.window, key) == glfw.PRESS

    def button(self, name: str) -> bool:
        return self._buttons[name]

    def drain(self) -> list[int]:
        taps, self.taps = self.taps, []
        return taps


def axis_angle_quat(axis: int, angle: float) -> np.ndarray:
    q = np.zeros(4)
    v = np.zeros(3)
    v[axis] = 1.0
    mujoco.mju_axisAngle2Quat(q, v, angle)
    return q


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def roll_stabilized_camera_pose(model: mujoco.MjModel,
                                data: mujoco.MjData,
                                camera_id: int) -> tuple[np.ndarray, np.ndarray]:
    """World pose for a body camera with its parent body's roll removed.

    This is used only by the interactive wrist preview. It preserves the
    camera's local mount position and tilt, and follows the gripper's position,
    yaw, and pitch, but constructs a level parent frame from the approach axis.
    The model camera and ``data.cam_*`` remain unchanged for other renderers.
    """
    body_id = model.cam_bodyid[camera_id]
    body_R = data.xmat[body_id].reshape(3, 3)
    forward = body_R[:, 0]
    world_up = np.array([0.0, 0.0, 1.0])
    left = np.cross(world_up, forward)
    norm = np.linalg.norm(left)
    if norm < 1e-8:
        # The regular teleop pitch limit avoids this pole, but world-frame
        # controls can still reach it. Choose a deterministic level axis.
        left = np.cross(np.array([1.0, 0.0, 0.0]), forward)
        norm = np.linalg.norm(left)
    left /= norm
    up = np.cross(forward, left)
    level_body_R = np.column_stack((forward, left, up))
    camera_R = level_body_R @ quat_to_mat(model.cam_quat[camera_id])
    camera_pos = data.xpos[body_id] + level_body_R @ model.cam_pos[camera_id]
    return camera_pos, camera_R


def look_from_quat(q: np.ndarray) -> tuple[float, float, float]:
    """Yaw / pitch / roll of a gripper quat, in the Minecraft convention.

    Body +x is the look/approach axis. Yaw is about world +z (0 = +x, +yaw
    toward +y). Pitch is elevation of +x (positive looks up). Roll is about
    +x, zero when the jaws stay level -- body +y horizontal.
    """
    R = quat_to_mat(q)
    forward = R[:, 0]
    yaw = float(np.arctan2(forward[1], forward[0]))
    pitch = float(np.arctan2(forward[2], np.hypot(forward[0], forward[1])))
    world_up = np.array([0.0, 0.0, 1.0])
    left_ref = np.cross(world_up, forward)
    ln = np.linalg.norm(left_ref)
    if ln < 1e-8:
        left_ref = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    else:
        left_ref = left_ref / ln
    up_ref = np.cross(forward, left_ref)
    left = R[:, 1]
    roll = float(np.arctan2(np.dot(left, up_ref), np.dot(left, left_ref)))
    return yaw, pitch, roll


def quat_from_look(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Inverse of `look_from_quat`: build a gripper quat from those angles."""
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    forward = np.array([cp * cy, cp * sy, sp])
    left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    ln = np.linalg.norm(left)
    if ln < 1e-8:
        left = np.array([-sy, cy, 0.0])
    else:
        left = left / ln
    up = np.cross(forward, left)
    if roll:
        c, s = np.cos(roll), np.sin(roll)
        left, up = c * left + s * up, -s * left + c * up
    return mat_to_quat(np.column_stack([forward, left, up]))


def read_frame(ctx, viewport) -> np.ndarray:
    """Pull the rendered frame back off the GPU, as BGR for the video writer."""
    rgb = np.empty((viewport.height, viewport.width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb, None, viewport, ctx)
    return cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--width", type=int, default=1100, help="window width")
    ap.add_argument("--height", type=int, default=825, help="window height")
    ap.add_argument("--speed", type=float, default=0.18,
                    help="translation rate, metres per second held (default 0.18)")
    ap.add_argument("--rot-speed", type=float, default=1.2,
                    help="rotation rate, radians per second held (default 1.2)")
    ap.add_argument("--grip-speed", type=float, default=2.5,
                    help="gripper rate, full travel per second (default 2.5)")
    ap.add_argument("--no-rotation", dest="rotation", action="store_false",
                    help="position only; hold the grasp orientation fixed")
    ap.add_argument("--no-video", dest="video", action="store_false",
                    help="log arrays only, no mp4")
    ap.add_argument("--max-duration", type=float, default=60.0,
                    help="maximum seconds in one automatically started take "
                         "(default 60)")
    ap.add_argument("--arm-start-range", type=float, nargs=6,
                    default=DEFAULT_ARM_START_RANGE,
                    metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                    help="box for random initial end-effector positions "
                         "(default: x .30-.48, y -.16-.16, z .10-.40 m)")
    ap.add_argument("--view", choices=("grid",) + VIEWS, default="grid",
                    help="start with all four views (default) or one of them "
                         "filling the window; 1-4 and g switch at any time")
    ap.add_argument("--region", type=float, nargs=6, default=DEFAULT_REGION,
                    metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                    help="box the commanded target is clamped to, metres")
    ap.add_argument("--minecraft", action="store_true",
                    help="start in Minecraft look: mouse aims, WASD stays "
                         "level on the heading, SPACE/SHIFT stay world-vertical")
    ap.add_argument("--look-sens", type=float, default=0.15,
                    help="Minecraft mouse-look sensitivity, degrees per pixel "
                         "(default 0.15)")
    ap.add_argument("--yaw-limit", type=float, default=np.rad2deg(YAW_LIMIT),
                    help="Minecraft look: clamp heading to ±this many degrees "
                         "from robot +x (default 60)")
    args = ap.parse_args()

    if args.max_duration <= 0:
        ap.error("--max-duration must be greater than zero")
    if any(args.arm_start_range[i] >= args.arm_start_range[i + 1]
           for i in (0, 2, 4)):
        ap.error("each --arm-start-range lower bound must be below its upper bound")

    sim = PantheraSim()
    randomize_arm_start(sim, args.arm_start_range)
    lo = np.array([min(args.region[0], args.region[1]),
                   min(args.region[2], args.region[3]),
                   min(args.region[4], args.region[5])])
    hi = np.array([max(args.region[0], args.region[1]),
                   max(args.region[2], args.region[3]),
                   max(args.region[4], args.region[5])])

    if not glfw.init():
        raise SystemExit("could not initialise GLFW -- is there a display?")
    window = glfw.create_window(args.width, args.height,
                                "RobotArmLearning  |  keyboard teleop", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("could not open a window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # One camera per view, set up once: a free camera keeps whatever the mouse
    # has done to it, so rebuilding them per frame would undo every orbit.
    cams = {}
    for name in VIEWS:
        cam = mujoco.MjvCamera()
        if set_view(cam, sim.model, name):
            cams[name] = cam
    if not cams:
        raise SystemExit("no usable views")
    views = tuple(cams)
    focus = "" if args.view == "grid" else args.view
    if focus and focus not in cams:
        focus = ""
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(sim.model, maxgeom=2000)
    ctx = mujoco.MjrContext(sim.model,
                            mujoco.mjtFontScale.mjFONTSCALE_150.value)

    inp = Input(window)
    inp.model, inp.scene, inp.cams = sim.model, scene, cams

    meta = {
        "robot": "Panthera-HT (HighTorque) 6-DoF + parallel gripper",
        "scene": str(sim.scene_path.relative_to(REPO_ROOT)),
        "control_mode": "keyboard",
        "simulation_dynamics": sim.dynamics,
        "track_rotation": bool(args.rotation),
        "recording": {"trigger": "keyboard_or_mouse_motion",
                      "max_duration_s": args.max_duration},
        "speed_m_s": args.speed,
        "rot_speed_rad_s": args.rot_speed,
        "look_mode": "minecraft" if args.minecraft else "world",
        "look_sens_deg_px": args.look_sens,
        "yaw_limit_deg": args.yaw_limit,
        "arm_joints": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        "gripper_open_m": 0.04,
        "arm_start_range_m": list(args.arm_start_range),
        "objects": sim.object_names,
        "region": [lo.tolist(), hi.tolist()],
        # What sim.mp4 shows: one view's name, or "grid" for all four.
        # Overwritten at save time with whatever was actually up, and the
        # logged arrays are unaffected by it either way.
        "view": args.view,
        "frames": {
            "ee_*/target_*": "robot base frame",
            "quat": "(w, x, y, z)",
        },
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    def set_title(minecraft: bool) -> None:
        title = "RobotArmLearning  |  keyboard teleop"
        if minecraft:
            title += "  |  minecraft"
        glfw.set_window_title(window, title)

    tp, tq = sim.ee_pose()
    tp = tp.copy()
    tq = tq.copy()
    yaw_limit = np.deg2rad(args.yaw_limit)

    def sync_look(q) -> list[float]:
        y, p, r = look_from_quat(q)
        return [float(np.clip(y, -yaw_limit, yaw_limit)),
                float(np.clip(p, -PITCH_LIMIT, PITCH_LIMIT)),
                float(r)]

    look = sync_look(tq)
    minecraft = bool(args.minecraft)
    look_sens = np.deg2rad(args.look_sens)
    inp.set_fps_look(minecraft)
    set_title(minecraft)
    q_cmd = sim.q.copy()
    grip = 1.0
    gain = 1.0
    episode: Episode | None = None
    recording = False
    motion_armed = True
    stuck_n = 0
    last = time.time()
    physics_remainder = 0.0
    fps_ema = 60.0
    print(__doc__.split("Controls")[1])
    if minecraft:
        print("minecraft look on -- mouse aims, WASD stays level on the heading")
        if not args.rotation:
            print("  --no-rotation is on: the mouse will not aim, but WASD "
                  "still moves on the current heading")

    while not glfw.window_should_close(window):
        glfw.poll_events()
        now = time.time()
        dt = min(max(now - last, 1e-3), 0.1)
        last = now
        fps_ema = 0.9 * fps_ema + 0.1 / dt

        # ---- discrete keys ----
        quit_now = False
        for key in inp.drain():
            if key == glfw.KEY_Q:
                quit_now = True
            elif key == glfw.KEY_ESCAPE:
                # ESC first gives the cursor back, the way Minecraft does;
                # a second press (or q) still quits.
                if minecraft:
                    minecraft = False
                    inp.set_fps_look(False)
                    set_title(False)
                    meta["look_mode"] = "world"
                    print("minecraft look off")
                else:
                    quit_now = True
            elif key == glfw.KEY_E:
                if episode is not None and len(episode):
                    meta["view"] = focus or "grid"
                    meta["look_mode"] = "minecraft" if minecraft else "world"
                    episode.save(next_episode_dir(args.out), meta,
                                 fps_ema, args.video)
                    episode = None
                    recording = False
                    motion_armed = False
                else:
                    print("  no take to save -- move the arm first")
            elif key == glfw.KEY_X:
                if episode is not None:
                    print(f"  discarded {len(episode)} steps")
                    episode = None
                    recording = False
                    motion_armed = False
            elif key == glfw.KEY_R:
                if episode is not None:
                    print(f"  reset discarded {len(episode)} unsaved steps")
                    episode = None
                    recording = False
                    motion_armed = False
                sim.reset()
                randomize_arm_start(sim, args.arm_start_range)
                q_cmd = sim.q.copy()
                tp, tq = (p.copy() for p in sim.ee_pose())
                look[:] = sync_look(tq)
                stuck_n = 0
                print("arm reset near home")
            elif key == glfw.KEY_C:
                # The commanded frame is free to run past the arm's reach, so
                # it can end up somewhere the gripper never followed it to.
                # This drops it back onto wherever the arm actually got to.
                tp, tq = (p.copy() for p in sim.ee_pose())
                look[:] = sync_look(tq)
                print("target re-centred on the arm")
            elif key == glfw.KEY_LEFT_BRACKET:
                gain = max(0.1, gain / 1.25)
                print(f"speed x{gain:.2f}")
            elif key == glfw.KEY_G:
                focus = ""
            elif key in VIEW_KEYS:
                # The same number twice is a toggle: blow a view up to look at
                # something, press it again to get the other three back.
                want = VIEWS[VIEW_KEYS[key]]
                focus = "" if want == focus or want not in cams else want
            elif key == glfw.KEY_RIGHT_BRACKET:
                gain = min(4.0, gain * 1.25)
                print(f"speed x{gain:.2f}")
            elif key == glfw.KEY_F:
                minecraft = not minecraft
                inp.set_fps_look(minecraft)
                set_title(minecraft)
                meta["look_mode"] = "minecraft" if minecraft else "world"
                if minecraft:
                    look[:] = sync_look(tq)
                    print("minecraft look on -- mouse aims, WASD stays level "
                          "on the heading")
                else:
                    print("minecraft look off")
        if quit_now:
            break

        # ---- held keys -> commanded pose ----
        v = np.zeros(3)
        moving = False
        if minecraft:
            # WASD stay on the ground plane of the current heading: flatten
            # the look axis onto xy so looking down and holding W slides
            # forward, not into the table. SPACE / SHIFT are the only
            # vertical keys, and they stay world-z.
            fwd = quat_to_mat(tq)[:, 0].copy()
            fwd[2] = 0.0
            n = np.linalg.norm(fwd)
            if n < 1e-8:
                yaw = look[0]
                fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            else:
                fwd = fwd / n
            right = np.array([fwd[1], -fwd[0], 0.0])
            move = np.zeros(3)
            if inp.held(glfw.KEY_W):
                move += fwd
            if inp.held(glfw.KEY_S):
                move -= fwd
            if inp.held(glfw.KEY_D):
                move += right
            if inp.held(glfw.KEY_A):
                move -= right
            n = np.linalg.norm(move)
            if n > 1e-9:
                v += move / n
            if inp.held(glfw.KEY_SPACE):
                v[2] += 1.0
            if inp.held(glfw.KEY_LEFT_SHIFT) or inp.held(glfw.KEY_RIGHT_SHIFT):
                v[2] -= 1.0
        else:
            for key, (axis, sign) in TRANSLATE.items():
                if inp.held(key):
                    v[axis] += sign
        moving = bool(np.linalg.norm(v) > 0)
        tp = np.clip(tp + v * args.speed * gain * dt, lo, hi)

        scroll_roll = inp.drain_roll()
        if minecraft and not args.rotation:
            inp.drain_look()
        if args.rotation:
            if minecraft:
                yaw, pitch, roll = look
                dx, dy = inp.drain_look()
                moving = moving or bool(dx or dy)
                # GLFW y grows down the window; mouse-up (negative dy) looks
                # up. Mouse-right decreases yaw, toward robot -y.
                yaw -= dx * look_sens
                pitch -= dy * look_sens
                step = args.rot_speed * gain * dt
                if inp.held(glfw.KEY_U):
                    pitch -= step
                    moving = True
                if inp.held(glfw.KEY_J):
                    pitch += step
                    moving = True
                if inp.held(glfw.KEY_I):
                    yaw += step
                    moving = True
                if inp.held(glfw.KEY_K):
                    yaw -= step
                    moving = True
                if inp.held(glfw.KEY_N):
                    roll += step
                    moving = True
                if inp.held(glfw.KEY_M):
                    roll -= step
                    moving = True
                if scroll_roll:
                    roll += scroll_roll * SCROLL_ROLL_STEP * gain
                    moving = True
                look[:] = (float(np.clip(yaw, -yaw_limit, yaw_limit)),
                           float(np.clip(pitch, -PITCH_LIMIT, PITCH_LIMIT)),
                           roll)
                yaw, pitch, roll = look
                tq = quat_from_look(yaw, pitch, roll)
            else:
                if scroll_roll:
                    moving = True
                    tq = quat_mul(
                        axis_angle_quat(0, scroll_roll * SCROLL_ROLL_STEP * gain),
                        tq)
                for key, (axis, sign) in ROTATE.items():
                    if inp.held(key):
                        moving = True
                        tq = quat_mul(
                            axis_angle_quat(axis, sign * args.rot_speed * gain * dt),
                            tq)
                mujoco.mju_normalize4(tq)

        for key, sign in GRIP.items():
            if inp.held(key):
                moving = True
                grip = float(np.clip(grip + sign * args.grip_speed * dt, 0, 1))
        if minecraft:
            # Held the same way as o/l: 1 is open, 0 is closed. Left click
            # closes, right click opens -- Minecraft-style use / place.
            if inp.button("left"):
                moving = True
                grip = float(np.clip(grip - args.grip_speed * dt, 0, 1))
            if inp.button("right"):
                moving = True
                grip = float(np.clip(grip + args.grip_speed * dt, 0, 1))

        if not moving:
            motion_armed = True
        if episode is None and motion_armed and moving:
            episode = Episode()
            recording = True
            motion_armed = False
            print(f"movement detected -- recording (up to {args.max_duration:g}s)")
        if recording and now - episode.t0 >= args.max_duration:
            recording = False
            print(f"  {args.max_duration:g}s limit reached -- press e to save")

        # ---- retarget -> IK -> sim ----
        q_cmd, ik_p, ik_r = sim.ik(tp, tq if args.rotation else None,
                                   q_init=q_cmd)
        sim.set_arm_ctrl(q_cmd)
        sim.set_gripper(grip)
        # Carry fractional ticks forward instead of losing simulation time on
        # every display frame. Record actual simulation time independently.
        physics_remainder += dt / sim.dt
        physics_steps = int(physics_remainder)
        physics_remainder -= physics_steps
        sim.step(physics_steps)

        # ---- render ----
        # Each view is a separate pass into its own sub-viewport of the one
        # framebuffer, which is how MuJoCo's own `simulate` draws its
        # picture-in-picture. The scene is rebuilt per pass because the target
        # decorations differ between views, and because a scene carries the
        # camera's own lighting and reflections.
        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        layout = tiles(views, focus, w, h)
        inp.tiles = layout
        for name, r in layout:
            if name == "wrist":
                # Level only this interactive preview. mjv_updateScene copies
                # the pose into `scene`, after which the real model pose is put
                # back so episode/model rendering still sees physical roll.
                camera_id = cams[name].fixedcamid
                camera_pos = sim.data.cam_xpos[camera_id].copy()
                camera_xmat = sim.data.cam_xmat[camera_id].copy()
                stable_pos, stable_R = roll_stabilized_camera_pose(
                    sim.model, sim.data, camera_id)
                sim.data.cam_xpos[camera_id] = stable_pos
                sim.data.cam_xmat[camera_id] = stable_R.ravel()
                try:
                    mujoco.mjv_updateScene(
                        sim.model, sim.data, opt, None, cams[name],
                        mujoco.mjtCatBit.mjCAT_ALL.value, scene)
                finally:
                    sim.data.cam_xpos[camera_id] = camera_pos
                    sim.data.cam_xmat[camera_id] = camera_xmat
            else:
                mujoco.mjv_updateScene(
                    sim.model, sim.data, opt, None, cams[name],
                    mujoco.mjtCatBit.mjCAT_ALL.value, scene)
            if name != "wrist":
                # From the wrist the ball is a few centimetres off the lens and
                # fills the frame, hiding the jaws it is there to be compared
                # against; the region box would wrap around the camera entirely.
                _add_box(scene, lo, hi, (0.2, 0.8, 1.0, 0.06))
                _add_sphere(scene, tp, 0.018, (1.0, 0.5, 0.0, 0.85))
            mujoco.mjr_render(r, scene, ctx)
            if not focus:
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_NORMAL.value,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT.value, r, name, "", ctx)
        if not focus:
            # Nothing renders into a quadrant with no view, and nothing clears
            # it either -- it would hold whatever was last drawn there.
            for r in quadrants(w, h)[len(layout):]:
                mujoco.mjr_rectangle(r, 0, 0, 0, 1)
            # Seams, so four skyboxes do not read as one continuous image.
            mujoco.mjr_rectangle(mujoco.MjrRect(w // 2 - 1, 0, 2, h), 0, 0, 0, 1)
            mujoco.mjr_rectangle(mujoco.MjrRect(0, h // 2 - 1, w, 2), 0, 0, 0, 1)

        rec = recording
        stuck_n = stuck_n + 1 if ik_p > 5e-3 else 0
        record_status = (f"REC {now - episode.t0:.1f}/{args.max_duration:g}s"
                         if rec else
                         (f"READY {episode.rows[-1]['t']:.1f}s -- e to save"
                          if episode is not None else "waiting for movement"))
        status = (f"{record_status}   "
                  f"{fps_ema:4.1f} Hz   grip {grip:.2f}   speed x{gain:.2f}"
                  f"   {focus or 'all views'}"
                  f"   {'minecraft' if minecraft else 'world'}")
        if stuck_n > 5:
            status += f"   IK {ik_p * 1000:.0f} mm -- out of reach"
        # Top right, so it does not sit on top of the top-left tile's label.
        views_help = ("all four views" if not focus
                      else f"{focus} -- press again for all")
        if minecraft:
            keys = ("wasd / space / shift\nmouse   n m   side-scroll\nlmb / rmb\n"
                    "1 2 3 4 / g\nc\nr\n"
                    "e / x\nf / esc\nq")
            labels = ("move (heading)\nlook / roll\nclose / open\n"
                      + views_help
                      + "\nre-centre target\nreset\nsave / discard\n"
                        "world-frame keys\nquit")
        else:
            keys = ("wasd / space / shift\nu j   i k   n m / side-scroll\no l\n"
                    "1 2 3 4 / g\nc\nr\n"
                    "e / x\nf\nq")
            labels = ("move\npitch yaw roll\ngripper\n"
                      + views_help
                      + "\nre-centre target\nreset\nsave / discard\n"
                        "minecraft look\nquit")
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL.value,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT.value, viewport,
            keys, labels, ctx)
        # Both lines in the left column: as a right-hand column the target
        # readout butts straight up against the end of the status line.
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL.value,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT.value, viewport,
            f"target  {tp[0]:+.3f} {tp[1]:+.3f} {tp[2]:+.3f}\n{status}",
            "", ctx)

        # ---- log ----
        # Read back before the buffer swap, and only while recording: the
        # round trip off the GPU costs more than the render itself.
        if rec:
            ee_p, ee_q = sim.ee_pose()
            obj_p, obj_q = sim.object_poses()
            episode.add({
                "t": now - episode.t0,
                "sim_time": float(sim.data.time),
                "physics_steps": physics_steps,
                "finger_q": sim.data.qpos[sim.finger_qadr].copy(),
                "finger_dq": sim.data.qvel[sim.finger_dofadr].copy(),
                "q": sim.q, "dq": sim.dq, "ctrl": sim.data.ctrl.copy(),
                "ee_pos": ee_p, "ee_quat": ee_q,
                "obj_pos": obj_p, "obj_quat": obj_q,
                "target_pos": np.asarray(tp), "target_quat": np.asarray(tq),
                "gripper": grip,
                "ik_pos_err": ik_p, "ik_rot_err": ik_r,
            }, sim=cv2.resize(read_frame(ctx, viewport), (640, 480))
               if args.video else None)

        glfw.swap_buffers(window)

    if episode is not None and len(episode):
        print(f"  quit with an unsaved take ({len(episode)} steps); discarded")
    glfw.terminate()


if __name__ == "__main__":
    main()
