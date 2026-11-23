"""Learning objects on first sight (mixin for core/skills.py Skills).

propose_points(camera):  numbered candidate points on the image (perception/candidates.py), for the LLM to choose
                         from. Nothing is learned yet.
learn(name, points, parts, deformable):
    1. the chosen points' regions -> the object's exact pixels (against the empty-table image) -> its DINO examples
       and colour, saved in the seen-objects list (perception/memory.py);
    2. its position from the overhead view at a guessed height, then the arm takes the wrist camera above it, the
       wrist view finds it with the new examples, and the two views together give its height (core/learn.py);
    3. footprint, outline and shape class at that height (from the wrist view: closer, so finer), the wrist view's
       own examples added, parts located on the object, then home.
From then on `find("<name>")` finds it with any segmenter that can (dino_local always can), and pick, place and the
checks use its measured shape.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from core.geometry import footprint, gripper_rotation, pixels_to_plane, rot_z
from core.learn import examples, measure_height, object_pixels, obstacles, shape_of, top_outline
from core.types import Result
from perception.candidates import draw as draw_points, describe, patch_lab, propose, table_roi
from perception.dino import PATCH
from perception.memory import Entry, Part, Shape


class LearnSkills:
    """Expects Skills' attributes (robot, cfg, table, memory, world, _observe, _emit, _record, _travel, home)."""

    def init_learning(self) -> None:
        self.proposals: dict[str, tuple] = {}          # camera -> (observation, features, proposals, step)
        self._empty: dict[str, dict] = {}              # camera -> {'rgb', 'features', 'lab'}
        self._dino = None

    @property
    def dino(self):
        if self._dino is None:
            from perception.dino import shared
            self._dino = getattr(self.segmenter, 'features', None) or shared()
        return self._dino

    def empty_table(self, camera: str) -> dict | None:
        """The camera's empty-table image and its features: from the robot (the sim renders one), or from the
        file tools/capture_empty_table.py saved (`memory/empty_<camera>.png`)."""
        if camera in self._empty:
            return self._empty[camera]
        rgb = None
        if hasattr(self.robot, 'empty_table_image') and camera == 'overhead':
            rgb = self.robot.empty_table_image(camera)
        else:
            path = self.memory.folder / f'empty_{camera}.png'
            if path.exists():
                rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        if rgb is None:
            return None
        f = self.dino(rgb)
        self._empty[camera] = dict(rgb=rgb, features=f, lab=patch_lab(rgb, f.shape[:2]))
        return self._empty[camera]

    def propose_points(self, camera: str = 'overhead', detail: str | None = None) -> Result:
        """Numbered points on things in a fresh image, to choose from for `learn`. `detail`: a region's letter from
        the previous call: dense points over that region alone, to mark a part precisely (a handle's very end)."""
        if detail:
            if camera not in self.proposals:
                return Result(False, 'Call propose_points without detail first.')
            obs, f, P0, _ = self.proposals[camera]
            try:
                from perception.candidates import detail as dense
                P = dense(P0, detail)
            except KeyError:
                return Result(False, f'No region {detail} in the last points image.')
            self.world.step += 1
            self.proposals[camera] = (obs, f, P, self.world.step)
            img = draw_points(obs.rgb, P)
            self._emit('points', camera=camera, count=len(P.candidates), image=img, groups=f'region {detail}, dense')
            return Result(True, f'{len(P.candidates)} points over region {detail}: any of them chosen as the object '
                                f'means the whole region; use single points for parts.',
                          [(f'{camera}, region {detail} in detail', img)])
        obs = self._observe(camera, 'propose points')
        f = self.dino(obs.rgb)
        empty = self.empty_table(camera) if camera == 'overhead' else None
        roi = table_roi(obs.camera, self.table, obs.rgb.shape, f.shape[:2])
        P = propose(f, roi=roi, empty=empty, lab=patch_lab(obs.rgb, f.shape[:2]))
        self.world.step += 1
        self.proposals[camera] = (obs, f, P, self.world.step)
        img = draw_points(obs.rgb, P)
        self._emit('points', camera=camera, count=len(P.candidates), image=img, groups=describe(P))
        self._record(dict(event='propose', camera=camera, count=len(P.candidates), step=self.world.step))
        seen = self.memory.listing()
        if not P.candidates:
            return Result(True, f'No distinct things on the table in the {camera} image.', [(camera, obs.rgb)])
        return Result(True, f'{len(P.candidates)} numbered points on things on the table ({camera} image). Points in '
                            f'one region are on one thing; larger dots mark a whole region, smaller ones its parts.\n'
                            f'{describe(P)}\nTo learn an object: learn(name, points on it, and optionally named '
                            f'parts such as handle or corner, each one point). Objects already learned:\n{seen}',
                      [(f'{camera}, numbered points', img)])

    # ---------- learn ----------
    def _survey_poses(self, xy: np.ndarray, extent: float, max_poses: int = 3, avoid: list | None = None):
        """Arm poses from which the wrist camera sees the whole object, best first, never lower than
        `learn.survey_min_height` above the table, and with the gripper clear of everything in `avoid` (footprints
        of the other things on the table, taken as `learn.obstacle_height` tall).

        The wrist camera is rigid on the hand, so its pose for any hand pose is the hand pose times one fixed
        transform (read off once, from where the camera is now). Candidate hand poses on a grid along the bearing
        to the object (backed off toward the base, at several heights) are kept when IK reaches them and the
        object's whole extent (a cylinder `extent` wide and up to as tall) projects inside the image with a
        margin; the ones where it looks largest come first. Objects out of the arm's reach can be seen this way
        too, from nearer the base."""
        from core.camera import Camera
        z0 = float(self.table['z'])
        cam = self.robot.observe('wrist').camera
        tcp_now, R_now = self.robot.ee_pose()
        T_tcp = np.eye(4)
        T_tcp[:3, :3], T_tcp[:3, 3] = R_now, tcp_now
        T_tcp_cam = np.linalg.inv(T_tcp) @ cam.T_base_cam
        bearing = float(np.arctan2(xy[1], xy[0]))
        u = np.array([np.cos(bearing), np.sin(bearing), 0.0])
        R = gripper_rotation(bearing, self.pitch)
        a = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ring = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1) * (extent / 2 + 0.01)
        pts = np.concatenate([ring + np.array([xy[0], xy[1], z]) for z in (z0, z0 + float(np.clip(extent, 0.04,
                                                                                                    0.10)))])
        found = []
        for back in np.arange(0.0, 0.46, 0.03):
            for z in np.arange(0.30, z0 + float(self.cfg.get('learn', {}).get('survey_min_height', 0.05)) - 1e-6,
                               -0.02):
                tcp = np.array([xy[0], xy[1], 0.0]) - back * u
                tcp[2] = z
                if np.linalg.norm(tcp[:2]) < 0.15 or not self._survey_clear(tcp, R, avoid or []) or \
                        not self.robot.reachable(tcp, R):
                    continue
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = R, tcp
                c = Camera(cam.name, cam.width, cam.height, cam.K, T @ T_tcp_cam, cam.dist, cam.model)
                uv = c.project(pts)
                m = 0.03
                if not np.isfinite(uv).all() or uv[:, 0].min() < m * cam.width or uv[:, 0].max() > (1 - m) * \
                        cam.width or uv[:, 1].min() < m * cam.height or uv[:, 1].max() > (1 - m) * cam.height:
                    continue
                found.append((float(np.ptp(uv[:, 0]) * np.ptp(uv[:, 1])), tcp, R))
        # Spread the chosen views out (different distances back), the largest view first in each.
        found.sort(key=lambda f: -f[0])
        out = []
        for f in found:
            if all(np.linalg.norm(f[1] - o[0]) > 0.04 for o in out):
                out.append((f[1], f[2]))
            if len(out) >= max_poses:
                break
        return out

    def _survey_clear(self, tcp, R, avoid: list, margin: float = 0.015) -> bool:
        """The gripper's boxes at this pose keep `margin` from every obstacle column (`learn.obstacle_height`)."""
        from core.geometry import convex_hull, polygon_gap
        top = float(self.table['z']) + float(self.cfg.get('learn', {}).get('obstacle_height', 0.12))
        for x0, x1, y0, y1, z0_, z1 in self.gripper_boxes:
            c = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0_, z1)]) @ R.T + tcp
            if c[:, 2].min() >= top + margin:
                continue
            hull = convex_hull(c[:, :2])
            if any(polygon_gap(hull, o) < margin for o in avoid):
                return False
        return True

    def learn(self, name: str, points: list[int], parts: dict[str, int] | None = None, deformable: bool = False,
              camera: str = 'overhead') -> Result:
        """Learn the object under the chosen points of the latest `propose_points` image."""
        if camera not in self.proposals:
            return Result(False, 'Call propose_points first: learn takes points from its numbered image.')
        obs, f, P, step = self.proposals[camera]
        if self.world.step - step > 2:
            return Result(False, 'Those numbered points are out of date: call propose_points again.')
        parts = parts or {}
        bad = [n for n in list(points) + list(parts.values()) if P.by_number(int(n)) is None]
        if bad or not points:
            return Result(False, f'No point(s) {bad} in the last image (points 1-{len(P.candidates)}).'
                          if bad else 'Give at least one point on the object.')
        chosen = [int(n) for n in points]
        empty = self.empty_table(camera) if camera == 'overhead' else None
        region = np.zeros(f.shape[:2], bool)
        for n in chosen:
            region |= P.by_number(n).patches
        pix = object_pixels(obs.rgb, empty['rgb'] if empty else None, region)
        fg, bg, _ = examples(f, P, chosen, pixels=pix)
        from perception.dino_local import lab_image
        if not pix.any():
            return Result(False, 'Those points cover no pixels that differ from the bare table.')
        colour = lab_image(obs.rgb)[pix].mean(axis=0)
        from perception.memory import name_words
        words = sorted(set(name_words(name)))
        old = self.memory.entries.get(Entry(name, [], []).slug)
        e = old or Entry(name, words, colour.tolist())
        from perception.dino_local import palette_of
        e.colour_lab, e.words = colour.tolist(), sorted(set(e.words) | set(words))
        e.colours = palette_of(lab_image(obs.rgb)[pix])
        e.fg[camera], e.bg[camera] = fg, bg
        thumb = _crop(obs.rgb, pix)
        self.memory.save(e, thumb)
        self._emit('learn_mask', name=name, image=_overlay(obs.rgb, pix, (0, 255, 0)))

        # Where it is, roughly: the overhead footprint at a guessed height (from 85 cm, a 3 cm height error moves
        # an object 20 cm off-centre by about 7 mm).
        z0 = float(self.table['z'])
        guess = footprint(pix, obs.camera, z0 + 0.03, z0)
        if guess is None:
            return Result(False, 'Learned its appearance, but could not place it on the table plane.')
        msg = [f'Learned the appearance of "{name}" ({len(fg)} examples from {int(pix.sum())} pixels).']
        images = [(f'{camera}: what I learned as {name}', _overlay(obs.rgb, pix, (0, 255, 0)))]

        # Height from the overhead and the wrist view together, from the first pose that shows all of it. A soft,
        # flat thing lying on the table (cloth) has no height worth measuring: a few millimetres, outline from above.
        height, gap, wrist = None, None, None
        if deformable:
            height, gap = float(self.cfg.get('learn', {}).get('cloth_thickness', 0.004)), 0.0
        # Up to three views; the median height, so one bad view (a mask that caught a shadow or missed a side)
        # does not decide it.
        views = []
        others = obstacles(obs.rgb, empty['rgb'] if empty else None, pix, obs.camera, z0)
        for tcp, R in ([] if deformable else self._survey_poses(guess.center, max(guess.size), max_poses=5,
                                                                avoid=others)):
            if self._travel(tcp, R, f'look at {name}'):
                continue
            wobs = self._observe('wrist', f'learn {name}: view {len(views) + 2}')
            inst = self._nearest(self.segmenter_for_learning().segment(wobs, name), wobs, guess.center, z0)
            if inst is None or _touches_edge(inst.mask):
                continue
            h, g, _ = measure_height(pix, obs.camera, inst.mask, wobs.camera, z0)
            if h is not None and g < float(self.cfg.get('learn', {}).get('max_view_gap', 0.005)):
                views.append((h, g, (wobs, inst.mask)))
            if len(views) >= 3:
                break
        if views:
            views.sort(key=lambda v: v[0])
            height, gap, wrist = views[len(views) // 2]
            spread = views[-1][0] - views[0][0]
            self._record(dict(event='learn_views', name=name, heights_mm=[round(v[0] * 1000, 1) for v in views],
                              spread_mm=round(spread * 1000, 1)))
        if height is None:
            self.home()
            return Result(True, ' '.join(msg) + ' The wrist camera could not find it again, so its height and shape '
                                                'are not measured: find works, pick does not yet. Try learn again '
                                                'from a clearer view.', images)
        # Shape at that height: from the wrist view (closer) when it sees the whole object, else from above.
        src_obs, src_mask = wrist if wrist is not None else (obs, pix)
        fp = footprint(src_mask, src_obs.camera, z0 + height, z0)
        if fp is None or _touches_edge(src_mask):
            src_obs, src_mask = obs, pix
            fp = footprint(pix, obs.camera, z0 + height, z0)
        if fp is None or min(fp.size) < 0.003:
            self.home()
            return Result(True, ' '.join(msg) + ' The two views did not give a consistent shape (it measured '
                                                'under 3 mm across), so its shape is not stored: find works, pick '
                                                'does not yet.', images)
        outline = top_outline(src_mask, src_obs.camera, z0 + height, z0)
        info = shape_of(outline, height, fp.center, fp.yaw) if outline is not None else dict(
            kind='box', symmetry_deg=90.0, footprint=fp.size, outline=[])
        e.shape = Shape(float(height), tuple(info['footprint']), info['kind'], float(info['symmetry_deg']),
                        info['outline'], bool(deformable))
        # Wrist examples: its pixels' patches, and the table around them.
        if wrist is not None:
            wobs, wmask = wrist
            wf = self.dino(wobs.rgb)
            gh, gw = wf.shape[:2]
            cover = cv2.resize(wmask[:gh * PATCH, :gw * PATCH].astype(np.float32), (gw, gh),
                               interpolation=cv2.INTER_AREA)
            near = cv2.dilate((cover > 0.02).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            tablelike = (wf.reshape(-1, wf.shape[-1]) @ e.bg[camera].astype(np.float32).T).max(1).reshape(gh, gw)
            far = ~near & (tablelike > 0.7)
            rng = np.random.default_rng(0)
            pick = lambda m, k: wf[tuple(np.argwhere(m)[rng.choice(int(m.sum()), min(k, int(m.sum())),
                                                                   replace=False)].T)]
            if (cover >= 0.4).sum() >= 2:
                e.fg['wrist'] = pick(cover >= 0.4, 48)
            if far.sum() >= 4:
                e.bg['wrist'] = pick(far, 96)
        # Parts: each chosen point on the top face, in the object's frame.
        for pname, n in parts.items():
            c = P.by_number(int(n))
            pt = pixels_to_plane(obs.camera, np.array([c.uv]), z0 + height)[0]
            local = rot_z(-fp.yaw) @ (pt - fp.center)
            e.parts[pname] = Part(pname, [round(float(v), 4) for v in local], f[c.patch].astype(float).tolist())
        self.memory.save(e, thumb)
        self._record(dict(event='learn', name=name, height=height, gap_mm=None if gap is None else gap * 1000,
                          shape=info['kind'], footprint=list(info['footprint']), parts=list(parts),
                          examples={k: len(v) for k, v in e.fg.items()}))
        self._emit('learned', name=name, summary=e.summary(), image=thumb,
                   wrist=None if wrist is None else _overlay(wrist[0].rgb, wrist[1], (0, 255, 0)))
        if wrist is not None:
            images.append((f'wrist: {name} seen again', _overlay(wrist[0].rgb, wrist[1], (0, 255, 0))))
        self.home()
        if deformable:
            msg.append(f'Soft and flat: {info["footprint"][0] * 1000:.0f} x {info["footprint"][1] * 1000:.0f} mm '
                       f'from above.')
        else:
            msg.append(f'Measured from two views: {info["kind"]}, {info["footprint"][0] * 1000:.0f} x '
                       f'{info["footprint"][1] * 1000:.0f} mm, {height * 1000:.0f} mm tall (the views agree within '
                       f'{gap * 1000:.1f} mm at that height).')
        if parts:
            msg.append('Parts: ' + ', '.join(f'{k} at ({v.offset[0] * 1000:.0f}, {v.offset[1] * 1000:.0f}) mm from '
                                             f'its centre' for k, v in e.parts.items()) + '.')
        msg.append(f'Now find("{name}") finds it.')
        return Result(True, ' '.join(msg), images)

    def segmenter_for_learning(self):
        """The segmenter that knows learned objects (dino_local), whatever `find` uses."""
        if getattr(self.segmenter, 'name', '') == 'dino_local':
            return self.segmenter
        if not hasattr(self, '_dino_seg'):
            from perception.dino_local import DinoLocalSegmenter
            self._dino_seg = DinoLocalSegmenter(self.memory, features=self.dino)
        return self._dino_seg

    def _nearest(self, instances, obs, xy, z0):
        best = None
        for inst in instances:
            fp = footprint(inst.mask, obs.camera, z0 + 0.03, z0)
            if fp is None:
                continue
            d = float(np.linalg.norm(fp.center[:2] - xy[:2]))
            if d < 0.08 and (best is None or d < best[0]):
                best = (d, inst)
        return None if best is None else best[1]

    def remember_grasp(self, obj, yaw_g: float, part, ok: bool, width: float) -> None:
        """Add a grasp's outcome to a learned object's experience."""
        if not getattr(obj.type, 'learned', False):
            return
        e = next((x for x in self.memory.lookup(obj.type.name)), None)
        if e is None:
            return
        e.grasps.append(dict(yaw_rel_deg=round(float(np.rad2deg(yaw_g - obj.yaw)), 1), part=part, ok=bool(ok),
                             width=round(float(width), 4), time=time.time()))
        self.memory.save(e)

    def known_objects(self) -> Result:
        """The seen-objects list."""
        return Result(True, 'Objects I have learned:\n' + self.memory.listing())


def _touches_edge(mask: np.ndarray, margin: int = 3) -> bool:
    return bool(mask[:margin].any() or mask[-margin:].any() or mask[:, :margin].any() or mask[:, -margin:].any())


def _overlay(rgb: np.ndarray, mask: np.ndarray, colour) -> np.ndarray:
    img = rgb.copy()
    tint = np.zeros_like(img)
    tint[:] = colour
    img[mask] = (0.55 * img[mask] + 0.45 * tint[mask]).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, colour, 2, cv2.LINE_AA)
    return img


def _crop(rgb: np.ndarray, mask: np.ndarray, pad: int = 24) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return rgb
    y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad)
    return cv2.resize(rgb[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
