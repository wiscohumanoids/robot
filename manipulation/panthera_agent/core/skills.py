"""The tools' bodies: find, pick, place, look, home. Geometry and IK only; no LLM inside.

pick(n):  pre-grasp from the estimate (rectangle yaw, pitched approach, `pregrasp_height` above the grasp point),
          travel there (up to travel height, across, down), refine with the wrist camera until a correction is under
          `refine.tolerance`, descend, close, check the jaw width, lift and check again.
place_on(n) / place_at(x, y): the same against the support's top face (or a spot on the table), release, retreat.

Every step appends to `trace` (what was estimated, what was corrected, how moves ended), which tools/run_scripted.py
compares with the simulator's ground truth and the agent's run log keeps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.geometry import convex_hull, fold_yaw, footprint, gripper_rotation, polygon_gap, rectangle
from core.world import Found, ObjectType, World
from perception.base import Segmenter
from perception.overlay import draw_instances, draw_table_grid
from robots.base import Observation, Robot


@dataclass
class Result:
    """What a skill reports: whether it worked, a message for the LLM, images, and details for the log."""
    ok: bool
    message: str
    images: list[tuple[str, np.ndarray]] = field(default_factory=list)      # (caption, rgb)
    details: dict = field(default_factory=dict)


@dataclass
class Held:
    """The object in the jaws, and where it sits relative to them."""
    obj: Found
    offset: np.ndarray            # TCP - object centre, in the gripper frame
    yaw_offset: float             # object yaw - gripper yaw at the grasp


class Skills:
    """The manipulation skills over a robot, a segmenter and the agent's world."""

    def __init__(self, robot: Robot, segmenter: Segmenter, world: World, config: dict):
        self.robot, self.segmenter, self.world, self.cfg = robot, segmenter, world, config
        self.table = config['table']
        g, p, r = config['grasp'], config['place'], config['refine']
        self.pitch = np.deg2rad(g['pitch'])
        self.tcp_offset, self.pre_h, self.travel_z = float(g['tcp_offset']), float(g['pregrasp_height']), \
            float(self.table['z']) + float(g['travel_height'])
        self.min_width = float(g['min_width'])
        self.lower_pre = [float(h) for h in g.get('lower_pregrasp', [])]
        self.clearance = float(g['clearance'])
        self.gripper_boxes = [np.array(b, float) for b in g['gripper_boxes']]
        self.gap = float(p['gap'])
        self.refine_iters, self.refine_tol = int(r['max_iterations']), float(r['tolerance'])
        self.max_correction, self.refine_size_tol = float(r['max_correction']), float(r['size_tolerance'])
        self.find_size_tol = float(config['find']['size_tolerance'])
        self.max_levels = int(config.get('max_stack_levels', 3))
        self.held: Held | None = None
        self.trace: list[dict] = []
        # Wrist corrections of overhead estimates: each one measures the overhead camera's calibration error where
        # it looked. Their median is added to overhead estimates (see `overhead_correction`).
        self.overhead_deltas: list[np.ndarray] = []

    # ---------- perception ----------
    def overhead_correction(self) -> np.ndarray:
        """(dx, dy, 0) to add to overhead estimates: the median of the wrist camera's corrections so far.

        A small error in the overhead camera's calibrated pose shifts everything it measures by nearly the same
        amount (1 degree of tilt from 0.85 m is about 15 mm), and the wrist camera, a few centimetres from the
        object, measures that shift whenever it refines a pick. The place target cannot be refined that way (the
        held block hides it from the wrist camera), so this correction is what keeps a place accurate.
        """
        if not self.overhead_deltas:
            return np.zeros(3)
        d = np.median(np.array(self.overhead_deltas[-5:]), axis=0)
        return np.array([d[0], d[1], 0.0])

    def _current(self, obj: Found) -> None:
        """Apply the latest overhead correction to an overhead estimate the wrist has not refined."""
        if obj.camera == 'overhead' and not obj.refined and obj.raw_top is not None:
            obj.top = obj.raw_top + self.overhead_correction()

    def _measure(self, mask: np.ndarray, obs: Observation, t: ObjectType, level: int | None = None,
                 tolerance: float | None = None):
        """Footprint of a mask for an object of type `t` stacked `level` high (1 = on the table); with level None,
        the level whose expected size the measurement matches best. Returns (footprint, level, trusted): trusted
        when the size is within `tolerance` (relative) of the expected one."""
        z0 = float(self.table['z'])
        best = None
        for n in ([level] if level else range(1, self.max_levels + 1)):
            fp = footprint(mask, obs.camera, z0 + n * t.height, z0 + (n - 1) * t.height)
            if fp is None:
                continue
            err = max(abs(m - e) / e for m, e in zip(fp.size, t.footprint))
            if best is None or err < best[2]:
                best = (fp, n, err)
        if best is None:
            return None, None, False
        return best[0], best[1], best[2] <= (self.find_size_tol if tolerance is None else tolerance)

    def look(self, camera: str = 'overhead') -> Result:
        """A fresh image (the overhead one with the table grid in centimetres) and the gripper's state."""
        obs = self.robot.observe(camera)
        img = obs.rgb
        if camera == 'overhead':
            img = draw_table_grid(img, obs.camera, float(self.table['z']), self.table['x'], self.table['y'])
        return Result(True, f'{camera} image. {self.gripper_state()}', [(camera, img)])

    def gripper_state(self) -> str:
        """The jaws and what they hold, in words."""
        w = self.robot.gripper_width()
        pos, _ = self.robot.ee_pose()
        held = f'holding #{self.held.obj.number} ({self.held.obj.label})' if self.held else 'holding nothing'
        return (f'Gripper {held}, jaws {w * 1000:.0f} mm apart, at x={pos[0] * 100:.1f} y={pos[1] * 100:.1f} '
                f'z={pos[2] * 100:.1f} cm.')

    def find(self, query: str, camera: str = 'overhead') -> Result:
        """Every object matching `query`, numbered, with its top face's position, size and yaw."""
        t = self.world.type_for(query)
        if t is None:
            known = ', '.join(sorted(w for ty in self.world.types.values() for w in ty.words))
            return Result(False, f'"{query}" names no object type I can measure (known words: {known}).')
        self.world.step += 1
        obs = self.robot.observe(camera)
        found = []
        for inst in self.segmenter.segment(obs, query):
            fp, level, trusted = self._measure(inst.mask, obs, t)
            if fp is None:
                continue
            top = fp.center + (self.overhead_correction() if camera == 'overhead' else 0.0)
            for old in [o for o in self.world.objects.values() if np.linalg.norm(o.top - top) < 0.02]:
                del self.world.objects[old.number]          # the same object, seen again: the new number replaces it
            f = self.world.add(query=query, label=inst.label, type=t, top=top, yaw=fp.yaw, size=fp.size,
                               score=inst.score, camera=camera, trusted=trusted, level=level, raw_top=fp.center)
            found.append((f, inst.mask))
        self.trace.append(dict(event='find', query=query, camera=camera, step=self.world.step,
                               found=[dict(number=f.number, label=f.label, top=f.top.tolist(), yaw=f.yaw,
                                           size=list(f.size), level=f.level, trusted=f.trusted) for f, _ in found]))
        if not found:
            return Result(True, f'No "{query}" found in the {camera} image.', [(camera, obs.rgb)])
        img = draw_instances(obs.rgb, [(f.number, m) for f, m in found])
        lines = [f.describe() + (f', stacked {f.level} high' if f.level > 1 else '') for f, _ in found]
        return Result(True, f'Found {len(found)} "{query}" (step {self.world.step}):\n' + '\n'.join(lines),
                      [(f'{camera}, numbered', img)])

    # ---------- motion ----------
    def _travel(self, position: np.ndarray, rotation: np.ndarray, what: str) -> Result | None:
        """Up to travel height, across, down to the pose. None when it got there, else the failure.

        The arm's reach shrinks with height on the pitched approach, so each waypoint is lowered when IK cannot reach
        it: the crossing to the pose's own height, the lift to the highest height reachable where the arm is.
        """
        pos, R0 = self.robot.ee_pose()
        z = max(self.travel_z, position[2])
        if not self.robot.reachable(np.array([position[0], position[1], z]), rotation):
            z = position[2]
        legs = []
        if pos[2] < z - 1e-3:
            up = z
            while up > pos[2] + 1e-3 and not self.robot.reachable(np.array([pos[0], pos[1], up]), R0):
                up -= 0.01
            if up > pos[2] + 1e-3:
                legs.append(('up', np.array([pos[0], pos[1], up]), R0))
        legs += [('across', np.array([position[0], position[1], z]), rotation), ('down', position, rotation)]
        for name, p, R in legs:
            r = self.robot.move_linear(p, R)
            self.trace.append(dict(event='move', what=f'{what}: {name}', target=p.tolist(), ok=r.ok,
                                   error=r.position_error))
            if not r.ok:
                return Result(False, f'Could not reach {what} ({name}): {r.reason}.')
        return None

    def _yaws(self, yaw: float, symmetry: float, at: np.ndarray) -> list[float]:
        """The equivalent gripper yaws (yaw + k * symmetry) within 100 deg of the radial bearing to `at`, the most
        radial first."""
        bearing = float(np.arctan2(at[1], at[0]))
        out = []
        for k in range(int(round(2 * np.pi / symmetry))):
            y = fold_yaw(yaw + k * symmetry, 2 * np.pi, bearing)
            if abs(y - bearing) <= np.deg2rad(100) and all(abs(y - o) > 1e-6 for o in out):
                out.append(y)
        return sorted(out, key=lambda y: abs(y - bearing))

    def _clearance(self, R: np.ndarray, tcp: np.ndarray, ignore: set[int]) -> tuple[float, Found | None]:
        """Smallest horizontal gap (m) between the open gripper at TCP pose (R, tcp) and any other known object,
        and that object.

        The gripper is `grasp.gripper_boxes`; an object is a column from the table up to its top face. A box whose
        lowest corner clears an object's top (by `clearance`) cannot touch it. The gripper reaches this pose by
        descending straight down, so every earlier position of a box is higher: checking the final pose covers the
        descent.
        """
        corners = []
        for x0, x1, y0, y1, z0, z1 in self.gripper_boxes:
            c = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
            corners.append(c @ R.T + tcp)
        best, who = np.inf, None
        for o in self.world.objects.values():
            if o.number in ignore:
                continue
            rect = rectangle(o.top[:2], o.type.footprint, o.yaw)
            for c in corners:
                if c[:, 2].min() >= o.top[2] + self.clearance:
                    continue
                gap = polygon_gap(convex_hull(c[:, :2]), rect)
                if gap < best:
                    best, who = gap, o
        return best, who

    def _choose(self, yaws: list[float], tcp_of, ignore: set[int], what: str):
        """First (yaw, pre-height) whose TCP and pre-pose are reachable and whose gripper footprint clears the other
        objects. Returns (yaw, R, tcp, pre) or a failing Result naming what is in the way."""
        blocked, reachable_any = None, False
        for yaw in yaws:
            R = gripper_rotation(yaw, self.pitch)
            tcp = tcp_of(R)
            for h in [self.pre_h] + self.lower_pre:
                pre = tcp + np.array([0.0, 0.0, h])
                if not (self.robot.reachable(tcp, R) and self.robot.reachable(pre, R)):
                    continue
                reachable_any = True
                gap, who = self._clearance(R, tcp, ignore)
                if gap >= self.clearance:
                    return yaw, R, tcp, pre
                if blocked is None or gap > blocked[0]:
                    blocked = (gap, who)
                break
        if blocked:
            return Result(False, f'{what}: the open gripper would hit #{blocked[1].number} ({blocked[1].label}, '
                                 f'{max(blocked[0], 0) * 1000:.0f} mm clearance) from every side it can reach. '
                                 f'Move #{blocked[1].number} away first.')
        if not reachable_any:
            return Result(False, f'{what}: out of the arm\'s reach.')
        return Result(False, f'{what}: no reachable approach.')

    def _refine(self, obj: Found, what: str, move_to) -> dict:
        """Re-measure `obj` from the wrist camera at the current (pre-)pose and correct its estimate until a
        correction is under tolerance. `move_to(obj)` re-plans and moves to the corrected pre-pose.

        A measurement is ignored when no mask is near the estimate, its size does not match (the jaws or a held
        block hide part of it), or it jumps further than `max_correction` (the wrist saw a different object).
        """
        info = dict(iterations=0, corrections_mm=[], note='')
        for _ in range(self.refine_iters):
            obs = self.robot.observe('wrist')
            best = None
            for inst in self.segmenter.segment(obs, obj.query):
                fp, _, trusted = self._measure(inst.mask, obs, obj.type, level=obj.level,
                                               tolerance=self.refine_size_tol)
                if fp is None:
                    continue
                d = float(np.linalg.norm(fp.center[:2] - obj.top[:2]))
                if best is None or d < best[1]:
                    best = (fp, d, trusted)
            if best is None:
                info['note'] = 'not seen by the wrist camera; kept the overhead estimate'
                break
            fp, d, trusted = best
            if not trusted:
                info['note'] = 'partly hidden in the wrist view; kept the previous estimate'
                break
            if d > self.max_correction:
                info['note'] = f'wrist estimate {d * 1000:.0f} mm away (another object?); kept the previous one'
                break
            info['iterations'] += 1
            info['corrections_mm'].append(round(d * 1000, 2))
            obj.top = np.array([fp.center[0], fp.center[1], obj.top[2]])
            obj.yaw = fp.yaw
            self.trace.append(dict(event='refine', what=what, number=obj.number, top=obj.top.tolist(), yaw=obj.yaw,
                                   correction_mm=round(d * 1000, 2)))
            if d < self.refine_tol:
                break
            err = move_to(obj)
            if err is not None:
                info['note'] = err.message
                break
        if info['iterations']:
            obj.refined = True
            if obj.camera == 'overhead' and obj.raw_top is not None:
                self.overhead_deltas.append(obj.top[:2] - obj.raw_top[:2])
        return info

    # ---------- skills ----------
    def pick(self, number: int) -> Result:
        """Pick up found object `number`."""
        if self.held:
            return Result(False, f'Already holding #{self.held.obj.number}; place it first.')
        obj = self.world.objects.get(int(number))
        if obj is None:
            return Result(False, f'No object #{number}; use find first.')
        self._current(obj)
        sym = obj.type.symmetry
        ignore = {obj.number}
        plan = self._choose(self._yaws(obj.yaw, sym, obj.top), lambda R: obj.center + self.tcp_offset * R[:, 0],
                            ignore, f'pick #{number}')
        if isinstance(plan, Result):
            return plan
        yaw_g, R, grasp, pre = plan
        h = pre[2] - grasp[2]
        self.robot.set_gripper(1.0)
        err = self._travel(pre, R, f'above #{number}')
        if err:
            return err

        def replan(o):                    # after a wrist correction: same side, same height
            nonlocal yaw_g, R, grasp, pre
            yaw_g = fold_yaw(o.yaw, sym, yaw_g)
            R = gripper_rotation(yaw_g, self.pitch)
            grasp = o.center + self.tcp_offset * R[:, 0]
            pre = grasp + np.array([0.0, 0.0, h])

        def move_pre(o):
            replan(o)
            r = self.robot.move_linear(pre, R)
            return None if r.ok else Result(False, f'Could not reach the corrected pre-grasp: {r.reason}.')

        refine = self._refine(obj, f'pick #{number}', move_pre)
        replan(obj)
        r = self.robot.move_linear(grasp, R)
        if not r.ok:
            return Result(False, f'Could not descend to #{number}: {r.reason}.', details=dict(refine=refine))
        self.robot.set_gripper(0.0)
        width = self.robot.gripper_width()
        if width < self.min_width:
            self.robot.set_gripper(1.0)
            self.robot.move_linear(pre, R)
            self.trace.append(dict(event='grasp', number=number, ok=False, width=width))
            return Result(False, f'Closed on nothing (jaws {width * 1000:.0f} mm apart); #{number} may have moved. '
                                 'Find it again.', details=dict(refine=refine))
        self.robot.move_linear(pre, R)
        lifted = self.robot.gripper_width()
        self.trace.append(dict(event='grasp', number=number, ok=lifted >= self.min_width, width=width,
                               width_after_lift=lifted, yaw=yaw_g))
        if lifted < self.min_width:
            return Result(False, f'Grasped #{number} but dropped it while lifting.', details=dict(refine=refine))
        self.held = Held(obj, R.T @ (grasp - obj.center), obj.yaw - yaw_g)
        self.world.objects.pop(obj.number, None)
        note = f' Wrist refine: {refine["corrections_mm"]} mm.' if refine['corrections_mm'] else ''
        note += f' ({refine["note"]})' if refine['note'] else ''
        return Result(True, f'Holding #{number} ({obj.label}), jaws {lifted * 1000:.0f} mm apart.{note}',
                      details=dict(refine=refine, width=lifted))

    def _place(self, top: np.ndarray, yaw: float, what: str, support: Found | None) -> Result:
        """Set the held object down with its base on `top` (a support's top face, or a spot on the table), turned to
        `yaw` (mod its symmetry), and release it."""
        held = self.held
        sym = held.obj.type.symmetry
        ignore = {held.obj.number}
        if support is not None:                     # the support and whatever it stands on
            ignore |= {o.number for o in self.world.objects.values()
                       if np.linalg.norm(o.top[:2] - support.top[:2]) < 0.015}
        center = top + np.array([0.0, 0.0, held.obj.type.height / 2 + self.gap])
        plan = self._choose(self._yaws(yaw - held.yaw_offset, sym, top), lambda R: center + R @ held.offset,
                            ignore, what)
        if isinstance(plan, Result):
            return plan
        yaw_g, R, tcp, pre = plan
        h = pre[2] - tcp[2]
        err = self._travel(pre, R, what)
        if err:
            return err
        refine = dict(iterations=0, corrections_mm=[], note='')
        if support is not None:
            def move_pre(o):
                nonlocal yaw_g, R, tcp, pre
                yaw_g = fold_yaw(o.yaw - held.yaw_offset, sym, yaw_g)
                R = gripper_rotation(yaw_g, self.pitch)
                tcp = o.top + np.array([0.0, 0.0, held.obj.type.height / 2 + self.gap]) + R @ held.offset
                pre = tcp + np.array([0.0, 0.0, h])
                r = self.robot.move_linear(pre, R)
                return None if r.ok else Result(False, f'Could not reach the corrected pre-place: {r.reason}.')
            refine = self._refine(support, what, move_pre)
        r = self.robot.move_linear(tcp, R)
        if not r.ok:
            return Result(False, f'Could not lower onto {what}: {r.reason}.', details=dict(refine=refine))
        self.robot.set_gripper(1.0)
        self.robot.move_linear(pre, R)
        number = held.obj.number
        self.held = None
        self.trace.append(dict(event='place', what=what, number=number, tcp=tcp.tolist(), yaw=yaw_g))
        note = f' Wrist refine: {refine["corrections_mm"]} mm.' if refine['corrections_mm'] else ''
        note += f' ({refine["note"]})' if refine['note'] else ''
        return Result(True, f'Released #{number} {what}.{note} Look or find to check it.', details=dict(refine=refine))

    def place_on(self, number: int) -> Result:
        """Set the held object on top of found object `number`, faces aligned."""
        if not self.held:
            return Result(False, 'Not holding anything.')
        support = self.world.objects.get(int(number))
        if support is None:
            return Result(False, f'No object #{number}; use find first.')
        self._current(support)
        return self._place(support.top.copy(), support.yaw, f'on #{number}', support)

    def place_at(self, x: float, y: float) -> Result:
        """Set the held object down on the table with its centre at (x, y) metres."""
        if not self.held:
            return Result(False, 'Not holding anything.')
        (x0, x1), (y0, y1) = self.table['x'], self.table['y']
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return Result(False, f'({x:.2f}, {y:.2f}) m is off the table (x {x0}..{x1}, y {y0}..{y1}).')
        return self._place(np.array([x, y, float(self.table['z'])]), float(np.arctan2(y, x)),
                           f'at ({x * 100:.0f}, {y * 100:.0f}) cm', None)

    def open_gripper(self) -> Result:
        """Open the jaws where the arm is (drops whatever is held)."""
        self.robot.set_gripper(1.0)
        if self.held:
            self.held = None
        return Result(True, 'Jaws open. ' + self.gripper_state())

    def home(self) -> Result:
        """Back to the start pose, out of the overhead camera's way."""
        if self.held:
            return Result(False, f'Holding #{self.held.obj.number}; place it before going home.')
        r = self.robot.home()
        return Result(r.ok, 'At home.' if r.ok else f'Could not go home: {r.reason}.')

