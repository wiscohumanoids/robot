"""The tools' bodies: find, pick, place, look, home. Geometry and IK only; no LLM inside.

pick(n):  pre-grasp from the estimate (rectangle yaw, pitched approach, `pregrasp_height` above the grasp point),
          travel there (up to travel height, across, down), refine with the wrist camera until a correction is under
          `refine.tolerance`, descend, close, check the jaw width, lift and check again.
place_on(n) / place_at(x, y): the same against the support's top face (or a spot on the table), release, retreat.

Every step appends to `trace` (what was estimated, what was corrected, how moves ended), which tools/run_scripted.py
compares with the simulator's ground truth and the agent's run log keeps. `on_event(kind, data)`, when set, also
hears each step as it starts and ends (tools/viewer.py shows them live, and may pause there).
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
        self.level_margin = float(config['find'].get('level_margin', 0.03))
        self.max_levels = int(config.get('max_stack_levels', 3))
        self.held: Held | None = None
        self.trace: list[dict] = []
        self.on_event = None          # (kind, data) -> None: every step as it happens (tools/viewer.py)
        # Wrist corrections of overhead estimates: each one measures the overhead camera's calibration error where
        # it looked. Their median is added to overhead estimates (see `overhead_correction`).
        self.overhead_deltas: list[np.ndarray] = []

    # ---------- events ----------
    def _emit(self, kind: str, **data) -> None:
        if self.on_event is not None:
            self.on_event(kind, data)

    def _record(self, entry: dict) -> None:
        """Append to the trace and tell the listener."""
        self.trace.append(entry)
        self._emit(entry['event'], **{k: v for k, v in entry.items() if k != 'event'})

    def _observe(self, camera: str, why: str) -> Observation:
        obs = self.robot.observe(camera)
        self._emit('observe', camera=camera, why=why, image=obs.rgb)
        return obs

    def _segment(self, obs: Observation, query: str):
        instances = self.segmenter.segment(obs, query)
        self._emit('segment', query=query, camera=obs.camera.name, segmenter=getattr(self.segmenter, 'name', '?'),
                   timing_ms=getattr(self.segmenter, 'last_timing', None),
                   instances=[dict(label=i.label, score=round(float(i.score), 3), pixels=int(i.mask.sum()))
                              for i in instances])
        return instances

    def _move(self, position: np.ndarray, rotation: np.ndarray, what: str):
        """One straight-line move, announced before and recorded after."""
        p = np.asarray(position, float)
        self._emit('move_start', what=what, target=p.tolist())
        r = self.robot.move_linear(p, rotation)
        self._record(dict(event='move', what=what, target=p.tolist(), ok=r.ok, error=r.position_error,
                          reason=r.reason))
        return r

    def _grip(self, opening: float, what: str) -> float:
        """Open (1) or close (0) the jaws; returns the jaw width after."""
        self._emit('gripper_start', what=what, opening=opening)
        self.robot.set_gripper(opening)
        w = self.robot.gripper_width()
        self._record(dict(event='gripper', what=what, opening=opening, width=w))
        return w

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
        the lowest level whose size error is within `find.level_margin` of the best one. Returns (footprint, level,
        trusted): trusted when the size is within `tolerance` (relative) of the expected one.

        The level comes from the apparent size: one level higher, the same mask measures smaller. From the overhead
        camera the difference is only about 6 %, and a segmenter's mask edges can be a few percent off (SAM 3's run
        about 4 % large), so a higher level must fit clearly better. From the wrist camera, a few centimetres away,
        one level changes the size by tens of percent and the choice is clear.

        That guess assumes the mask is one block: its top face and its own sides. A segmenter's mask of a tower
        covers the sides of every block below as well, which makes it look a level or more taller, so a level the
        agent knows (a stack it built) is passed in instead. With a known level above the table, the mask is also
        measured as the whole tower down to the table, and the better fit is kept: one block's mask (MuJoCo's)
        fits the first, a tower mask the second (about 0.3 mm from the true centre instead of 2 to 5 mm).
        """
        z0 = float(self.table['z'])
        fits = []
        for n in ([level] if level else range(1, self.max_levels + 1)):
            z = z0 + n * t.height
            for base in ({z - t.height, z0} if level and n > 1 else {z - t.height}):
                fp = footprint(mask, obs.camera, z, base)
                if fp is not None:
                    fits.append((fp, n, max(abs(m - e) / e for m, e in zip(fp.size, t.footprint))))
        if level and fits:
            fits = [min(fits, key=lambda f: f[2])]
        if not fits:
            return None, None, False
        least = min(f[2] for f in fits)
        fp, n, err = next(f for f in fits if f[2] <= least + self.level_margin)
        return fp, n, err <= (self.find_size_tol if tolerance is None else tolerance)

    @staticmethod
    def _size_error(fp, t: ObjectType) -> float:
        return max(abs(m - e) / e for m, e in zip(fp.size, t.footprint))

    def _use_built_stacks(self, dets: list, obs: Observation, t: ObjectType, corr: np.ndarray) -> None:
        """Give the detections on a stack the agent built that stack's known height (see `World.stacks`).

        Several detections can sit on one stack: its top block, and (with per-object masks) slivers of the blocks
        under it. The one that fits the known height best is the top; the others are the blocks under it, partly
        hidden. A tower mask can only look taller than the tower, never shorter, so a top that clearly fits a lower
        level better means the top block is gone (fell, or was knocked off): then the size wins and the stack is
        remembered lower.
        """
        for built in list(self.world.stacks):
            near = [d for d in dets if np.linalg.norm(d[1].center[:2] + corr[:2] - built.xy) < 0.025]
            fits = [(self._measure(d[0].mask, obs, t, level=built.levels), d) for d in near]
            fits = [(m, d) for m, d in fits if m[0] is not None]
            if not fits:
                continue
            (fp, level, trusted), top = min(fits, key=lambda f: self._size_error(f[0][0], t))
            if (top[2] < min(built.levels, self.max_levels) and top[3]
                    and self._size_error(top[1], t) + self.level_margin < self._size_error(fp, t)):
                top[5] = f' (I stacked {built.levels} high here, but it measures {top[2]} high: did the top block fall?)'
                self.world.set_stack(built.xy, top[2])
                continue
            top[1:5] = [fp, level, trusted, True]
            for d in near:
                if d is not top:
                    d[2], d[3] = min(d[2], built.levels - 1), False

    def look(self, camera: str = 'overhead') -> Result:
        """A fresh image (the overhead one with the table grid in centimetres) and the gripper's state."""
        obs = self._observe(camera, 'look')
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
        obs = self._observe(camera, f'find "{query}"')
        corr = self.overhead_correction() if camera == 'overhead' else np.zeros(3)
        dets = []                                          # [instance, footprint, level, trusted, known, note]
        for inst in self._segment(obs, query):
            fp, level, trusted = self._measure(inst.mask, obs, t)
            if fp is not None:
                dets.append([inst, fp, level, trusted, False, ''])
        self._use_built_stacks(dets, obs, t, corr)
        # The same object seen again, or an old estimate now under a newly seen top: the new number replaces it. (An
        # old estimate above a new one stays: "green cube" under a known red one does not erase red.)
        for old in [o for o in self.world.objects.values() if any(
                np.linalg.norm(o.top[:2] - (d[1].center[:2] + corr[:2])) < 0.02
                and o.top[2] < d[1].center[2] + corr[2] + 0.02 for d in dets)]:
            del self.world.objects[old.number]
        found = []
        for inst, fp, level, trusted, known, note in dets:
            f = self.world.add(query=query, label=inst.label, type=t, top=fp.center + corr, yaw=fp.yaw, size=fp.size,
                               score=inst.score, camera=camera, trusted=trusted, level=level, raw_top=fp.center,
                               level_known=known)
            found.append((f, inst.mask, note))
        self._record(dict(event='find', query=query, camera=camera, step=self.world.step,
                               found=[dict(number=f.number, label=f.label, top=f.top.tolist(), yaw=f.yaw,
                                           size=list(f.size), level=f.level, level_known=f.level_known,
                                           trusted=f.trusted) for f, _, _ in found]))
        if not found:
            return Result(True, f'No "{query}" found in the {camera} image.', [(camera, obs.rgb)])
        img = draw_instances(obs.rgb, [(f.number, m) for f, m, _ in found])
        lines = [f.describe() + (f', stacked {f.level} high' + (' (a stack I built)' if f.level_known else
                                                                ' (judged from its size)') if f.level > 1 else '')
                 + note for f, _, note in found]
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
            r = self._move(p, R, f'{what}: {name}')
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
        blocked, reachable_any, tried = None, False, []
        for yaw in yaws:
            R = gripper_rotation(yaw, self.pitch)
            tcp = tcp_of(R)
            option = dict(yaw_deg=round(float(np.rad2deg(yaw)), 1), tcp=tcp.tolist(), reachable=False)
            tried.append(option)
            for h in [self.pre_h] + self.lower_pre:
                pre = tcp + np.array([0.0, 0.0, h])
                if not (self.robot.reachable(tcp, R) and self.robot.reachable(pre, R)):
                    continue
                reachable_any = True
                gap, who = self._clearance(R, tcp, ignore)
                option.update(reachable=True, pregrasp_height=h, clearance_mm=round(float(min(gap, 1.0)) * 1000, 1),
                              nearest=None if who is None else who.number)
                if gap >= self.clearance:
                    self._emit('plan', what=what, options=tried, chosen=option)
                    return yaw, R, tcp, pre
                if blocked is None or gap > blocked[0]:
                    blocked = (gap, who)
                break
        self._emit('plan', what=what, options=tried, chosen=None)
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
        block hide part of it), or it jumps further than `max_correction` (the wrist saw a different object). The
        wrist view also settles the stack level, which the overhead camera can only weakly tell apart.
        """
        info = dict(iterations=0, corrections_mm=[], note='')
        for _ in range(self.refine_iters):
            obs = self._observe('wrist', f'refine {what}')
            best = None
            for inst in self._segment(obs, obj.query):
                fp, level, trusted = self._measure(inst.mask, obs, obj.type, tolerance=self.refine_size_tol,
                                                   level=obj.level if obj.level_known else None)
                if fp is None:
                    continue
                d = float(np.linalg.norm(fp.center[:2] - obj.top[:2]))
                if best is None or d < best[1]:
                    best = (fp, d, trusted, level)
            if best is None:
                info['note'] = 'not seen by the wrist camera; kept the overhead estimate'
                break
            fp, d, trusted, level = best
            if not trusted:
                info['note'] = 'partly hidden in the wrist view; kept the previous estimate'
                break
            if d > self.max_correction:
                info['note'] = f'wrist estimate {d * 1000:.0f} mm away (another object?); kept the previous one'
                break
            info['iterations'] += 1
            info['corrections_mm'].append(round(d * 1000, 2))
            if level != obj.level:
                info['note'] = f'the wrist view puts it {level} high, not {obj.level}'
                obj.level = level
            obj.top = fp.center.copy()                     # its z is the (possibly corrected) level's top face
            obj.yaw = fp.yaw
            self._record(dict(event='refine', what=what, number=obj.number, top=obj.top.tolist(), yaw=obj.yaw,
                                   correction_mm=round(d * 1000, 2)))
            if d < self.refine_tol:
                break
            err = move_to(obj)
            if err is not None:
                info['note'] = err.message
                break
        if info['note']:
            self._emit('refine_note', what=what, note=info['note'])
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
        self._grip(1.0, 'open before the approach')
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
            r = self._move(pre, R, f'corrected pre-grasp over #{number}')
            return None if r.ok else Result(False, f'Could not reach the corrected pre-grasp: {r.reason}.')

        refine = self._refine(obj, f'pick #{number}', move_pre)
        replan(obj)
        r = self._move(grasp, R, f'descend to #{number}')
        if not r.ok:
            return Result(False, f'Could not descend to #{number}: {r.reason}.', details=dict(refine=refine))
        width = self._grip(0.0, f'close on #{number}')
        if width < self.min_width:
            self._grip(1.0, 'open (closed on nothing)')
            self._move(pre, R, f'back up from #{number}')
            self._record(dict(event='grasp', number=number, ok=False, width=width))
            return Result(False, f'Closed on nothing (jaws {width * 1000:.0f} mm apart); #{number} may have moved. '
                                 'Find it again.', details=dict(refine=refine))
        self._move(pre, R, f'lift #{number}')
        lifted = self.robot.gripper_width()
        self._record(dict(event='grasp', number=number, ok=lifted >= self.min_width, width=width,
                               width_after_lift=lifted, yaw=yaw_g))
        if lifted < self.min_width:
            return Result(False, f'Grasped #{number} but dropped it while lifting.', details=dict(refine=refine))
        self.held = Held(obj, R.T @ (grasp - obj.center), obj.yaw - yaw_g)
        self.world.objects.pop(obj.number, None)
        if self.world.stack_at(obj.top[:2]) is not None:      # a stack it built is one lower now
            self.world.set_stack(obj.top[:2], obj.level - 1)
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
                r = self._move(pre, R, f'corrected pre-place {what}')
                return None if r.ok else Result(False, f'Could not reach the corrected pre-place: {r.reason}.')
            refine = self._refine(support, what, move_pre)
        r = self._move(tcp, R, f'lower {what}')
        if not r.ok:
            return Result(False, f'Could not lower onto {what}: {r.reason}.', details=dict(refine=refine))
        self._grip(1.0, f'release {what}')
        self._move(pre, R, f'retreat from {what}')
        number = held.obj.number
        self.held = None
        # Remember the stack's height where the block went (on the table: forget any stack remembered there).
        self.world.set_stack(tcp[:2] - (R @ held.offset)[:2], support.level + 1 if support is not None else 1)
        self._record(dict(event='place', what=what, number=number, tcp=tcp.tolist(), yaw=yaw_g,
                          stack_levels=support.level + 1 if support is not None else 1))
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
        self._grip(1.0, 'open')
        if self.held:
            self.held = None
        return Result(True, 'Jaws open. ' + self.gripper_state())

    def home(self) -> Result:
        """Back to the start pose, out of the overhead camera's way."""
        if self.held:
            return Result(False, f'Holding #{self.held.obj.number}; place it before going home.')
        self._emit('move_start', what='home', target=None)
        r = self.robot.home()
        self._record(dict(event='move', what='home', target=None, ok=r.ok, error=r.position_error, reason=r.reason))
        return Result(r.ok, 'At home.' if r.ok else f'Could not go home: {r.reason}.')

