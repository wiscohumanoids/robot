"""Contact and tool skills (mixin for core/skills.py Skills): geometry brings the held object to just before contact;
the last centimetres are a closed-loop local skill.

insert(number):          the held object into a hole in found object #number (its learned "hole" part, else its top
                         centre). Lined up by sight from above, lowered to just above the opening, then a local
                         controller finishes: the trained policy for "insert" when there is one (learning/, run at
                         10 Hz on the wrist camera and the pose relative to the start), else a scripted search
                         (press down; when blocked, spiral out in small steps). Success: the held object went down
                         into the hole by `contact.insert_depth`.
pull_with_tool(number):  with a held tool that has a learned "hook" (or "tip") part, reach past found object #number
                         and drag it toward the arm's base, into reach.
pour_into(number):       tip the held container over found object #number and back.

Every local-controller step is reported (`finish_step`) and the whole attempt can be recorded as an episode for
training (learning/episodes.py): the policies get better from these.
"""
from __future__ import annotations

import numpy as np

from core.geometry import fold_yaw, gripper_rotation, rot_z
from core.types import Result


def _axis_angle(axis, angle):
    import cv2
    axis = np.asarray(axis, float)
    return cv2.Rodrigues(axis / np.linalg.norm(axis) * angle)[0]


class ContactSkills:
    """Expects Skills' attributes (robot, cfg, held, world, _move, _travel, _grip, _choose, _emit, _record, ...)."""

    policies: dict = {}             # skill name -> callable policy (learning/policy.py LocalPolicy.act), set by build
    recorder = None                 # learning/episodes.py EpisodeRecorder, when recording

    def _contact_cfg(self) -> dict:
        return {**dict(insert_depth=0.015, hover=0.006, step=0.002, max_steps=60, spiral_step=0.0015,
                       pull_to=0.45, clearance=0.006, pour_clearance=0.03), **(self.cfg.get('contact') or {})}

    def _held_center(self):
        tcp, R = self.robot.ee_pose()
        return tcp - R @ self.held.offset, tcp, R

    # ---------- insert ----------
    def insert_handoff(self, number: int):
        """Bring the held object to just above the hole of found object `number`: (start TCP, R, hole top centre,
        pre-pose, alignment summary), or a failing Result."""
        if not self.held:
            return Result(False, 'Not holding anything to insert.')
        target = self.world.objects.get(int(number))
        if target is None:
            return Result(False, f'No object #{number}; use find first.')
        self._current(target)
        c = self._contact_cfg()
        held = self.held
        t = held.obj.type
        hole = target.top.copy()
        if target.type.parts and 'hole' in target.type.parts:
            hole = target.top + rot_z(target.yaw) @ np.asarray(target.type.parts['hole'], float)
            hole[2] = target.top[2]
        # Above the opening: the held object's bottom `hover` above the target's top, yaw matched to the target.
        center = hole + np.array([0.0, 0.0, t.height / 2 + c['hover']])
        sym = t.symmetry if t.symmetry > 1e-6 else 2 * np.pi
        yaws = self._yaws(target.yaw - held.yaw_offset, sym, hole)
        plan = self._choose(yaws, lambda R: center + R @ held.offset, {held.obj.number, target.number},
                            f'insert into #{number}')
        if isinstance(plan, Result):
            return plan
        yaw_g, R, tcp, pre = plan
        err = self._travel(pre, R, f'above #{number}')
        if err:
            return err

        def shift(d):
            nonlocal tcp, pre
            tcp = tcp + np.array([d[0], d[1], 0.0])
            pre = pre + np.array([d[0], d[1], 0.0])
            r = self._move(pre, R, 'aligned above the hole')
            return None if r.ok else Result(False, r.reason)
        align = self.align_over(target, hole, f'over #{number}', shift) if self.checks['align'] else None
        r = self._move(tcp, R, f'down to just above #{number}')
        if not r.ok:
            return Result(False, f'Could not get down to the opening: {r.reason}.')
        start, _ = self.robot.ee_pose()
        return start, R, hole, pre, align

    def insert_depth(self, hole: np.ndarray) -> float:
        """How far the held object's bottom is below the hole's top (m), from the arm's pose."""
        return float(hole[2] - (self._held_center()[0][2] - self.held.obj.type.height / 2))

    def insert(self, number: int) -> Result:
        """Put the held object into the hole of found object `number`."""
        h = self.insert_handoff(number)
        if isinstance(h, Result):
            return h
        start, R, hole, pre, align = h
        c = self._contact_cfg()
        policy = self.policies.get('insert')
        how = 'trained policy' if policy is not None else 'scripted search'
        self._emit('finish_start', skill='insert', how=how, start=start.tolist())
        if self.recorder is not None:
            self.recorder.begin('insert', dict(number=number, how=how, target=hole.tolist()))
        if policy is not None:
            done, steps = self._run_policy(policy, start, R, hole, c)
        else:
            done, steps = self._spiral(start, R, c)
        depth = self.insert_depth(hole)
        ok = depth >= c['insert_depth'] - 1e-3
        if self.recorder is not None:
            self.recorder.end(success=ok, info=dict(depth=depth, steps=steps))
        self._record(dict(event='finish', skill='insert', how=how, ok=ok, depth_mm=round(depth * 1000, 1),
                          steps=steps))
        if not ok:
            self._move(pre, R, 'back up from the hole')
            return Result(False, f'Insert into #{number} did not go in ({how}, {steps} steps): the held object\'s '
                                 f'bottom is {depth * 1000:.0f} mm below the hole\'s top, needs '
                                 f'{c["insert_depth"] * 1000:.0f}. Still holding it.')
        self._grip(1.0, f'release into #{number}')
        tcp_now, _ = self.robot.ee_pose()
        self._move(tcp_now + np.array([0, 0, 0.06]), R, 'retreat from the hole')
        self.held = None
        note = f' Lined up by sight: moved {align["moved_mm"]} mm.' if align and align.get('moved_mm') else ''
        return Result(True, f'Inserted into #{number}: {depth * 1000:.0f} mm deep ({how}, {steps} steps).{note}')

    def _spiral(self, start, R, c):
        """Press down; when the object stops going down, try the next point of a small spiral around the start."""
        k, steps = 0, 0
        z_goal = start[2] - c['insert_depth'] - c['hover'] - 0.004
        xy = start[:2].copy()
        while steps < c['max_steps']:
            pos, _ = self.robot.ee_pose()
            target = np.array([xy[0], xy[1], max(z_goal, pos[2] - 0.004)])
            r = self.robot.move_linear(target, R, speed=0.03)
            steps += 1
            now, _ = self.robot.ee_pose()
            self._emit('finish_step', skill='insert', step=steps, tcp=now.tolist(), action='press')
            if now[2] <= z_goal + 0.002:
                return True, steps
            if now[2] > target[2] + 0.0015:            # blocked: lift a little, next spiral point
                k += 1
                a = 2.4 * k
                rad = c['spiral_step'] * np.sqrt(k)
                xy = start[:2] + rad * np.array([np.cos(a), np.sin(a)])
                self.robot.move_linear(np.array([xy[0], xy[1], start[2]]), R, speed=0.03)
                if rad > 0.012:
                    break
        return False, steps

    def _run_policy(self, policy, start, R, hole, c):
        """The trained local policy, 10 Hz: wrist image + pose relative to the start -> a small relative move."""
        from learning.cond import condition, memory_feature
        from learning.envs import local_observation
        entry = next(iter(self.memory.lookup(self.held.obj.type.name)), None) if self.held.obj.type.learned else None
        cond = condition('insert', memory_feature(entry))       # which skill, which object (learning/cond.py)
        steps = 0
        yaw0 = 0.0
        for steps in range(1, c['max_steps'] + 1):
            obs = local_observation(self.robot, start, yaw0)
            obs['cond'] = cond
            a = np.asarray(policy(obs), float)
            if a[-1] > 0.5:                           # the policy says it is done
                break
            pos, Rn = self.robot.ee_pose()
            d = np.clip(a[:3], -1, 1) * np.array([0.003, 0.003, 0.003])
            dyaw = float(np.clip(a[3], -1, 1)) * np.deg2rad(3)
            yaw0 += dyaw
            self.robot.move_linear(pos + d, rot_z(dyaw) @ Rn, speed=0.03)
            if self.recorder is not None:
                self.recorder.step(obs, a)
            self._emit('finish_step', skill='insert', step=steps, tcp=(pos + d).tolist(), action=a.round(2).tolist())
            if self.insert_depth(hole) >= c['insert_depth']:
                return True, steps
        return False, steps

    # ---------- tools ----------
    def pull_with_tool(self, number: int, distance: float | None = None) -> Result:
        """Reach past found object `number` with the held tool's hook and drag it toward the base.

        The tool's learned outline and hook part plan it: the tool is turned so its hook points away from the base
        along the line to the object; then, over lateral offsets and distances past the object, the first placement
        where the tool's outline clears the object (so it can be lowered beside and behind it), the hook would catch
        it when dragged back, and the arm reaches both ends of the drag."""
        if not self.held:
            return Result(False, 'Not holding a tool.')
        tool = self.held.obj.type
        part = next((p for p in ('hook', 'tip', 'end') if tool.parts and p in tool.parts), None)
        if part is None or not tool.outline:
            return Result(False, f'The held {tool.name} has no learned hook (learn it with a part named "hook").')
        target = self.world.objects.get(int(number))
        if target is None:
            return Result(False, f'No object #{number}; use find first.')
        self._current(target)
        held = self.held
        z0 = float(self.table['z'])
        c = self._contact_cfg()
        u = target.top[:2] / np.linalg.norm(target.top[:2])                   # away from the base
        v = np.array([-u[1], u[0]])
        r_obj = max(target.type.footprint) / 2
        pull = distance if distance is not None else max(0.05, float(np.linalg.norm(target.top[:2])) - c['pull_to'])
        hook = np.asarray(tool.parts[part][:2], float)
        outline = np.asarray(tool.outline, float)
        dense = np.concatenate([np.linspace(outline[i], outline[(i + 1) % len(outline)], 8, endpoint=False)
                                for i in range(len(outline))])
        yaw_tool = float(np.arctan2(u[1], u[0]) - np.arctan2(hook[1], hook[0]))
        Rt = rot_z(yaw_tool)[:2, :2]
        yaw_g = yaw_tool - held.yaw_offset
        R = gripper_rotation(yaw_g, self.pitch)
        z_tool = z0 + tool.height / 2 + 0.004                                 # tool centre height while dragging
        plan, tried = None, 0
        for beyond in np.arange(r_obj + 0.008, r_obj + 0.05, 0.006):
            for lateral in sorted(np.linspace(-0.05, 0.05, 21), key=abs):
                tried += 1
                hook_w = target.top[:2] + u * beyond + v * lateral
                centre = hook_w - Rt @ hook
                pts = dense @ Rt.T + centre
                rel = pts - target.top[:2]
                if np.min(np.linalg.norm(rel, axis=1)) < r_obj + c['clearance']:
                    continue                    # the tool would come down on the object
                catches = (rel @ u > 0) & (np.abs(rel @ v) < 0.8 * r_obj)
                if not catches.any():
                    continue                    # dragging back would not meet it
                tcp = np.append(centre, z_tool) + R @ held.offset
                end = tcp - np.append(u * pull, 0.0)
                hover = tcp + np.array([0, 0, 0.05])
                if all(self.robot.reachable(p, R) for p in (tcp, end, hover)):
                    plan = (tcp, end, hover, lateral, beyond)
                    break
            if plan:
                break
        self._emit('plan', what=f'tool past #{number}', options=[dict(tried=tried)], chosen=None if plan is None else
                   dict(yaw_deg=round(float(np.rad2deg(yaw_g)), 1), tcp=plan[0].tolist(), pregrasp_height=0.05,
                        clearance_mm=round(c['clearance'] * 1000, 1), nearest=int(number)))
        if plan is None:
            return Result(False, f'Cannot place the {tool.name}\'s {part} behind #{number} from here (out of reach '
                                 f'in all {tried} placements tried). Grip the tool nearer its end (a part such as '
                                 f'"handle"), or move closer objects away.')
        tcp, end, hover, lateral, beyond = plan
        err = self._travel(hover, R, f'tool past #{number}')
        if err:
            return err
        r = self._move(tcp, R, f'lower the {part} behind #{number}')
        if not r.ok:
            return Result(False, f'Could not lower the tool: {r.reason}.')
        r = self.robot.move_linear(end, R, speed=0.04)
        self._record(dict(event='tool', skill='pull', number=number, pulled_m=pull, ok=r.ok,
                          lateral_mm=round(lateral * 1000), beyond_mm=round(beyond * 1000)))
        self._move(self.robot.ee_pose()[0] + np.array([0, 0, 0.05]), R, 'lift the tool')
        return Result(True, f'Pulled #{number} about {pull * 100:.0f} cm toward the base with the {tool.name}\'s '
                            f'{part}. Find it again to see where it is now; put the tool down when done.')

    def pour_into(self, number: int, angle_deg: float = 115.0, hold_s: float = 1.5) -> Result:
        """Tip the held container over found object `number`, hold, and tip it back.

        Tipping turns the container about a horizontal axis through its centre (turning about the tilted approach
        axis only swings it round a cone and never tips it past about 70 deg). The arm's reach decides which
        horizontal direction works: each of 12 is tried in steps, by IK alone; the one reaching the largest angle is
        used, with the container placed so that its mouth ends over the target's centre."""
        if not self.held:
            return Result(False, 'Not holding anything to pour from.')
        target = self.world.objects.get(int(number))
        if target is None:
            return Result(False, f'No object #{number}; use find first.')
        self._current(target)
        held = self.held
        h = held.obj.type.height
        u = target.top[:2] / np.linalg.norm(target.top[:2])
        clear = float((self.cfg.get('contact') or {}).get('pour_clearance', 0.03))
        best = None
        for yaw_g in self._yaws(float(np.arctan2(u[1], u[0])), np.pi / 6, target.top)[:4]:
            R = gripper_rotation(yaw_g, self.pitch)
            for k_ax in range(12):                  # horizontal axes all round: tipping needs one
                th = np.pi * k_ax / 6
                axis = np.array([np.cos(th), np.sin(th), 0.0])
                name = f'{np.rad2deg(th):.0f} deg' 
                full = _axis_angle(axis, np.deg2rad(angle_deg))
                mouth = full @ np.array([0.0, 0.0, h / 2])
                centre = target.top + np.array([0.0, 0.0, clear + max(mouth[2], 0.0) + h / 2]) - np.append(mouth[:2],
                                                                                                          0.0)
                centre[2] = max(centre[2], target.top[2] + clear + h / 2)
                tcp0 = centre + R @ held.offset
                if not self.robot.reachable(tcp0, R):
                    continue
                reached = 0.0                       # about a horizontal axis, the turn is the tilt
                for k in range(1, 13):
                    Ra = _axis_angle(axis, np.deg2rad(angle_deg) * k / 12)
                    if not self.robot.reachable(centre + Ra @ (tcp0 - centre), Ra @ R):
                        break
                    reached = angle_deg * k / 12
                if best is None or reached > best[0]:
                    best = (reached, name, axis, R, tcp0, centre)
                if reached >= angle_deg - 1e-6:
                    break
            if best and best[0] >= angle_deg - 1e-6:
                break
        if best is None or best[0] < 45:
            got = 0 if best is None else best[0]
            return Result(False, f'Cannot tip the {held.obj.label} more than {got:.0f} deg over #{number} from any side '
                                 f'the arm reaches.')
        reached, name, axis, R, tcp0, centre = best
        err = self._travel(tcp0, R, f'over #{number}')
        if err:
            return err
        self._emit('finish_start', skill='pour', how=f'tip about the {name} axis', start=tcp0.tolist())
        for k in range(1, 13):
            a = np.deg2rad(reached) * k / 12
            Ra = _axis_angle(axis, a)
            self.robot.move_linear(centre + Ra @ (tcp0 - centre), Ra @ R, speed=0.05)
            self._emit('finish_step', skill='pour', step=k, angle=round(float(np.rad2deg(a)), 1))
        if hasattr(self.robot, 'settle'):
            self.robot.settle(hold_s)
        for k in range(11, -1, -1):
            Ra = _axis_angle(axis, np.deg2rad(reached) * k / 12)
            self.robot.move_linear(centre + Ra @ (tcp0 - centre), Ra @ R, speed=0.08)
        self._record(dict(event='pour', number=number, angle_deg=round(float(reached), 1), axis=name))
        return Result(True, f'Tipped the {held.obj.label} {reached:.0f} deg over #{number} (about the {name} axis), '
                            f'held {hold_s:.1f} s, and tipped it back. Look to see what landed where.')
