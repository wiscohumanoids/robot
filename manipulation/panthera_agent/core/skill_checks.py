"""Checks by sight around a grasp and a place (mixin for core/skills.py Skills).

- `check_in_hand`: after the lift, the wrist camera looks between the jaws. The held object's mask must fill part of
  the region between the fingers (projected from the TCP pose). For thin or soft objects (a towel) the jaw width
  says nothing, so this is the grasp check.
- `align_over`: before lowering a held object onto a target, the overhead camera sees both. Measured in the same
  image, the offset between them is free of most calibration error (both shift alike), so moving by it lines the
  object up. Skipped, with a note, when either is not measured confidently (the gripper hides part of the held
  object from above).
- `check_place`: after releasing and backing off, the wrist camera, a few centimetres away, measures where the
  object ended up: its offset from the target's centre, its turn, and whether it sits on the support.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.geometry import footprint


def _outline(img, mask, colour, thickness=2):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, colour, thickness, cv2.LINE_AA)


class CheckSkills:
    """Checks by sight. Expects Skills' attributes (robot, cfg, held, _observe, _segment, _emit, _record)."""

    # The region between the open jaws, TCP frame (x along the approach, y along the closing axis).
    JAW_BOX = (-0.04, 0.012, -0.045, 0.045, -0.025, 0.025)

    def _jaw_polygon(self, camera) -> np.ndarray | None:
        x0, x1, y0, y1, z0, z1 = self.cfg.get('checks', {}).get('jaw_box', self.JAW_BOX)
        corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
        pos, R = self.robot.ee_pose()
        uv = camera.project(corners @ R.T + pos)
        uv = uv[np.isfinite(uv).all(axis=1)]
        if len(uv) < 3:
            return None
        return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)

    def _seen_at(self, mask, camera, obj, tol: float = 0.015) -> bool:
        """Whether a mask is the object at its old place on the table."""
        fp = footprint(mask, camera, float(obj.top[2]), float(obj.top[2] - obj.type.height))
        return fp is not None and float(np.linalg.norm(fp.center[:2] - obj.top[:2])) < tol

    def _object_polygon(self, camera, obj, center: np.ndarray) -> np.ndarray | None:
        """Where the held object should appear: its box (footprint x height, at its yaw) around `center`, projected.
        Gripped near its top, most of an object hangs below the jaws, so this beats the region between them."""
        L, W = obj.type.footprint
        h = obj.type.height
        c, s_ = np.cos(obj.yaw), np.sin(obj.yaw)
        pts = np.array([center + [x * c - y * s_, x * s_ + y * c, z] for x in (-L / 2, L / 2) for y in (-W / 2, W / 2)
                        for z in (-h / 2, h / 2)])
        uv = camera.project(pts)
        uv = uv[np.isfinite(uv).all(axis=1)]
        if len(uv) < 4:
            return None
        uv[:, 0] = np.clip(uv[:, 0], 0, camera.width - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, camera.height - 1)
        return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)

    def check_in_hand(self, obj, number: int, expected_center: np.ndarray | None = None) -> dict:
        """{'held': True / False / None (cannot tell), 'note', 'image', 'fill', 'measured': the object's centre as
        seen when the view shows enough of it (base frame, at `expected_center`'s height)}."""
        obs = self._observe('wrist', f'in-hand check #{number}')
        poly = self._object_polygon(obs.camera, obj, expected_center) if expected_center is not None else None
        if poly is None:
            poly = self._jaw_polygon(obs.camera)
        if poly is None:
            return dict(held=None, note='the jaws are not in the wrist view', image=None)
        region = np.zeros(obs.rgb.shape[:2], np.uint8)
        cv2.fillPoly(region, [np.round(poly).astype(np.int32)], 1)
        region = region.astype(bool)
        best, fill = None, 0.0
        instances = self._segment(obs, obj.query)
        for inst in instances:
            f = (inst.mask & region).sum() / max(region.sum(), 1)
            if f > fill:
                best, fill = inst, f
        img = obs.rgb.copy()
        cv2.polylines(img, [np.round(poly).astype(np.int32)], True, (255, 255, 0), 1, cv2.LINE_AA)
        need = float(self.cfg.get('checks', {}).get('in_hand_fill', 0.08 if expected_center is None else 0.10))
        if best is not None:
            _outline(img, best.mask, (0, 255, 0) if fill >= need else (255, 80, 80))
        held = bool(fill >= need)
        where = 'where it should be' if expected_center is not None else 'between the jaws'
        note = (f'{"seen" if held else "not seen"} {where} ({fill * 100:.0f} % of that region'
                f'{", needs " + str(round(need * 100)) + " %" if not held else ""})')
        if not held:
            # Not seeing it in the hand is not seeing it dropped: only a sighting elsewhere (still on the table
            # where it was) says the grasp failed; otherwise the jaw width decides.
            # (Lifted straight up, the held object projects onto its old place: a sighting there counts only if it is
            # outside where the held object should appear.)
            left = [i for i in instances if self._seen_at(i.mask, obs.camera, obj)
                    and (i.mask & region).sum() < 0.3 * max(i.mask.sum(), 1)]
            if left:
                note += '; it is still where it was on the table'
            else:
                held = None
                note += '; not seen elsewhere either, so the jaw width decides'
            if best is None and not left:
                fill = 0.0
        # Where it sits in the jaws: its footprint at the height the grasp put it, if the view shows enough of it.
        measured = None
        if held and best is not None and expected_center is not None:
            zc, h = float(expected_center[2]), obj.type.height
            fp = footprint(best.mask, obs.camera, zc + h / 2, zc - h / 2)
            if fp is not None:
                err = max(abs(m - e) / e for m, e in zip(fp.size, obj.type.footprint))
                if err < 0.2 and np.linalg.norm(fp.center[:2] - expected_center[:2]) < 0.015:
                    measured = np.array([fp.center[0], fp.center[1], zc])
                    off = np.linalg.norm(measured[:2] - expected_center[:2]) * 1000
                    note += f', sitting {off:.1f} mm from where the grasp aimed'
        self._record(dict(event='in_hand', number=number, held=held, fill=round(float(fill), 3),
                          measured=None if measured is None else measured.tolist()))
        self._emit('in_hand_image', number=number, held=held, image=img)
        return dict(held=held, note=note, image=img, fill=float(fill), measured=measured)

    def _measure_at(self, obs, query, z_top, height, near_xy):
        """The instance of `query` nearest `near_xy`, measured with its top face at `z_top`: (footprint, mask,
        trusted)."""
        best = None
        for inst in self._segment(obs, query):
            fp = footprint(inst.mask, obs.camera, z_top, z_top - height)
            if fp is None:
                continue
            d = float(np.linalg.norm(fp.center[:2] - near_xy[:2]))
            if best is None or d < best[3]:
                best = (fp, inst.mask, None, d)
        if best is None:
            return None, None, False
        fp, mask, _, d = best
        return fp, mask, d < 0.04

    def align_over(self, support, top: np.ndarray, what: str, shift) -> dict:
        """Line the held object up over `support` (or the table point `top`) in the overhead image, moving the arm
        by `shift((dx, dy))` (which returns a failing Result or None). Returns a summary."""
        held = self.held
        t = held.obj.type
        tol = float(self.cfg.get('checks', {}).get('align_tolerance', 0.002))
        cap = float(self.cfg.get('checks', {}).get('align_max', 0.02))
        out = dict(iterations=0, moved_mm=0.0, note='', offsets_mm=[])
        for _ in range(int(self.cfg.get('checks', {}).get('align_iterations', 2))):
            tcp, R = self.robot.ee_pose()
            centre = tcp - R @ held.offset
            obs = self._observe('overhead', f'line up {what}')
            hfp, hmask, h_ok = self._measure_at(obs, held.obj.query, centre[2] + t.height / 2, t.height, centre)
            if hfp is None or not h_ok:
                out['note'] = 'could not see the held object from above; kept the plan'
                break
            size_err = max(abs(m - e) / e for m, e in zip(hfp.size, t.footprint))
            if size_err > float(self.cfg.get('checks', {}).get('align_size_tolerance', 0.2)):
                out['note'] = (f'the held object is partly hidden from above (size {size_err * 100:.0f} % off); '
                               'kept the plan')
                break
            if support is not None:
                sfp, smask, s_ok = self._measure_at(obs, support.query, support.top[2], support.type.height,
                                                    support.raw_top if support.raw_top is not None else support.top)
                s_err = 1.0 if sfp is None else max(abs(m - e) / e for m, e in zip(sfp.size, support.type.footprint))
                if sfp is None or not s_ok or s_err > float(self.cfg.get('checks', {}).get('align_size_tolerance',
                                                                                           0.2)):
                    out['note'] = 'the target is hidden from above (under the held object); kept the plan'
                    break
                target = sfp.center[:2]
            else:                       # a table point read off the overhead image's grid: same camera, same frame
                target = top[:2]
            d = target - hfp.center[:2]
            n = float(np.linalg.norm(d))
            out['offsets_mm'].append(round(n * 1000, 1))
            img = obs.rgb.copy()
            _outline(img, hmask, (255, 255, 0))
            if support is not None:
                _outline(img, smask, (0, 255, 255))
            self._emit('align', what=what, offset_mm=round(n * 1000, 1), image=img)
            if n < tol:
                break
            if n > cap:
                out['note'] = f'measured {n * 1000:.0f} mm off, more than the {cap * 1000:.0f} mm allowed; kept the plan'
                break
            err = shift(d)
            out['iterations'] += 1
            out['moved_mm'] = round(out['moved_mm'] + n * 1000, 1)
            if err is not None:
                out['note'] = err.message
                break
        self._record(dict(event='align', what=what, **{k: v for k, v in out.items()}))
        return out

    def check_place(self, obj, expected_top: np.ndarray, support, yaw: float) -> dict:
        """Where the released object is, from the wrist camera at the retreat pose."""
        t = obj.type
        obs = self._observe('wrist', f'check the place of #{obj.number}')
        fp, mask, ok = self._measure_at(obs, obj.query, float(expected_top[2]), t.height, expected_top)
        if fp is None or not ok:
            note = 'the wrist camera does not see it where it was released (fell, or hidden): look or find to check.'
            self._record(dict(event='place_check', number=obj.number, seen=False))
            return dict(seen=False, note=note, image=obs.rgb)
        ref = support.top[:2] if support is not None else expected_top[:2]
        off = fp.center[:2] - ref
        sym = t.symmetry if t.symmetry > 1e-6 else 2 * np.pi
        turn = float(np.rad2deg(abs((fp.yaw - yaw + sym / 2) % sym - sym / 2))) if t.kind != 'round' else 0.0
        size_err = max(abs(m - e) / e for m, e in zip(fp.size, t.footprint))
        on = None
        if support is not None:
            on = bool(float(np.linalg.norm(off)) < float(self.cfg.get('checks', {}).get('seated_fraction', 0.25))
                      * min(support.type.footprint) and size_err < 0.35)
        what = f"#{support.number}'s centre" if support is not None else 'the target point'
        note = (f'{np.linalg.norm(off) * 1000:.1f} mm from {what}'
                + (f', turned {turn:.0f} deg from the plan' if t.kind != 'round' else '')
                + ('' if on is None else ('; it is on it.' if on else '; it does NOT look seated on it.'))
                + ('' if size_err < 0.35 else f' (size {size_err * 100:.0f} % off: partly hidden or tilted)'))
        img = obs.rgb.copy()
        _outline(img, mask, (0, 255, 0) if on in (None, True) else (255, 80, 80))
        self._record(dict(event='place_check', number=obj.number, seen=True, offset_mm=round(float(np.linalg.norm(off))
                                                                                             * 1000, 1),
                          turn_deg=round(turn, 1), on=on))
        return dict(seen=True, offset=off.tolist(), turn_deg=turn, on=on, note=note, image=img)
