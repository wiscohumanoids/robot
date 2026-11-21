"""Generate a socketed-block stacking scene for the left Panthera arm."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from .mega_bloks import BODY_HEIGHT_M, COLORS, PITCH_M, STUD_HEIGHT_M, TABLE_Z
from .mega_bloks import _build_xml, _numbers, _write_scene


_SOCKET_DEPTH_M = .016
_SOCKET_HALF_WIDTH_M = .013
STACK_CENTER_GAP_M = BODY_HEIGHT_M + STUD_HEIGHT_M - _SOCKET_DEPTH_M


def _add_sockets(body, block):
    # The original box becomes the solid upper portion of the shell. The
    # remaining boxes form one open socket beneath each stud.
    roof = body.find(f"geom[@name='{block['body_geom']}']")
    half_length, half_width = np.asarray(block['dimensions_m'][:2]) / 2
    rim_height = _SOCKET_DEPTH_M / 2
    rim_z = rim_height - BODY_HEIGHT_M / 2
    roof.set('pos', _numbers([0, 0, rim_height]))
    roof.set('size', _numbers([half_length, half_width, BODY_HEIGHT_M / 2 - rim_height]))
    shell_geoms = [(roof, half_length * half_width * (BODY_HEIGHT_M / 2 - rim_height))]

    def add_wall(label, x, y, hx, hy):
        wall = ET.SubElement(body, 'geom', {
            **roof.attrib, 'name': f"{block['body_name']}_shell_{label}",
            'pos': _numbers([x, y, rim_z]), 'size': _numbers([hx, hy, rim_height]),
        })
        shell_geoms.append((wall, hx * hy * rim_height))

    stud_count = block['shape'][0]
    end_inner = (stud_count - 1) * PITCH_M / 2 + _SOCKET_HALF_WIDTH_M
    end_half = (half_length - end_inner) / 2
    side_half = (half_width - _SOCKET_HALF_WIDTH_M) / 2
    for sign, side, end in ((-1, 'left', 'back'), (1, 'right', 'front')):
        add_wall(side, 0, sign * (half_width - side_half), half_length, side_half)
        add_wall(end, sign * (half_length - end_half), 0, end_half, _SOCKET_HALF_WIDTH_M)
    for i in range(stud_count - 1):
        x = (i + 1 - stud_count / 2) * PITCH_M
        label = 'divider' if stud_count == 2 else f'divider_{i}'
        add_wall(label, x, 0, PITCH_M / 2 - _SOCKET_HALF_WIDTH_M, _SOCKET_HALF_WIDTH_M)

    # Give the shell the same total mass as the original body box.
    total = sum(volume for _, volume in shell_geoms)
    original_mass = float(roof.get('mass'))
    for geom, volume in shell_geoms:
        geom.set('mass', str(original_mass * volume / total))
    block['shell_geoms'] = [geom.get('name') for geom, _ in shell_geoms]


def build_stack_scene(output_path, *, seed=0, count=None, lengths=None):
    """Write free-jointed, socketed blocks in the left arm's work area.

    Without lengths, count selects the standard two- to four-block arrangement.
    An explicit sequence gives each row's length in studs and determines the
    number of blocks. All rows are one stud wide. Seated layers have body-center
    spacing STACK_CENTER_GAP_M regardless of the number of engaged studs.
    """
    if lengths is None:
        count = 2 if count is None else count
        if not isinstance(count, int) or not 2 <= count <= 4:
            raise ValueError("count must be 2, 3, or 4")
        lengths = (2,) * count
        centers = np.array([[.231, .072], [.231, .177]])
        if count > 2:
            centers = np.array([[.182, .072], [.182, .177], [.277, .125], [.277, .225]])[:count]
        jitter = [.012, .008]
        yaw_range = 12.
    else:
        lengths = tuple(lengths)
        if not 2 <= len(lengths) <= 6 or any(
                type(n) is not int or n not in (2, 3, 4) for n in lengths):
            raise ValueError("lengths must describe two to six rows of 2, 3, or 4 studs")
        if count is not None and count != len(lengths):
            raise ValueError("count must match the number of entries in lengths")
        count = len(lengths)
        # Two columns leave room for the ends of 4-stud rows. The first slot
        # has space on its +Y side to assemble the gate and approach both walls.
        centers = np.array([[.180, .068], [.182, .240], [.323, .075],
                            [.320, .183], [.178, -.040], [.324, -.032]])[:count]
        # Give a pair the unobstructed central wrist view. In the denser
        # inventory, draw the far wall inward so both of its studs stay visible.
        if count == 2:
            centers[1] = [.285, .140]
        elif count >= 5:
            centers[1] = [.182, .195]
        jitter = [.004, .004]
        yaw_range = 5.
    rng = np.random.default_rng(seed)
    colors = rng.choice(tuple(COLORS), size=count, replace=False)
    centers += rng.uniform(-np.array(jitter), jitter, size=(count, 2))
    yaws = rng.uniform(-np.deg2rad(yaw_range), np.deg2rad(yaw_range), size=count)
    placements = [
        (length, 1, [*xy, TABLE_Z + BODY_HEIGHT_M / 2 + .0002], yaw, color)
        for length, xy, yaw, color in zip(lengths, centers, yaws, colors)
    ]
    root, metadata = _build_xml(placements)
    root.set('model', 'panthera_wood_mega_bloks_stack')
    for block in metadata:
        body = root.find(f"worldbody/body[@name='{block['body_name']}']")
        _add_sockets(body, block)
    return _write_scene(root, output_path), metadata
