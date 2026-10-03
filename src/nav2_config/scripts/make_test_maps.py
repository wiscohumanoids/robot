#!/usr/bin/env python3
"""Generate the fake test maps in ../maps (plain Python, no ROS, no numpy).

Every map is the same size as slam_stub's (10 m x 10 m, 0.05 m/cell, origin -5,-5), all free
except the obstacles below, and the robot starts at (0, 0). Each map gets three files:
<name>.pgm + <name>.yaml (what Nav2's map_server loads) and <name>.json (title, goal and the
obstacle rectangles, read by avoidance_test.py and the HTML report).

  obstacle_1/2/3  1, 2 or 3 square boxes side by side across the straight line to the goal
                  (4, 0); each extra box widens the wall to go around.
  unreachable     the goal (4, 0) sits inside a closed box of walls, so no path exists. The test
                  expects Nav2 to give up quickly (ABORTED) instead of wandering or retrying.
  zigzag          three walls that span the whole map from edge to edge, each with a gap
                  1.2 m wide, alternating top, bottom, top. There is no way around, so the
                  only route to the goal (4.5, 0) weaves through all three gaps.

    python3 scripts/make_test_maps.py
"""
import json
import os

SIZE, RES, ORIGIN = 200, 0.05, -5.0
EDGE = 5.0         # m, the map spans [-EDGE, EDGE] on both axes
BOX = 0.6          # m, side of one box
ROW_X = 2.0        # m, centre of the box row along x


def row_of_boxes(n):
    half = n * BOX / 2.0
    return {'title': f'{n} box' + ('es' if n > 1 else ''), 'goal': [4.0, 0.0],
            'note': f'wall {n * BOX:.1f} m wide, across the straight line to the goal',
            'rects': [[ROW_X - BOX / 2.0, -half, ROW_X + BOX / 2.0, half]]}


def zigzag():
    thick, gap_w = 0.3, 1.2
    walls = [  # (x of wall centre, gap y-range)
        (1.0, (1.0, 1.0 + gap_w)),        # gap high
        (2.2, (-1.0 - gap_w, -1.0)),      # gap low
        (3.4, (1.0, 1.0 + gap_w)),        # gap high
    ]
    rects = []
    for x, (g0, g1) in walls:
        rects.append([x - thick / 2, -EDGE, x + thick / 2, g0])   # below the gap
        rects.append([x - thick / 2, g1, x + thick / 2, EDGE])    # above the gap
    return {'title': 'Zigzag', 'goal': [4.5, 0.0],
            'note': '3 full-width walls, gaps 1.2 m wide alternating high, low, high: no way around',
            'rects': rects}


def unreachable():
    t = 0.3   # wall thickness; the free interior is 1.2 m x 1.2 m around the goal
    return {'title': 'Sealed goal', 'goal': [4.0, 0.0], 'expect': 'no_path',
            'note': 'goal inside a closed box: no path exists, so Nav2 should give up at once',
            'rects': [[3.1, -0.9, 3.1 + t, 0.9], [4.9 - t, -0.9, 4.9, 0.9],
                      [3.1 + t, -0.9, 4.9 - t, -0.9 + t], [3.1 + t, 0.9 - t, 4.9 - t, 0.9]]}


MAPS = {'obstacle_1': row_of_boxes(1), 'obstacle_2': row_of_boxes(2),
        'obstacle_3': row_of_boxes(3), 'zigzag': zigzag(), 'unreachable': unreachable()}


def build(rects):
    grid = [[254] * SIZE for _ in range(SIZE)]   # 254 = free, 0 = occupied; row 0 is the top (max y)
    for row in range(SIZE):
        y = ORIGIN + (SIZE - 1 - row + 0.5) * RES
        for col in range(SIZE):
            x = ORIGIN + (col + 0.5) * RES
            if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in rects):
                grid[row][col] = 0
    return grid


def write(name, spec, out_dir):
    with open(os.path.join(out_dir, name + '.pgm'), 'wb') as f:
        f.write(b'P5\n%d %d\n255\n' % (SIZE, SIZE))
        for row in build(spec['rects']):
            f.write(bytes(row))
    with open(os.path.join(out_dir, name + '.yaml'), 'w') as f:
        f.write(f'image: {name}.pgm\nmode: trinary\nresolution: {RES}\n'
                f'origin: [{ORIGIN}, {ORIGIN}, 0.0]\nnegate: 0\n'
                'occupied_thresh: 0.65\nfree_thresh: 0.25\n')
    with open(os.path.join(out_dir, name + '.json'), 'w') as f:
        json.dump(spec, f, indent=1)


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'maps')
    for name, spec in MAPS.items():
        write(name, spec, out)
        print(f"wrote {name}: {spec['title']}, goal {spec['goal']}")
