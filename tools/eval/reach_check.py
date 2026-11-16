"""GRADER: after a navigate_to, is the robot really within arm's reach of the object, facing it?

Uses the simulator's TRUE robot pose and the env's TRUE geometry (graders may; the robot may not):
  - furniture (table, fridge, sink...): the distance from the body center straight ahead to the first
    solid surface on the env's geometry map, which must belong to the object (inside its box).
    PASS if 0.15-0.40 m (within arm's reach, not touching), square-on within 10 deg, and standing in front
    of the object (the surface straight ahead is within the object's width, +-0.10 m).
  - small objects (block, mug): the horizontal distance from the body center to the object, which must
    be within arm's reach (0.60 m), and the robot must face it within 20 deg.
  (Tightened 2026-11-10 from <= 0.45 m / 20 deg and 0.65 m / 45 deg: "got there" = ready to reach for it.)

    python3 tools/eval/reach_check.py ENV OBJECT_NAME
    e.g. python3 tools/eval/reach_check.py robocasa/kitchen_l1_s1 fridge_main_group
"""
import math, os, sys, time
import numpy as np
import rclpy, yaml
from rclpy.node import Node
from nav_msgs.msg import Odometry

MIN_GAP, MAX_GAP, MAX_SQUARE_DEG = 0.15, 0.40, 10.0
# per kind (matches hsim_behavior standoff_for): appliances whose doors swing toward the robot need room
WINDOWS = {'fridge': (0.30, 0.60), 'refrigerator': (0.30, 0.60), 'dishwasher': (0.28, 0.55), 'oven': (0.28, 0.55),
           'cabinet': (0.20, 0.50), 'drawer': (0.20, 0.50)}
CENTERED = ('fridge', 'refrigerator', 'dishwasher', 'oven', 'stove', 'sink', 'microwave')   # approached at mid-front
ARM_REACH, MAX_FACE_DEG = 0.60, 20.0
env, name = sys.argv[1], sys.argv[2]
build = f"/hsim/assets/envs/{env}/{os.environ.get('HSIM_BUILD_DIR', 'build')}"   # Wisco: build_wisco
objs = yaml.safe_load(open(f'{build}/objects.yaml'))['objects']
o = next((x for x in objs if x['name'] == name), None)
if o is None:
    print(f'no object {name!r} in {env}: ' + ', '.join(x['name'] for x in objs)); sys.exit(2)


def load_map():
    meta = yaml.safe_load(open(f'{build}/map.yaml'))
    raw = open(f"{build}/{meta['image']}", 'rb').read()
    fields, i = [], 0
    while len(fields) < 4:                       # P5 header: magic, width, height, maxval
        while raw[i:i + 1].isspace():
            i += 1
        if raw[i:i + 1] == b'#':
            while raw[i:i + 1] != b'\n':
                i += 1
            continue
        j = i
        while not raw[j:j + 1].isspace():
            j += 1
        fields.append(raw[i:j]); i = j
    w, h = int(fields[1]), int(fields[2])
    img = np.frombuffer(raw[i + 1:i + 1 + w * h], np.uint8).reshape(h, w)
    return img, meta['resolution'], meta['origin'][0], meta['origin'][1]


rclpy.init(); n = Node('reach_check'); st = {}
n.create_subscription(Odometry, '/sim/ground_truth/odom', lambda m: st.__setitem__('p', m.pose.pose), 10)
t0 = time.monotonic()
while 'p' not in st and time.monotonic() - t0 < 5:
    rclpy.spin_once(n, timeout_sec=0.1)
if 'p' not in st:
    print('no ground truth pose'); sys.exit(2)
p = st['p']
x, y = p.position.x, p.position.y
q = p.orientation
yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
cx, cy = o['position'][0], o['position'][1]
sx, sy = o['size'][0], o['size'][1]
pose = f'robot ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg)'

if max(sx, sy) < 0.35:                            # small object: in reach and roughly facing it
    d = math.hypot(cx - x, cy - y)
    face = math.degrees(math.atan2(math.sin(math.atan2(cy - y, cx - x) - yaw), math.cos(math.atan2(cy - y, cx - x) - yaw)))
    ok = d <= ARM_REACH and abs(face) < MAX_FACE_DEG
    print(f'{pose} | {name}: {d:.2f} m from my center (reach {ARM_REACH}), at {face:+.0f} deg -> {"PASS" if ok else "FAIL"}')
    sys.exit(0 if ok else 1)

img, res, ox, oy = load_map()
h, w = img.shape
gap = hit = None
for t in np.arange(0.0, 2.0, res / 2):            # straight ahead from the body center
    hx, hy = x + t * math.cos(yaw), y + t * math.sin(yaw)
    c, r = int((hx - ox) / res), h - 1 - int((hy - oy) / res)
    if 0 <= r < h and 0 <= c < w and img[r, c] < 50:
        gap, hit = t, (hx, hy); break
if gap is None:
    print(f'{pose} | {name}: nothing solid straight ahead within 2 m -> FAIL'); sys.exit(1)
# the surface hit is the object, or the counter edge right in front of it (a sink sits inside a counter): along
# the facing direction allow 0.3 m (counter depth in front of a sink); sideways the robot must be in front of it
facing_x = abs(math.cos(yaw)) > abs(math.sin(yaw))
along, side = (abs(hit[0] - cx), abs(hit[1] - cy)) if facing_x else (abs(hit[1] - cy), abs(hit[0] - cx))
half_along, half_side = (sx / 2, sy / 2) if facing_x else (sy / 2, sx / 2)
mine = along <= half_along + 0.3 and side <= half_side + 0.10
# square-on: compare with the nearest axis direction (map-aligned furniture)
sq = math.degrees(abs(math.atan2(math.sin(4 * yaw), math.cos(4 * yaw)) / 4))
kind = f"{o.get('label', '')} {o.get('class', '')} {name}".lower()
lo, hi = next((w for k, w in WINDOWS.items() if k in kind), (MIN_GAP, MAX_GAP))
# appliances: the robot's center must be in front of the middle part of the front (within 25 % of its width, >= 0.15 m)
centered_ok = True
if any(k in kind for k in CENTERED):
    lateral = abs(y - cy) if facing_x else abs(x - cx)
    width = sy if facing_x else sx
    centered_ok = lateral <= max(0.15, 0.25 * width)
MIN_GAP, MAX_GAP = lo, hi
ok = mine and centered_ok and MIN_GAP <= gap <= MAX_GAP and sq <= MAX_SQUARE_DEG
print(f'{pose} | {name}: surface {gap:.2f} m straight ahead ({"it is the " + name if mine else "NOT in front of the " + name}'
      f'{"" if centered_ok else ", not centered on its front"}), '
      f'{sq:.0f} deg off square -> {"PASS" if ok else "FAIL"} ({MIN_GAP}-{MAX_GAP} m, <= {MAX_SQUARE_DEG:.0f} deg)')
sys.exit(0 if ok else 1)
