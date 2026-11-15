"""GRADER: an open viewer must not block navigation, and its arrow keys must still take over.

Regression test for: the viewer sends its key command 30x a second even when idle (all zeros); the
teleop node used to republish those zeros at twist_mux priority 100, so Nav2 (priority 10) never got
through and the robot stood still with a viewer open.

Run with the stack launched with viewer:=true (so the `teleop` node is up):
    python3 tools/eval/viewer_idle_nav.py X Y YAW_DEG

1. A fake viewer sends idle (zero) commands straight to the teleop node (UDP 9871) the whole time.
2. A Nav2 goal must still succeed (graded by nav_goal.py on true position).
3. Then a fake arrow key (vx = 0.5) must reach /cmd_vel, and stop reaching it ~0.5 s after release.
"""
import struct, subprocess, sys, threading, time
from pathlib import Path

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node

ADDR = ("127.0.0.1", 9871)
PACKET = struct.Struct("<4s3fB")
cmd = [0.0, 0.0, 0.0]
running = True


def fake_viewer():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while running:
        s.sendto(PACKET.pack(b"HCMD", *cmd, 0), ADDR)
        time.sleep(1 / 30)


threading.Thread(target=fake_viewer, daemon=True).start()
time.sleep(1.0)

# 2. navigation with an idle viewer open
nav = subprocess.run([sys.executable, str(Path(__file__).parent / "nav_goal.py"), *sys.argv[1:4]],
                     capture_output=True, text=True)
nav_line = (nav.stdout.strip().splitlines() or ["no output"])[-1]
nav_ok = nav.returncode == 0

# 3. arrow keys still win, and let go
rclpy.init()
n = Node("viewer_idle_check")
seen = []
n.create_subscription(TwistStamped, "/cmd_vel", lambda m: seen.append((time.monotonic(), m.twist.linear.x)), 50)


def spin_for(t):
    end = time.monotonic() + t
    while time.monotonic() < end:
        rclpy.spin_once(n, timeout_sec=0.02)


spin_for(3.0)                                  # let ROS discovery connect us to /cmd_vel first
cmd[0] = 0.5
t_press = time.monotonic()
spin_for(2.0)
pressed = [vx for t, vx in seen if t > t_press + 0.5]
cmd[0] = 0.0
t_release = time.monotonic()
spin_for(2.0)
after = [vx for t, vx in seen if t > t_release + 1.0]
running = False
keys_ok = bool(pressed) and max(pressed) > 0.4
release_ok = not after or max(abs(v) for v in after) < 0.05

ok = nav_ok and keys_ok and release_ok
print(f"idle viewer + Nav2: {nav_line} | arrow key reached /cmd_vel: {'yes' if keys_ok else 'NO'} | "
      f"released: {'yes' if release_ok else 'NO'} -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
