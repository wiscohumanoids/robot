"""Native MuJoCo window for the ROS stack (Mac side). Run via tools/viewer.sh.

Shows exactly what the ROS `sim` node is simulating, what the robot itself perceives, and drives it.
No physics happens here: this window copies the robot pose streamed by the sim (UDP 9870) onto the
same model. Arrow-key commands and typed tasks go to the `teleop` node (it says hello on UDP 9873 and we
answer to that address; direct UDP 9871 as a fallback); the robot's own
map / lidar / object memory / path arrive from the `viewer_bridge` node (UDP 9872).

Main view:  a high camera that follows the robot and swings round behind it; walls between the
            camera and the robot are hidden. Orbit/zoom with the mouse any time.
Panel:      the robot's view: its own map filling in, lidar, object memory, Nav2's path.

Keys (click the window first):
  Up/Down = forward/back speed   Left/Right = turn   , . = sidestep   / = stop
  R = reset robot                F = camera swings with the robot on/off
  V = layout: panel in the corner / side by side / robot's view only / panel off
Tasks: type them in THIS terminal and press Enter, e.g.  walk to the fridge   (stop = cancel)
"""

import argparse
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from perception_view import PerceptionView
from viewer_camera import Cutaway, FollowCam

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets/robots/g1/mjlab_1.6.0/g1_velocity_flat.xml"
PROFILES = ROOT / "ros2_ws/src/hsim_bringup/config/robots"     # robot profiles (same files the launch uses)
CMD_LIMITS = np.array([[-1.0, 1.5], [-0.5, 0.5], [-0.7, 0.7]])
CMD_STEP = 0.1
KEYS = {265: (0, +1), 264: (0, -1), 263: (2, +1), 262: (2, -1), 44: (1, +1), 46: (1, -1)}
KEY_F, KEY_V, KEY_R, KEY_SLASH = 70, 86, 82, 47
CMD_PACKET = struct.Struct("<4s3fB")
SEND_HZ = 30.0
PANEL_HZ = 30.0
LAYOUTS = ["corner", "side by side", "robot's view", "off"]


def load_world(env, scene, build_dir="build"):
    if env:
        scene = ROOT / "assets/envs" / env / build_dir / "world.mjb"
        if not scene.exists():
            raise SystemExit(f"{scene} missing: run .venv/bin/python tools/env_build.py assets/envs/{env} --robot <robot>")
    return mujoco.MjModel.from_binary_path(str(scene)) if str(scene).endswith(".mjb") \
        else mujoco.MjModel.from_xml_path(str(scene))


def panel_rect(layout, vp):
    """Where the robot's-view panel goes in the window (MjrRect origin is bottom-left)."""
    W, H = vp.width, vp.height
    if layout == "corner":
        w = int(W * 0.36)
        h = int(w * 0.62)
        return mujoco.MjrRect(W - w - 12, 12, w, h)
    if layout == "side by side":
        return mujoco.MjrRect(W // 2, 0, W - W // 2, H)
    if layout == "robot's view":
        return mujoco.MjrRect(0, 0, W, H)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", help="environment, e.g. test/table_room (must match the sim's env:=)")
    ap.add_argument("--scene", type=Path, default=None, help="world without --env (default: the robot's training world)")
    ap.add_argument("--robot", default="g1", help="g1 | wisco: must match the stack's robot:= (robot profile)")
    ap.add_argument("--state-port", type=int, default=9870)
    ap.add_argument("--perception-port", type=int, default=9872)
    ap.add_argument("--cmd-host", default="127.0.0.1")
    ap.add_argument("--cmd-port", type=int, default=9871)
    ap.add_argument("--hello-port", type=int, default=9873)
    ap.add_argument("--layout", choices=LAYOUTS, default="corner")
    args = ap.parse_args()

    import yaml
    prof = yaml.safe_load((PROFILES / f"{args.robot}.yaml").read_text())
    scene = args.scene or ROOT / prof["world"].removeprefix("/hsim/")
    m = load_world(args.env, scene, prof["env_build_dir"])
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "init_state"))
    mujoco.mj_forward(m, d)
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
    panel = PerceptionView(load_world(args.env, scene, prof["env_build_dir"]))
    follow, cutaway = FollowCam(), Cutaway(m)

    def udp_in(port):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 21)
        s.bind(("0.0.0.0", port))
        s.setblocking(False)
        return s
    rx, rx_per = udp_in(args.state_port), udp_in(args.perception_port)
    # commands go back to wherever the teleop node's hellos come from (Docker passes replies to
    # traffic the container started, but not always UDP the Mac starts); direct port as a fallback
    tx = udp_in(args.hello_port)
    teleop_addr = [(args.cmd_host, args.cmd_port)]

    def send_cmd(data):
        try:
            tx.sendto(data, teleop_addr[0])
        except OSError:
            pass

    cmd = np.zeros(3)
    pending = {"reset": False, "follow": False, "layout": False}
    layout = [LAYOUTS.index(args.layout)]
    lock = threading.Lock()

    def show():
        print(f"\rcommand: vx={cmd[0]:+.1f}  vy={cmd[1]:+.1f}  wz={cmd[2]:+.1f}      ", end="", flush=True)

    def on_key(key):
        with lock:
            if key in KEYS:
                axis, sign = KEYS[key]
                cmd[axis] = np.clip(cmd[axis] + sign * CMD_STEP, *CMD_LIMITS[axis])
            elif key == KEY_SLASH:
                cmd[:] = 0.0
            elif key == KEY_R:
                cmd[:] = 0.0
                pending["reset"] = True
            elif key == KEY_F:
                pending["follow"] = True
            elif key == KEY_V:
                pending["layout"] = True
            else:
                return
            show()

    # The panel renders in its own thread (~30-60 ms a frame), so the main view stays smooth.
    panel_job = {"qpos": None, "cam": None, "size": None, "img": None}
    panel_cv = threading.Condition()

    def panel_worker():
        while True:
            with panel_cv:
                while panel_job["qpos"] is None:
                    panel_cv.wait()
                qpos, cam, size = panel_job["qpos"], panel_job["cam"], panel_job["size"]
                panel_job["qpos"] = None
            try:
                img = panel.render(qpos, cam, *size)
            except Exception as e:              # never let one bad frame kill the panel
                print(f"\npanel frame skipped: {e!r}", flush=True)
                continue
            with panel_cv:
                panel_job["img"] = img
    threading.Thread(target=panel_worker, daemon=True).start()

    def read_tasks():                     # typed tasks -> teleop node -> /planner/command
        for line in sys.stdin:
            text = line.strip()
            if text:
                send_cmd(b"HTXT" + text.encode("utf-8")[:2000])
                print(f'sent task: "{text}"', flush=True)
    threading.Thread(target=read_tasks, daemon=True).start()

    print(__doc__)
    print(f"waiting for the ROS sim on udp/{args.state_port} "
          f"(inside tools/ros.sh shell: ros2 launch hsim_bringup autonomy.launch.py env:=...) ...")
    connected, last_rx, next_send, next_panel = False, 0.0, 0.0, 0.0
    last_loop = time.monotonic()
    # these keys also flip MuJoCo's own display toggles; put those back
    vis_restore = {KEY_F: mujoco.mjtVisFlag.mjVIS_CONTACTFORCE, KEY_V: mujoco.mjtVisFlag.mjVIS_TENDON}

    with mujoco.viewer.launch_passive(m, d, key_callback=on_key, show_left_ui=False, show_right_ui=False) as viewer:
        saved_flags = {k: viewer.opt.flags[f] for k, f in vis_restore.items()}
        while viewer.is_running():
            now = time.monotonic()
            dt, last_loop = now - last_loop, now

            # newest robot state from the sim (drain the socket, keep the last packet)
            latest = None
            while True:
                try:
                    latest, _ = rx.recvfrom(65536)
                except BlockingIOError:
                    break
            if latest and latest[:4] == b"HSIM":
                nq, _t = struct.unpack_from("<Id", latest, 4)
                if nq == m.nq:
                    d.qpos[:] = np.frombuffer(latest, dtype="<f8", count=nq, offset=16)
                    mujoco.mj_forward(m, d)
                    last_rx = now
                    if not connected:
                        connected = True
                        print("\nconnected to the ROS sim. Click the window for arrow keys; type tasks here.")
                        show()
                else:
                    print(f"\nworld mismatch: sim has nq={nq}, this window nq={m.nq}. "
                          f"Use the same environment on both sides (env:= / --env).")
            if connected and now - last_rx > 1.0:
                connected = False
                print("\nlost the ROS sim (stopped?). Waiting ...")
            while True:                      # teleop node's hello: where to send commands
                try:
                    pkt, addr = tx.recvfrom(64)
                except BlockingIOError:
                    break
                if pkt[:4] == b"HELO":
                    teleop_addr[0] = addr
            while True:                      # what the robot perceives
                try:
                    pkt, _ = rx_per.recvfrom(65536)
                except BlockingIOError:
                    break
                panel.on_packet(pkt)

            with lock:
                if pending["follow"]:
                    pending["follow"] = False
                    on = follow.toggle_heading(d)
                    print(f"\ncamera swings with the robot: {'on' if on else 'off'}")
                if pending["layout"]:
                    pending["layout"] = False
                    layout[0] = (layout[0] + 1) % len(LAYOUTS)
                    print(f"\nlayout: {LAYOUTS[layout[0]]}")
                    viewer.clear_images()
            for k, f in vis_restore.items():
                viewer.opt.flags[f] = saved_flags[k]

            # main camera + see-through walls
            vp = viewer.viewport
            side = LAYOUTS[layout[0]] == "side by side"
            follow.update(viewer.cam, d, pelvis, dt, shift=0.25 if side else 0.0,
                          aspect=(vp.width / max(vp.height, 1)) if vp else 1.6)
            p = d.xpos[pelvis]
            cutaway.update(d, viewer.cam, [p + [0, 0, 0.5], p, p - [0, 0, 0.6]], now)

            # robot's-view panel
            rect = panel_rect(LAYOUTS[layout[0]], vp) if vp else None
            if rect is not None and now >= next_panel:
                next_panel = now + 1.0 / PANEL_HZ
                pcam = mujoco.MjvCamera()                # same angle, robot centered
                pcam.lookat[:] = follow.lookat
                pcam.azimuth, pcam.elevation, pcam.distance = viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance
                with panel_cv:
                    img = panel_job["img"]
                    panel_job["img"] = None
                    panel_job.update(qpos=d.qpos.copy(), cam=pcam, size=(rect.width, rect.height))
                    panel_cv.notify()
                if img is not None:
                    h, w = img.shape[:2]
                    viewer.set_images([(mujoco.MjrRect(rect.left, rect.bottom + rect.height - h, w, h),
                                        np.ascontiguousarray(img))])

            # status text
            st = panel.live.get("status", {}) if panel.connected else {}
            lines = [("Task", st.get("skill") or "-"), ("Robot", st.get("reply") or "-")]
            if st.get("say"):
                lines.append(("Says", st["say"]))
            if st.get("state"):
                lines.append(("State", st["state"]))
            if not panel.connected:
                lines.append(("Panel", "waiting for the robot's perception (viewer_bridge)"))
            texts = [(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                      "\n".join(k for k, _ in lines), "\n".join(v for _, v in lines)),
                     (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                      "arrows drive  ,/. sidestep  / stop  R reset\nF camera follows heading  V layout",
                      "type tasks in the terminal")]
            viewer.set_texts(texts)

            if now >= next_send:
                next_send = now + 1.0 / SEND_HZ
                with lock:
                    flags = 1 if pending["reset"] else 0
                    pending["reset"] = False
                    send_cmd(CMD_PACKET.pack(b"HCMD", *cmd.astype(np.float32), flags))
            viewer.sync()
            time.sleep(1 / 120)


if __name__ == "__main__":
    main()
