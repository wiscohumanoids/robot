#!/usr/bin/env python3
"""Laptop stand-in for the ESP32 transmitter, for testing estop_receiver.py.

Same packets as the firmware: "ALIVE <seq>" every 50 ms, "STOP <seq>" every
20 ms while the button is pressed.

Keys:  s = press/release button   k = kill/revive transmitter
       l = lossy radio (drop ~50%)  q = quit
"""

import argparse
import os
import random
import select
import socket
import sys
import termios
import time
import tty


def log(msg):
    t = time.time()
    print(time.strftime("[%H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}] {msg}",
          flush=True)


def main():
    p = argparse.ArgumentParser(description="Fake ESP32 E-stop transmitter")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4242)
    p.add_argument("--period-ms", type=float, default=50)
    p.add_argument("--stop-period-ms", type=float, default=20)
    p.add_argument("--loss", type=float, default=0.5)
    args = p.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pressed, powered, lossy, seq = False, True, False, 0
    next_send = time.monotonic()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd) if os.isatty(fd) else None
    if old:
        tty.setcbreak(fd)          # read single keys without Enter
    stdin_open = True
    log(f"sending to {args.host}:{args.port}. Keys: s=button k=kill l=lossy q=quit")
    try:
        while True:
            now = time.monotonic()
            if powered and now >= next_send:
                seq += 1
                if not (lossy and random.random() < args.loss):
                    sock.sendto(f"{'STOP' if pressed else 'ALIVE'} {seq}".encode(), (args.host, args.port))
                next_send = now + (args.stop_period_ms if pressed else args.period_ms) / 1000

            wait = max(0.0, next_send - time.monotonic()) if powered else 0.1
            ready, _, _ = select.select([fd] if stdin_open else [], [], [], wait)
            if not ready:
                continue
            key = os.read(fd, 1).decode(errors="ignore").lower()
            if not key:
                stdin_open = False  # stdin closed (piped input ended)
            elif key == "s":
                pressed = not pressed
                next_send = time.monotonic()
                log("BUTTON PRESSED -> STOP" if pressed else "button released -> ALIVE")
            elif key == "k":
                powered = not powered
                next_send = time.monotonic()
                log("transmitter revived" if powered else "TRANSMITTER KILLED (no packets)")
            elif key == "l":
                lossy = not lossy
                log(f"lossy radio {'ON' if lossy else 'OFF'}")
            elif key == "q":
                break
    except KeyboardInterrupt:
        pass
    finally:
        if old:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
