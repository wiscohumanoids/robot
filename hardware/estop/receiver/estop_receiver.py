#!/usr/bin/env python3
"""Wireless E-stop receiver for the RK3588 (runs on any Linux; macOS for testing).

Listens for UDP "ALIVE <seq>" / "STOP <seq>" from the ESP32 and drives the
E-stop GPIO: HIGH = power allowed, LOW = power cut.

  python3 estop_receiver.py run                       # laptop: GPIO is printed
  python3 estop_receiver.py run --gpio gpiod --pin GPIO1_B2   # RK3588
  python3 estop_receiver.py status                    # from another shell / SSH
  python3 estop_receiver.py reset                     # arm or reset

Rules:
  * Starts with power OFF.
  * Power goes ON only after 5 ALIVEs in a row AND a `reset` command.
  * STOP, or no ALIVE for > 150 ms, cuts power and LATCHES it off.
  * A latched stop clears only with a healthy link (button released, 5 fresh
    ALIVEs) AND another `reset`. Nothing re-powers the robot on its own.

`reset` / `status` go to a control port on 127.0.0.1 only, so resetting needs
a shell on the robot (local or SSH); nothing on the Wi-Fi network can do it.
"""

import argparse
import select
import signal
import socket
import sys
import time

BOOT, READY, RUNNING, LATCHED = "BOOT", "READY", "RUNNING", "LATCHED"


def log(msg):
    t = time.time()
    print(time.strftime("[%H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}] {msg}",
          flush=True)


# ---------------------------------------------------------------- GPIO output

class FakeGPIO:
    """Laptop stand-in: prints instead of touching hardware."""

    def __init__(self):
        self.value = None

    def set(self, high):
        if high != self.value:
            self.value = high
            log(">>> GPIO HIGH -> POWER ON" if high else ">>> GPIO LOW  -> POWER CUT")

    def close(self):
        self.set(False)


class GpiodOutput:
    """Real pin via libgpiod v2 (`pip install "gpiod>=2"`). UNTESTED on hardware."""

    def __init__(self, chip, line):
        import gpiod
        from gpiod.line import Direction, Value

        self._Value = Value
        self._line = line
        self.value = False
        # Requested already LOW, so there is never a glitch HIGH.
        self._req = gpiod.request_lines(chip, consumer="estop_receiver", config={
            line: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE)})
        log(f"gpiod: {chip} line {line} driven LOW")

    def set(self, high):
        if high != self.value:
            self.value = high
            self._req.set_value(self._line, self._Value.ACTIVE if high else self._Value.INACTIVE)
            log(">>> GPIO HIGH -> POWER ON" if high else ">>> GPIO LOW  -> POWER CUT")

    def close(self):
        # Drive LOW before letting go. A hard crash skips this; see README.
        self.set(False)
        self._req.release()


def rockchip_pin(name):
    """'GPIO1_B2' -> ('/dev/gpiochip1', 10). Same rule on RK3588 and RK3588S."""
    try:
        chip, rest = name.upper().replace("GPIO", "").split("_")
        return f"/dev/gpiochip{int(chip)}", "ABCD".index(rest[0]) * 8 + int(rest[1:])
    except (ValueError, IndexError):
        sys.exit(f"bad --pin {name!r}; expected like GPIO1_B2")


# ---------------------------------------------------------------- state machine

class Receiver:
    def __init__(self, gpio, timeout_s=0.150, needed=5):
        self.gpio = gpio
        self.timeout = timeout_s
        self.needed = needed        # ALIVEs in a row before arm/reset is allowed
        self.state = BOOT
        self.last_alive = None      # time.monotonic() of last ALIVE
        self.last_stop = None
        self.streak = 0             # consecutive on-time ALIVEs
        self.was_healthy = False
        self.last_seq = None
        self.rx_alive = self.rx_stop = self.lost = 0
        self.gpio.set(False)

    def healthy(self, now):
        return (self.streak >= self.needed
                and now - self.last_alive <= self.timeout
                and (self.last_stop is None or now - self.last_stop > self.timeout))

    def why_not_healthy(self, now):
        if self.last_stop is not None and now - self.last_stop <= self.timeout:
            return "STOP still arriving - release the E-stop button first"
        if self.last_alive is None:
            return "no heartbeat received yet"
        if now - self.last_alive > self.timeout:
            return f"no heartbeat for {(now - self.last_alive) * 1000:.0f} ms"
        return f"need {self.needed} heartbeats in a row, have {self.streak}"

    def on_packet(self, data, now):
        parts = data.decode(errors="ignore").split()
        if not parts:
            return
        if len(parts) > 1 and parts[1].isdigit():
            seq = int(parts[1])
            if self.last_seq is not None and seq > self.last_seq + 1:
                self.lost += seq - self.last_seq - 1
            self.last_seq = seq

        if parts[0] == "ALIVE":
            self.rx_alive += 1
            on_time = self.last_alive is not None and now - self.last_alive <= self.timeout
            self.streak = self.streak + 1 if on_time else 1
            self.last_alive = now
        elif parts[0] == "STOP":
            self.rx_stop += 1
            first = self.last_stop is None or now - self.last_stop > self.timeout
            self.last_stop = now
            self.streak = 0
            if self.state == RUNNING:
                self.latch("STOP received (E-stop button pressed)")
            elif first:
                log("STOP received (power already off)")
        # anything else is ignored

    def tick(self, now):
        """Called every loop. The timeout check is what catches a dead/out-of-range ESP."""
        if self.last_alive is not None and now - self.last_alive > self.timeout:
            if self.state == RUNNING:
                self.latch(f"heartbeat lost: no ALIVE for {(now - self.last_alive) * 1000:.0f} ms")
            self.streak = 0

        h = self.last_alive is not None and self.healthy(now)
        if h != self.was_healthy:
            self.was_healthy = h
            if h:
                log(f"link healthy ({self.streak} heartbeats in a row)")
                if self.state == BOOT:
                    self.state = READY
                    log("BOOT -> READY: send `reset` to arm")
                elif self.state == LATCHED:
                    log("reset now allowed")
            else:
                log(f"link not healthy: {self.why_not_healthy(now)}")
                if self.state == READY:
                    self.state = BOOT
                    log("READY -> BOOT")

    def latch(self, reason):
        self.gpio.set(False)
        prev, self.state = self.state, LATCHED
        log(f"!!! E-STOP LATCHED ({prev} -> LATCHED): {reason}")

    def reset(self, now):
        if self.state == RUNNING:
            log("reset ignored: already RUNNING")
        elif self.last_alive is None or not self.healthy(now):
            log(f"reset REFUSED in {self.state}: {self.why_not_healthy(now)}")
        else:
            prev, self.state = self.state, RUNNING
            self.gpio.set(True)
            log(f"{prev} -> RUNNING ({'armed' if prev == READY else 'reset'})")

    def status_line(self, now):
        h = self.last_alive is not None and self.healthy(now)
        age = "never" if self.last_alive is None else f"{(now - self.last_alive) * 1000:.0f} ms ago"
        return (f"state={self.state} power={'ON' if self.gpio.value else 'OFF'} healthy={h}"
                + ("" if h else f" ({self.why_not_healthy(now)})")
                + f" last_alive={age} rx_alive={self.rx_alive} rx_stop={self.rx_stop}"
                f" lost={self.lost}")


# ---------------------------------------------------------------- commands

def run(args):
    if args.gpio == "gpiod":
        chip, line = rockchip_pin(args.pin) if args.pin else (args.chip, args.line)
        gpio = GpiodOutput(chip, line)
    else:
        gpio = FakeGPIO()
    rx = Receiver(gpio, args.timeout_ms / 1000, args.needed)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    sock.setblocking(False)
    ctl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctl.bind(("127.0.0.1", args.control_port))
    ctl.setblocking(False)

    def on_sigterm(_signum, _frame):   # `systemctl stop` cuts power like Ctrl+C
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_sigterm)

    log(f"listening on UDP {args.bind}:{args.port}, gpio={args.gpio}, "
        f"timeout {args.timeout_ms:.0f} ms. State BOOT, power OFF.")
    try:
        while True:
            ready, _, _ = select.select([sock, ctl], [], [], 0.005)
            if sock in ready:
                while True:
                    try:
                        data, _ = sock.recvfrom(1024)
                    except BlockingIOError:
                        break
                    rx.on_packet(data, time.monotonic())
            if ctl in ready:
                data, addr = ctl.recvfrom(1024)
                cmd = data.decode(errors="ignore").strip().upper()
                if cmd == "RESET":
                    rx.reset(time.monotonic())
                if cmd in ("RESET", "STATUS"):
                    ctl.sendto(rx.status_line(time.monotonic()).encode(), addr)
            rx.tick(time.monotonic())
    except KeyboardInterrupt:
        pass
    finally:
        log("shutting down: power cut first")
        gpio.close()
        log(rx.status_line(time.monotonic()))


def control(args):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(1.0)
    s.sendto(args.command.upper().encode(), ("127.0.0.1", args.control_port))
    try:
        reply = s.recvfrom(4096)[0].decode()
    except (socket.timeout, ConnectionRefusedError):
        sys.exit("no answer - is `estop_receiver.py run` running?")
    print(reply)
    if args.command == "reset" and "state=RUNNING" not in reply:
        sys.exit(1)   # refused; the reason is in the line above


def main():
    p = argparse.ArgumentParser(description="Wireless E-stop receiver")
    p.add_argument("--control-port", type=int, default=4250, help="localhost UDP port for status/reset")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the receiver")
    r.add_argument("--bind", default="0.0.0.0")
    r.add_argument("--port", type=int, default=4242)
    r.add_argument("--timeout-ms", type=float, default=150)
    r.add_argument("--needed", type=int, default=5, help="ALIVEs in a row before reset is allowed")
    r.add_argument("--gpio", choices=["fake", "gpiod"], default="fake")
    r.add_argument("--pin", help="Rockchip pin name, e.g. GPIO1_B2")
    r.add_argument("--chip", default="/dev/gpiochip0")
    r.add_argument("--line", type=int, default=0)
    r.set_defaults(func=run)
    for name in ("status", "reset"):
        sub.add_parser(name).set_defaults(func=control)

    args = p.parse_args()
    if args.command == "run" and args.gpio == "gpiod" and not args.pin and args.line == 0:
        sys.exit("--gpio gpiod needs --pin (e.g. GPIO1_B2) or --chip/--line")
    args.func(args)


if __name__ == "__main__":
    main()
