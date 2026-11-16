# Wireless E-Stop

Handheld ESP32-S3 → Wi-Fi (UDP) → RK3588 → E-stop GPIO → MOSFET → 48 V contactor.

- ESP32 sends `ALIVE <seq>` every 50 ms, or `STOP <seq>` every 20 ms while the button is pressed.
- RK3588 keeps the GPIO **HIGH only while ALIVEs keep arriving**. STOP, or 150 ms of silence
  (dead battery, crash, out of range), drives it LOW and **latches** until someone runs `reset`.

**Status (2026-11-03):** tested end to end on an ESP32-S3 dev board → phone hotspot → laptop
(receiver with printed GPIO; a jumper GPIO 4–GND as the NC button). Not yet tested on an RK3588;
`GpiodOutput` (the real pin) is untested.

| Test | Result |
|---|---|
| Arm after 5 heartbeats + `reset` | power on |
| Pull jumper (button pressed) | cut + latched immediately on first STOP |
| `reset` while pulled | refused ("release the E-stop button first") |
| Reinsert jumper, `reset` | power on |
| Unplug ESP32 USB | cut + latched after 153 ms |
| ESP32 reboots and reconnects | link healthy, power stays off until `reset` |
| Packet loss (~9,200 packets, ~7 min) | 2 lost |

**Nuisance trips:** with ESP32 → phone hotspot → laptop (two Wi-Fi hops), heartbeats sometimes
arrived 150+ ms late and then in a burst (ESP32 stayed connected; packets were delayed, not lost).
That tripped the 150 ms timeout about once every 1–2 minutes. It fails safe (stops), but the
timeout needs re-checking on the real setup, where the RK3588 is the access point (one hop).

## Layout

```
esp32_transmitter/   ESP-IDF project (C): main/main.c, settings in main/Kconfig.projbuild
receiver/            estop_receiver.py (runs on the RK3588), estop-receiver.service (systemd)
tools/               fake_sender.py: laptop stand-in for the ESP32
```

## ESP32-S3

Wire a **normally-closed** button between **GPIO 4** and **GND** (internal pull-up on).
Pressed or wire cut → pin reads HIGH → STOP.

```bash
cd esp32_transmitter
idf.py set-target esp32s3
idf.py menuconfig          # "Wireless E-Stop Transmitter": SSID, password, receiver IP, GPIO
idf.py -p <port> flash monitor
```

Defaults: SSID `EstopNet`, receiver `10.42.0.1:4242` (the nmcli hotspot address).

## RK3588

```bash
sudo nmcli device wifi hotspot ifname wlan0 ssid EstopNet password "<password>"
sudo nmcli connection modify Hotspot connection.autoconnect yes 802-11-wireless.powersave 2
pip install "gpiod>=2"

cd receiver
sudo python3 estop_receiver.py run --gpio gpiod --pin GPIO1_B2   # use the real pin
python3 estop_receiver.py status
python3 estop_receiver.py reset    # arm / reset; refused unless the link is healthy
```

`--pin` takes Rockchip names (`GPIO1_B2` → gpiochip1 line 10); same on RK3588 and RK3588S.
To start on boot: edit and install `receiver/estop-receiver.service`.

## Test on a laptop

```bash
python3 receiver/estop_receiver.py run        # terminal 1: GPIO is printed
python3 tools/fake_sender.py                  # terminal 2: s = button, k = kill, l = 50% loss
python3 receiver/estop_receiver.py reset      # terminal 3
```

With a real ESP32: put the laptop's IP and a shared 2.4 GHz network in menuconfig.

## Things to check

These came up while writing and reviewing the code. None has been confirmed on hardware,
so treat them as questions rather than findings.

1. **Pin state after a hard crash.** On Ctrl+C / SIGTERM the receiver drives the pin LOW first.
   After SIGKILL or a kernel hang, Linux may not guarantee that, so the pin could stay HIGH.
   A hardware watchdog (e.g. gate held on only by a toggling signal) would cover this, if needed.
2. **Schematic:** the on-robot button and the GPIO both seem to feed the MOSFET gate through
   100 Ω (from a low-res image). If that is a parallel (OR) connection, GPIO HIGH might keep
   the gate on with the button pressed. Intended behavior: either one alone cuts power.
3. A short in the button wiring would read as "not pressed" (NC catches a cut wire, not a short).
4. 50 / 150 ms timing on the real RK3588 hotspot (see "Nuisance trips" above).
5. The chosen pin's state during boot, before the service starts.
