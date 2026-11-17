#!/usr/bin/env bash
# Real-time setup and checks for the RK3588 (Rockchip), the robot's real-time computer (ros2_control + EtherCAT).
# Run on the RK3588 itself. Steps, in order (README.md in this folder explains each):
#
#   sudo infra/rk3588/setup_rt.sh check                 kernel, boot arguments, limits, governor, irqbalance, NIC
#   sudo infra/rk3588/setup_rt.sh bootargs [--apply]    print (or add) isolcpus / nohz_full / rcu_nocbs / irqaffinity
#   sudo infra/rk3588/setup_rt.sh tune                  rtprio + memlock limits, performance governor, irqbalance off,
#                                                       EtherCAT NIC interrupts pinned to the control core
#   sudo infra/rk3588/setup_rt.sh latency [minutes]     cyclictest under load on the isolated cores; writes a report
#   sudo infra/rk3588/setup_rt.sh rate-check <hz> [s]   with ros2_control running at update_rate:=<hz>: EtherCAT frame
#                                                       loss and working-counter changes over <s> seconds (default 60)
#
# Settings (environment): RT_CORES (isolated cores, default 6-7: the second Cortex-A76 cluster), CONTROL_CORE (the
# core the control loop and the NIC interrupts use, default 7), EC_IFACE (EtherCAT interface; default from
# /etc/ethercat.conf MASTER0_DEVICE, else eth0), RT_USER (user that runs ros2_control_node, default $SUDO_USER).
set -euo pipefail

RT_CORES="${RT_CORES:-6-7}"
CONTROL_CORE="${CONTROL_CORE:-7}"
RT_USER="${RT_USER:-${SUDO_USER:-$(id -un)}}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPORTS="$HERE/reports"

ok()   { printf '  [ok]   %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; FAILS=$((FAILS + 1)); }
warn() { printf '  [warn] %s\n' "$*"; }
need_root() { [[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }; }
FAILS=0

ec_iface() {
  # The EtherCAT interface: EC_IFACE, else the IgH master's configured device (a name or a MAC address), else eth0.
  if [[ -n "${EC_IFACE:-}" ]]; then echo "$EC_IFACE"; return; fi
  local dev=""
  [[ -f /etc/ethercat.conf ]] && dev="$(sed -n 's/^MASTER0_DEVICE="\{0,1\}\([^"]*\)"\{0,1\}/\1/p' /etc/ethercat.conf)"
  if [[ "$dev" =~ ^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$ ]]; then
    for n in /sys/class/net/*; do
      [[ "$(cat "$n/address" 2>/dev/null)" == "${dev,,}" ]] && { basename "$n"; return; }
    done
  fi
  echo "${dev:-eth0}"
}

nic_irqs() {
  # IRQ numbers of an interface (by name in /proc/interrupts, which covers stmmac's per-queue lines too).
  grep -E "[[:space:]]$1(-|$|[[:space:]])" /proc/interrupts | awk -F: '{gsub(/ /, "", $1); print $1}'
}

boot_config() {
  # The file that holds the kernel command line: Armbian (armbianEnv.txt) or extlinux (Ubuntu Rockchip, Radxa, Orange Pi).
  for f in /boot/armbianEnv.txt /boot/extlinux/extlinux.conf /boot/firmware/extlinux/extlinux.conf; do
    [[ -f "$f" ]] && { echo "$f"; return; }
  done
  echo ""
}

rt_args() { echo "isolcpus=nohz,domain,managed_irq,$RT_CORES nohz_full=$RT_CORES rcu_nocbs=$RT_CORES irqaffinity=0-3"; }

cmd_check() {
  echo "== kernel"
  if [[ "$(cat /sys/kernel/realtime 2>/dev/null)" == 1 ]] || uname -v | grep -q PREEMPT_RT; then
    ok "PREEMPT_RT kernel: $(uname -r)"
  else
    bad "not a PREEMPT_RT kernel ($(uname -r), $(uname -v | grep -o 'PREEMPT[_A-Z]*' | head -1)): see README.md, 'Kernel'"
  fi
  echo "== boot arguments ($(boot_config || true))"
  local cl; cl="$(cat /proc/cmdline)"
  for a in isolcpus nohz_full rcu_nocbs; do
    grep -qw "$a=[^ ]*" <<<"$cl" && ok "$(grep -o "$a=[^ ]*" <<<"$cl")" || bad "$a missing: setup_rt.sh bootargs --apply"
  done
  echo "== limits for $RT_USER"
  local rtp mem
  rtp="$(su - "$RT_USER" -c 'ulimit -r' 2>/dev/null || echo 0)"; mem="$(su - "$RT_USER" -c 'ulimit -l' 2>/dev/null || echo 0)"
  [[ "$rtp" == unlimited || "$rtp" -ge 98 ]] && ok "rtprio $rtp" || bad "rtprio $rtp (needs 98): setup_rt.sh tune"
  [[ "$mem" == unlimited ]] && ok "memlock unlimited" || bad "memlock $mem: setup_rt.sh tune"
  echo "== CPU"
  local gov; gov="$(cat /sys/devices/system/cpu/cpu"$CONTROL_CORE"/cpufreq/scaling_governor 2>/dev/null || echo '?')"
  [[ "$gov" == performance ]] && ok "cpu$CONTROL_CORE governor performance" || bad "cpu$CONTROL_CORE governor $gov: setup_rt.sh tune"
  systemctl is-active --quiet irqbalance 2>/dev/null && bad "irqbalance running (moves the NIC interrupts): setup_rt.sh tune" \
    || ok "irqbalance off"
  echo "== EtherCAT interface $(ec_iface)"
  local ifc drv; ifc="$(ec_iface)"
  if [[ -e /sys/class/net/$ifc ]]; then
    drv="$(basename "$(readlink -f /sys/class/net/"$ifc"/device/driver 2>/dev/null)" 2>/dev/null || true)"
    [[ -z "$drv" || "$drv" == . || "$drv" == / ]] && drv='none (virtual interface?)'
    ok "driver $drv (IgH: a native driver for it, or the generic driver: README.md, 'EtherCAT master')"
    for irq in $(nic_irqs "$ifc"); do
      local aff; aff="$(cat /proc/irq/"$irq"/smp_affinity_list)"
      [[ "$aff" == "$CONTROL_CORE" ]] && ok "IRQ $irq on cpu$aff" || bad "IRQ $irq on cpu$aff (want $CONTROL_CORE): setup_rt.sh tune"
    done
  else
    warn "no interface $ifc (set EC_IFACE); the IgH native driver may have taken it over, which is fine"
  fi
  command -v ethercat >/dev/null && ok "IgH tools: $(ethercat version 2>/dev/null | head -1)" || warn "IgH 'ethercat' tool not installed"
  echo; [[ $FAILS -eq 0 ]] && echo "all checks passed" || { echo "$FAILS check(s) failed"; exit 1; }
}

cmd_bootargs() {
  local f; f="$(boot_config)"
  echo "arguments: $(rt_args)"
  [[ "${1:-}" == --apply ]] || { echo "file: ${f:-unknown}. Add them there (or run with --apply), then reboot."; return; }
  need_root
  [[ -n "$f" ]] || { echo "no armbianEnv.txt or extlinux.conf found: add the arguments by hand" >&2; exit 1; }
  grep -q 'isolcpus=' "$f" && { echo "$f already sets isolcpus; edit it by hand" >&2; exit 1; }
  cp "$f" "$f.bak.$(date +%s)"
  if [[ "$f" == */armbianEnv.txt ]]; then
    if grep -q '^extraargs=' "$f"; then sed -i "s|^extraargs=\(.*\)|extraargs=\1 $(rt_args)|" "$f"
    else echo "extraargs=$(rt_args)" >> "$f"; fi
  else
    sed -i "s|^\([[:space:]]*append[[:space:]].*\)|\1 $(rt_args)|" "$f"
  fi
  echo "updated $f (backup next to it); reboot, then setup_rt.sh check"
}

cmd_tune() {
  need_root
  cat > /etc/security/limits.d/90-ros2-control-rt.conf <<EOF
# ros2_control_node real-time scheduling (written by infra/rk3588/setup_rt.sh)
$RT_USER - rtprio 98
$RT_USER - memlock unlimited
EOF
  echo "limits: /etc/security/limits.d/90-ros2-control-rt.conf (log in again to apply)"
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g" 2>/dev/null || true; done
  echo "governor: performance (until reboot; the systemd unit below re-applies it)"
  systemctl disable --now irqbalance 2>/dev/null && echo "irqbalance: disabled" || echo "irqbalance: not installed"
  local ifc; ifc="$(ec_iface)"
  for irq in $(nic_irqs "$ifc"); do
    echo "$CONTROL_CORE" > /proc/irq/"$irq"/smp_affinity_list && echo "IRQ $irq ($ifc) -> cpu$CONTROL_CORE"
  done
  # Re-apply the governor and the IRQ pinning at every boot (the EtherCAT service may re-create the interrupts).
  cat > /etc/systemd/system/rk3588-rt-tune.service <<EOF
[Unit]
Description=Real-time tuning for ros2_control (infra/rk3588/setup_rt.sh)
After=network.target ethercat.service

[Service]
Type=oneshot
Environment=EC_IFACE=$ifc CONTROL_CORE=$CONTROL_CORE RT_CORES=$RT_CORES RT_USER=$RT_USER
ExecStart=$HERE/setup_rt.sh tune-boot

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload && systemctl enable rk3588-rt-tune.service >/dev/null && echo "boot unit: rk3588-rt-tune.service"
}

cmd_tune_boot() {
  # The boot-time part of `tune` (governor and IRQ pinning only).
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g" 2>/dev/null || true; done
  for irq in $(nic_irqs "$(ec_iface)"); do echo "$CONTROL_CORE" > /proc/irq/"$irq"/smp_affinity_list || true; done
}

cmd_latency() {
  need_root
  local minutes="${1:-10}"
  command -v cyclictest >/dev/null || { echo "install rt-tests: sudo apt install rt-tests stress-ng" >&2; exit 1; }
  mkdir -p "$REPORTS"
  local out; out="$REPORTS/cyclictest_$(uname -r)_$(date +%Y%m%d_%H%M).txt"
  echo "cyclictest on cpus $RT_CORES for $minutes min, with stress-ng on cpus 0-5 (if installed)"
  local load=""
  if command -v stress-ng >/dev/null; then
    taskset -c 0-5 stress-ng --cpu 4 --io 2 --vm 1 --vm-bytes 256M --timeout "${minutes}m" >/dev/null 2>&1 & load=$!
  fi
  # -m lock memory, -p80 SCHED_FIFO 80, -i 1000 µs = a 1 kHz control period, -h histogram (max-latency line)
  cyclictest -m -p80 -i1000 -a "$RT_CORES" -t -q -D "${minutes}m" -h 400 --mainaffinity=0 > "$out.hist"
  [[ -n "$load" ]] && wait "$load" 2>/dev/null || true
  {
    echo "kernel: $(uname -r) $(uname -v)"
    echo "cmdline: $(cat /proc/cmdline)"
    echo "run: cyclictest -m -p80 -i1000 -a $RT_CORES -t -D ${minutes}m, load: ${load:+stress-ng on 0-5}"
    grep -E '^# (Min|Avg|Max) Latencies|^# Histogram Overflows' "$out.hist"
  } > "$out"
  cat "$out"
  echo "report: $out (copy the Max line into ros2_ws/src/wisco_ethercat_ros2_hw/README.md, 'Real-time setup')"
}

cmd_rate_check() {
  local hz="${1:?usage: rate-check <hz> [seconds]}" secs="${2:-60}"
  command -v ethercat >/dev/null || { echo "IgH 'ethercat' tool not found" >&2; exit 1; }
  lost() { ethercat master | awk -F: '/Lost frames|Frames lost/ {gsub(/ /, "", $2); print $2; exit}'; }
  local l0 since; l0="$(lost || echo 0)"; since="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "watching $secs s at update_rate:=$hz (ros2_control must be running with it)"
  sleep "$secs"
  local l1 wkc; l1="$(lost || echo 0)"
  wkc="$(journalctl -k --since "$since" 2>/dev/null | grep -c 'Working counter changed' || true)"
  echo "lost frames: $((${l1:-0} - ${l0:-0}))   working-counter changes: $wkc"
  if [[ $((${l1:-0} - ${l0:-0})) -eq 0 && "$wkc" -eq 0 ]]; then
    echo "PASS at $hz Hz: also check controller_manager overrun warnings in its log, then try the next rate"
  else
    echo "FAIL at $hz Hz: stay at the previous rate (README.md, 'Raising update_rate')"; exit 1
  fi
}

case "${1:-}" in
  check) cmd_check ;;
  bootargs) shift; cmd_bootargs "$@" ;;
  tune) cmd_tune ;;
  tune-boot) cmd_tune_boot ;;
  latency) shift; cmd_latency "$@" ;;
  rate-check) shift; cmd_rate_check "$@" ;;
  *) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
