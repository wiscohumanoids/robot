#!/usr/bin/env bash
# The robot's internal network: the Jetson (central computer) and the RK3588 (real-time computer) on one Ethernet
# cable, with ROS 2 (DDS) kept on that cable and the RK3588's clock following the Jetson's. README.md explains it.
#
#   infra/robot_network/robot_network.sh plan  <jetson|rk3588> [--iface IFACE]   print every file it would write
#   sudo infra/robot_network/robot_network.sh apply <jetson|rk3588> --iface IFACE  write them (netplan, chrony, ROS env)
#   infra/robot_network/robot_network.sh check <jetson|rk3588>                     link, clock offset, ROS discovery
#
# Addresses come from the repo's .env (ROBOT_JETSON_IP, ROBOT_RK3588_IP, ROBOT_LINK_PREFIX, ROBOT_ROS_DOMAIN_ID);
# --iface is this computer's port for the robot cable (e.g. eth1; `ip -br link` lists them).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# the repo .env, without overriding values already set in the environment
VARS="ROBOT_JETSON_IP ROBOT_RK3588_IP ROBOT_LINK_PREFIX ROBOT_ROS_DOMAIN_ID"
for v in $VARS; do eval "_keep_$v=\"\${$v:-}\""; done
# shellcheck disable=SC1091
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
for v in $VARS; do eval "if [ -n \"\$_keep_$v\" ]; then $v=\"\$_keep_$v\"; fi"; done

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
CMD="${1:-}"; ROLE="${2:-}"; shift 2 2>/dev/null || usage
IFACE=""
while [ $# -gt 0 ]; do case "$1" in --iface) IFACE="$2"; shift 2 ;; *) usage ;; esac; done
[ "$ROLE" = jetson ] || [ "$ROLE" = rk3588 ] || usage

: "${ROBOT_JETSON_IP:?set ROBOT_JETSON_IP in the repo .env (see .env.example)}"
: "${ROBOT_RK3588_IP:?set ROBOT_RK3588_IP in the repo .env (see .env.example)}"
PREFIX="${ROBOT_LINK_PREFIX:-24}"
DOMAIN="${ROBOT_ROS_DOMAIN_ID:-42}"
if [ "$ROLE" = jetson ]; then SELF="$ROBOT_JETSON_IP"; PEER="$ROBOT_RK3588_IP"; else SELF="$ROBOT_RK3588_IP"; PEER="$ROBOT_JETSON_IP"; fi

netplan_file() {
  # static address on the robot cable; no gateway (the cable is only the two computers)
  cat <<EOF
# written by infra/robot_network/robot_network.sh ($ROLE): the robot cable
network:
  version: 2
  ethernets:
    ${IFACE:-<iface>}:
      dhcp4: false
      addresses: [$SELF/$PREFIX]
      link-local: []
EOF
}

chrony_file() {
  if [ "$ROLE" = jetson ]; then
    # time server for the robot: follows the internet when there is one, keeps serving on its own when there is not
    cat <<EOF
# written by infra/robot_network/robot_network.sh (jetson): serve time to the RK3588 on the robot cable
allow $PEER/32
local stratum 8 orphan
EOF
  else
    # the RK3588 follows the Jetson only, polling fast so it locks within seconds of boot
    cat <<EOF
# written by infra/robot_network/robot_network.sh (rk3588): follow the Jetson's clock
server $PEER iburst minpoll 0 maxpoll 2 prefer
makestep 0.1 -1
EOF
  fi
}

ros_env_file() {
  # ROS 2 Jazzy discovery: only this subnet and the other computer, on the robot cable (Fast DDS allowlist)
  cat <<EOF
# written by infra/robot_network/robot_network.sh ($ROLE): ROS 2 on the robot cable
export ROS_DOMAIN_ID=$DOMAIN
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export ROS_STATIC_PEERS=$PEER
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/etc/wisco/fastdds_robot.xml
EOF
}

fastdds_file() { sed "s/@LINK_IP@/$SELF/" "$HERE/fastdds_robot.xml.in"; }

case "$CMD" in
  plan)
    echo "=== /etc/netplan/60-wisco-robot-link.yaml"; netplan_file
    echo "=== /etc/chrony/conf.d/wisco-robot.conf"; chrony_file
    echo "=== /etc/profile.d/wisco-ros.sh"; ros_env_file
    echo "=== /etc/wisco/fastdds_robot.xml"; fastdds_file ;;
  apply)
    [ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
    [ -n "$IFACE" ] || { echo "--iface: this computer's port for the robot cable (ip -br link)" >&2; exit 1; }
    [ -e "/sys/class/net/$IFACE" ] || { echo "no interface $IFACE" >&2; exit 1; }
    command -v chronyd >/dev/null || apt-get install -y chrony
    netplan_file > /etc/netplan/60-wisco-robot-link.yaml && chmod 600 /etc/netplan/60-wisco-robot-link.yaml
    mkdir -p /etc/chrony/conf.d /etc/wisco
    chrony_file > /etc/chrony/conf.d/wisco-robot.conf
    grep -q '^confdir /etc/chrony/conf.d' /etc/chrony/chrony.conf || echo 'confdir /etc/chrony/conf.d' >> /etc/chrony/chrony.conf
    ros_env_file > /etc/profile.d/wisco-ros.sh
    fastdds_file > /etc/wisco/fastdds_robot.xml
    netplan apply
    systemctl restart chrony
    echo "applied ($ROLE, $IFACE = $SELF/$PREFIX). Log in again for the ROS environment, then: $0 check $ROLE" ;;
  check)
    fails=0
    ok() { printf '  [ok]   %s\n' "$*"; }
    bad() { printf '  [FAIL] %s\n' "$*"; fails=$((fails + 1)); }
    ping -c 3 -W 1 "$PEER" >/dev/null 2>&1 && ok "$PEER answers on the robot cable" || bad "$PEER does not answer (cable, address)"
    if [ "$ROLE" = rk3588 ]; then
      off=$(chronyc -c tracking 2>/dev/null | awk -F, '{print $5}')      # system time offset, s
      ref=$(chronyc -c tracking 2>/dev/null | awk -F, '{print $2}')
      if [ -n "$off" ] && awk -v o="$off" 'BEGIN{exit !(o < 0.001 && o > -0.001)}'; then
        ok "clock follows $ref, offset $(awk -v o="$off" 'BEGIN{printf "%.1f", o*1e6}') us"
      else
        bad "clock offset ${off:-unknown} s (chronyc tracking; wants < 1 ms)"
      fi
    else
      chronyc -c clients 2>/dev/null | grep -q "$PEER" && ok "serving time to $PEER" || bad "$PEER has not asked for time yet (chronyc clients)"
    fi
    [ "${ROS_DOMAIN_ID:-}" = "$DOMAIN" ] && ok "ROS_DOMAIN_ID $DOMAIN" || bad "ROS_DOMAIN_ID is '${ROS_DOMAIN_ID:-}' (log in again after apply)"
    if command -v ros2 >/dev/null; then
      ros2 node list --no-daemon 2>/dev/null | grep -q . && ok "ROS 2 nodes visible: $(ros2 node list --no-daemon 2>/dev/null | wc -l)" \
        || bad "no ROS 2 nodes visible (is the other computer's stack running?)"
    fi
    [ "$fails" = 0 ] && echo "all checks passed" || { echo "$fails check(s) failed"; exit 1; } ;;
  *) usage ;;
esac
