#!/usr/bin/env bash
# One entry point for manipulation fine-tuning: a method on an arm profile (manipulation/profiles/<profile>.yaml).
# The remaining arguments go to the method's trainer, which documents its own flags.
#
#   manipulation/finetune.sh <method> <profile> [trainer arguments]
#
#   method  profiles                       trainer
#   act     panthera_sim                   panthera_sim/train_act.py (LeRobot ACT; closed-loop rollouts in the sim)
#   pi05    panthera_sim, panthera_bimanual finetune/pi05_full.py (full pi0.5 fine-tune through the pinned LeRobot
#                                          trainer; flags are LeRobot's, plus --expect_resume_step=N)
#   groot   g1_dex1                        training/manipulation/groot/finetune.sh <dataset version>
#
# Environment: PYTHON (interpreter with the method's packages; default: panthera_sim/.venv-act/bin/python for act,
# python3 otherwise). pi05 needs LeRobot at the commit in finetune/pi05_full.py and the base weights downloaded.
# Examples:
#   manipulation/finetune.sh act panthera_sim --dataset outputs/lerobot/<name> --output outputs/act/<name> --steps 100000
#   manipulation/finetune.sh pi05 panthera_bimanual --config_path=<ckpt>/train_config.json --resume=true \
#       --dataset.root=<export> --dataset.repo_id=$HF_NAMESPACE/<dataset> --expect_resume_step=24000
#   manipulation/finetune.sh groot g1_dex1 v1
#   manipulation/finetune.sh --list
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }

profiles() {
  # name, methods and description of every profile
  python3 - "$HERE/profiles" <<'EOF'
import sys, pathlib, yaml
for p in sorted(pathlib.Path(sys.argv[1]).glob('*.yaml')):
    d = yaml.safe_load(p.read_text())
    print(f"{d['name']:<20} {','.join(d.get('methods', [])):<12} {d.get('description', '')}")
EOF
}

[[ $# -ge 1 && ( "$1" == -h || "$1" == --help ) ]] && usage 0
[[ $# -ge 1 && "$1" == --list ]] && { profiles; exit 0; }
[[ $# -ge 2 ]] || usage
method="$1" profile="$2"; shift 2
[[ -f "$HERE/profiles/$profile.yaml" ]] || { echo "no profile $profile; profiles:" >&2; profiles >&2; exit 1; }
python3 - "$HERE/profiles/$profile.yaml" "$method" <<'EOF' || exit 1
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
if sys.argv[2] not in d.get('methods', []):
    sys.exit(f"profile {d['name']} supports {d.get('methods', [])}, not {sys.argv[2]}")
EOF

case "$method" in
  act)
    cd "$HERE/panthera_sim"
    PY="${PYTHON:-$( [[ -x .venv-act/bin/python ]] && echo .venv-act/bin/python || echo python3 )}"
    exec "$PY" train_act.py "$@" ;;
  pi05)
    exec "${PYTHON:-python3}" "$HERE/finetune/pi05_full.py" "--profile=$profile" "$@" ;;
  groot)
    exec "$ROOT/training/manipulation/groot/finetune.sh" "$@" ;;
  *) echo "unknown method $method (act, pi05, groot)" >&2; usage ;;
esac
