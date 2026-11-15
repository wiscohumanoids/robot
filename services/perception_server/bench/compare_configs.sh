#!/usr/bin/env bash
# Start the server with each config (tmux hsim-E-test, port 8101 so it never clashes with S1), score it, stop it.
# VM:  bash bench/compare_configs.sh   -> ~/hsim_runs/S1_perception/compare/<config>/report.json
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="env -u LD_LIBRARY_PATH $HOME/hsim_envs/perception/bin/python"   # torch's own cuDNN, not the system CUDA 13.2 one
OUT=$HOME/hsim_runs/S1_perception/compare; mkdir -p "$OUT"
run() {  # name, server args..., -- eval args...
  local name=$1; shift
  local sargs=() eargs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do sargs+=("$1"); shift; done; shift; eargs=("$@")
  tmux kill-session -t hsim-E-test 2>/dev/null
  tmux new-session -d -s hsim-E-test "cd $HERE && HF_HOME=\$HOME/hsim_data/hf-cache HF_HUB_OFFLINE=1 $PY -m perception_server.server --backend real --port 8101 ${sargs[*]} > $OUT/$name.server.log 2>&1"
  for i in $(seq 1 60); do sleep 2; curl -s localhost:8101/health | grep -q '"ok"' && break; done
  mkdir -p "$OUT/$name"
  $PY "$HERE/bench/eval_server.py" --frames $HOME/hsim_data/perc_bench --url http://127.0.0.1:8101 --overlays 4 --out "$OUT/$name" "${eargs[@]}" > /dev/null 2>&1
  python3 -c "import json;r=json.load(open('$OUT/$name/report.json'));c=r['precision_vs_min_score'];print('$name', json.dumps({k:r[k] for k in ('small_objects','fixtures','precision','detections')}), 'prio-prec', c['priority@0.25'], c['priority@0.35'], 'lat', r['latency_ms']['server_median'], 'gpu', r['server']['gpu_mem_peak_gb'])"
  tmux kill-session -t hsim-E-test 2>/dev/null
}
run ens30_noprio  --dino IDEA-Research/grounding-dino-base,IDEA-Research/grounding-dino-tiny --box-threshold 0.30 --priority-mode none -- --vocab lean --priority
run ens35_noprio  --dino IDEA-Research/grounding-dino-base,IDEA-Research/grounding-dino-tiny --box-threshold 0.35 --priority-mode none -- --vocab lean --priority
run tiny30_noprio --dino IDEA-Research/grounding-dino-tiny --box-threshold 0.30 --priority-mode none -- --vocab lean --priority
