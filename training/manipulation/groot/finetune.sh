#!/usr/bin/env bash
# Fine-tune GR00T on a converted manipulation dataset (run on the GPU machine, directly or as a queued job).
#
#   training/manipulation/groot/finetune.sh <version> [extra launch_finetune.py flags]
#     <version>  dataset version, e.g. v1  -> ~/hsim_data/manip/v1/lerobot
# Env overrides:
#   GROOT=n17|n16|auto (default auto: N1.7 if the gated Cosmos-Reason2-2B backbone is accessible, else N1.6). n16 is the one in use (open weights).
#   MAX_STEPS (20000)  BATCH (global batch, 96)  SAVE_STEPS (5000)  WORKERS (3: 8 vCPUs are shared with RL jobs)
#   DATASET (override path)  OUT (default ~/hsim_runs/groot_finetune_<version>)
# Memory: batch 96 measured ~46-51 GB (N1.6), inside the ~60 GB budget next to the RL jobs (logs/gpu_mem.log).
# Disk: keeps at most 2 checkpoints. N1.7 saves model weights only (--save-only-model; such a run can't be resumed).
#   N1.6 can't skip the optimizer, so the loop below deletes optimizer.pt from all but the newest checkpoint.
#   Measured N1.6 sizes: checkpoint with optimizer 22 GB, without 9 GB, final model (run dir) 9.2 GB
#   -> per run: N1.6 ~40 GB (22 + 9 + 9), N1.7 ~28 GB (estimated: 2 x 9 + 9; same 3B size class).
set -euo pipefail
VERSION="${1:?usage: finetune.sh <version> [extra flags]}"; shift || true
HERE="$(cd "$(dirname "$0")" && pwd)"
ENVS="$HOME/hsim_envs/gr00t"
export HF_HOME="$HOME/hsim_data/hf-cache"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export CUDA_HOME=/usr/local/cuda-12.8
# The VM puts system CUDA 13.2 libs first on LD_LIBRARY_PATH; torch would load that cuDNN (9.23) instead of its own.
unset LD_LIBRARY_PATH
DATASET="${DATASET:-$HOME/hsim_data/manip/$VERSION/lerobot}"
OUT="${OUT:-$HOME/hsim_runs/groot_finetune_$VERSION}"
MAX_STEPS="${MAX_STEPS:-20000}"; BATCH="${BATCH:-96}"; SAVE_STEPS="${SAVE_STEPS:-5000}"; WORKERS="${WORKERS:-3}"

GROOT="${GROOT:-auto}"
if [ "$GROOT" = auto ]; then
  if "$ENVS/Isaac-GR00T/.venv/bin/python" - <<'EOF' 2>/dev/null; then GROOT=n17; else GROOT=n16; fi
from huggingface_hub import hf_hub_download
hf_hub_download("nvidia/Cosmos-Reason2-2B", "config.json")
EOF
fi
if [ "$GROOT" = n17 ]; then REPO="$ENVS/Isaac-GR00T"; BASE=nvidia/GR00T-N1.7-3B; SAVE_FLAGS=(--save-only-model)
else REPO="$ENVS/Isaac-GR00T-n1d6"; BASE=nvidia/GR00T-N1.6-3B; SAVE_FLAGS=(); fi

[ -f "$DATASET/meta/modality.json" ] || { echo "no dataset at $DATASET (run raw_to_lerobot.py first)"; exit 1; }
mkdir -p "$OUT/logs"
echo "GR00T $GROOT ($BASE) | dataset $DATASET | out $OUT | steps $MAX_STEPS batch $BATCH" | tee "$OUT/logs/run_info.txt"
# every 30 s: GPU memory log; drop optimizer.pt from all but the newest checkpoint (disk budget, see top)
( while sleep 30; do
    nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader >> "$OUT/logs/gpu_mem.log"
    ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | head -n -1 | while read c; do rm -f "$c/optimizer.pt"; done
  done ) &
MON=$!; trap 'kill $MON 2>/dev/null || true' EXIT   # || true: the monitor may already be gone, and a failed kill made a finished run exit 1 (job runner then restarted it)

cd "$REPO"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE" \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$HERE/g1_dex1_config.py" \
  --num-gpus 1 \
  --output-dir "$OUT" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit 2 \
  "${SAVE_FLAGS[@]}" \
  --global-batch-size "$BATCH" \
  --dataloader-num-workers "$WORKERS" \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  "$@" 2>&1 | tee -a "$OUT/logs/train.log"
