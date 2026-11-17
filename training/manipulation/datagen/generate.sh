#!/usr/bin/env bash
# Generate manipulation demos on the VM (scripted sources -> MimicGen), successes only.
#   training/manipulation/datagen/generate.sh <version> <n_episodes> [workers=3]
# Scene: ~/hsim_data/manip/scenes/<scene>_<version>.mjb (compiled on the Mac by sim/scene.py, then copied over)
# Output: ~/hsim_data/manip/<version>/raw/episode_XXXXXX.{npz,json}, progress.jsonl, stats.json
set -euo pipefail
VERSION=${1:?usage: generate.sh <version> <n_episodes> [workers]}
N=${2:?usage: generate.sh <version> <n_episodes> [workers]}
W=${3:-3}
SCENE=${SCENE:-kitchen_l1_s1}
cd "$(dirname "$0")"
unset LD_LIBRARY_PATH                   # system CUDA libs first on this VM's default path; not needed here
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1
PY=${HSIM_DATAGEN_PY:-$HOME/hsim_envs/datagen/bin/python}
MJB=${MJB:-$HOME/hsim_data/manip/scenes/${SCENE}_${VERSION}.mjb}
OUT=${OUT:-$HOME/hsim_data/manip/$VERSION}
echo "datagen: $VERSION, $N episodes, $W workers, scene $MJB -> $OUT"
exec "$PY" generate.py --mjb "$MJB" --out "$OUT" --episodes "$N" --workers "$W"
