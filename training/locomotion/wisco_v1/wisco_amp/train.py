"""Train Wisco V1 walking with amp-rsl-rl.

Run from training/locomotion:

  uv run python wisco_v1/wisco_amp/train.py Wisco-V1-AMP-Velocity-Flat \
      --env.scene.num-envs 512 \
      --agent.dataset.amp-data-path ~/hsim_data/amp \
      \
      --agent.style-weight 0.2
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wisco_amp  # noqa: E402,F401  (registers the task)
from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
  main()
