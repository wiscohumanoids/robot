"""Train a Wisco V1 walking policy with a clean-room adversarial motion prior.

Run from training/locomotion (the uv project that pins mjlab 1.6.0) on the GPU VM:

  uv run python ../../training/locomotion/wisco_v1/wisco_amp/train.py \
      Wisco-V1-AMP-Velocity-Flat \
      --env.scene.num-envs 512 \
      --agent.amp.motion-files /path/to/walk.npz
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wisco_amp  # noqa: E402,F401  (registers the task)
from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
  main()
