"""Train the Wisco V2 walking policy with mjlab's own trainer.

Run from training/locomotion (the uv project that pins mjlab 1.6.0) on the GPU VM:
  uv run python ../../wiscohumanoids/training/mjlab_task/train.py Wisco-V2-Velocity-Flat \
      --env.scene.num-envs 4096 --agent.max-iterations 2000 --log-root logs/rsl_rl
Same flags as `uv run train ...` for the G1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wisco_mjlab  # noqa: E402,F401  (registers the task)
from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
  main()
