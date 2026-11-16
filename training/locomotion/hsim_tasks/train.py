"""Train an hsim task with mjlab's own trainer (same flags as `uv run train ...`).
  python hsim_tasks/train.py Hsim-G1-Velocity-Flat-ArmsFree --env.scene.num-envs 4096 --log-root <dir>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hsim_tasks  # noqa: E402,F401  (registers tasks)
from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
  main()
