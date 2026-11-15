"""Play (watch / record) a trained Wisco V2 policy with mjlab's own player. Same flags as `uv run play ...`.
  uv run python ../../wiscohumanoids/training/mjlab_task/play.py Wisco-V2-Velocity-Flat --checkpoint-file <path>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wisco_mjlab  # noqa: E402,F401  (registers the task)
from mjlab.scripts.play import main  # noqa: E402

if __name__ == "__main__":
  main()
