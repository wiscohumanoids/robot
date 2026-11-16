"""Play (watch / record) a trained Wisco V1 policy with mjlab's own player. Same flags as `uv run play ...`.
  uv run python ../../training/locomotion/wisco_v1/play.py Wisco-V1-Velocity-Flat --checkpoint-file <path>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wisco_mjlab  # noqa: E402,F401  (registers the task)
from mjlab.scripts.play import main  # noqa: E402

if __name__ == "__main__":
  main()
