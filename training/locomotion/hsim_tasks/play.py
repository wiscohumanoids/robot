"""Play / record an hsim task policy with mjlab's own player (same flags as `uv run play ...`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hsim_tasks  # noqa: E402,F401  (registers tasks)
from mjlab.scripts.play import main  # noqa: E402

if __name__ == "__main__":
  main()
