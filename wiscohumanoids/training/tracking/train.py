"""Train the Wisco V2 tracking policy with mjlab's own train entry point (same flags as `uv run train ...`)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mjlab_task"))   # wisco_mjlab (robot config, helper A)
sys.path.insert(0, str(HERE))                          # wisco_tracking
import wisco_mjlab  # noqa: E402,F401
import wisco_tracking  # noqa: E402,F401
from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
  main()
