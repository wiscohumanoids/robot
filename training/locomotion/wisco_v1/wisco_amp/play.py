"""Play an AMP-trained checkpoint with mjlab.

The AMP dataset path is still required because mjlab's play path constructs the
registered AMP runner before loading the actor.

  uv run python wisco_v1/wisco_amp/play.py Wisco-V1-AMP-Velocity-Flat \
      --checkpoint-file <path> --agent.dataset.amp-data-path ~/hsim_data/amp
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wisco_amp  # noqa: E402,F401  (registers the task)
from mjlab.scripts.play import main  # noqa: E402

if __name__ == "__main__":
  main()
