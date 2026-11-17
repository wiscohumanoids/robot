#!/usr/bin/env python3
"""Full pi0.5 fine-tune for the simulated arm: manipulation/finetune/pi05_full.py with the `panthera_sim` profile.

    python tools/train_pi05_full.py <lerobot_train flags>      (same as manipulation/finetune.sh pi05 panthera_sim)

Kept so existing commands and `evaluate_pi05.py` (which loads checkpoints with the same strict loader and checks)
work unchanged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "finetune"))
import pi05_full  # noqa: E402
from pi05_full import strict_from_pretrained  # noqa: E402  (re-exported for evaluate_pi05.py)

__all__ = ["strict_from_pretrained", "check_full_model", "CAMERAS"]

PROFILE = pi05_full.load_profile("panthera_sim")
CAMERAS = PROFILE["cameras"]


def check_full_model(policy):
    """pi05_full.check_full_model against the panthera_sim profile."""
    return pi05_full.check_full_model(policy, PROFILE)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _, step, rest = pi05_full.split_args(["--profile=panthera_sim", *sys.argv[1:]])
    sys.argv = [sys.argv[0], *rest]
    pi05_full.install_guards(PROFILE, step).main()
