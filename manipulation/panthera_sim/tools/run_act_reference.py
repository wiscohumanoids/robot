#!/usr/bin/env python3
"""Run the reference benchmark through our trainer, then evaluate its best model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/act/reference_transfer_cube")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "benchmark_status.json"
    if not args.resume and (args.output / "training_settings.json").exists():
        parser.error("output already contains a run; use --resume or a fresh --output")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MUJOCO_GL="egl", PYTHONUNBUFFERED="1")
    train = [sys.executable, str(ROOT / "train_act.py"), "--reference-act", "--output", str(args.output),
             "--steps", str(args.steps), "--full-val-at-end"]
    if args.resume:
        train += ["--resume", str(args.resume.resolve())]
    for name in ("workers", "prefetch_factor"):
        if getattr(args, name) is not None:
            train += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    for name in ("amp", "cudnn_benchmark", "channels_last"):
        if getattr(args, name) is not None:
            train += ["--" + ("" if getattr(args, name) else "no-") + name.replace("_", "-")]
    evaluate = [str(ROOT / ".venv-act-sim/bin/python"), str(ROOT / "tools/evaluate_act_reference.py"),
                "--checkpoint", str(args.output / "best"), "--output", str(args.output / "evaluation"),
                "--episodes", str(args.eval_episodes), "--video"]
    (args.output / "commands.json").write_text(json.dumps({"train": train, "evaluate": evaluate}, indent=2) + "\n")
    def status(stage, **extra):
        status_path.write_text(json.dumps({"status": stage, "pid": os.getpid(), "updated_at_unix": time.time(), **extra}, indent=2) + "\n")
    try:
        for stage, command in (("training", train), ("evaluating", evaluate)):
            status(stage)
            with (args.output / f"{stage}.log").open("a") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        status("complete")
    except BaseException as exc:
        status("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
