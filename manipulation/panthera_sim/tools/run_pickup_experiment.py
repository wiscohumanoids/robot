#!/usr/bin/env python3
"""Run a matched first-pickup absolute/relative ACT experiment on one GPU."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop.dataset_contract import DEFAULT_DATASET, file_hash
from tools.act_pickup import build_manifest, load_manifest


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/act/pickup_ablation_20260923")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--selection-episodes", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=64)
    args = parser.parse_args()
    if args.steps < 1 or min(args.selection_episodes, args.test_episodes) < 1:
        parser.error("step and episode budgets must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        parser.error("experiment already initialized; use a new output directory")
    manifest_path = output / "pickup_manifest.json"
    manifest = load_manifest(manifest_path, DEFAULT_DATASET) if manifest_path.exists() else build_manifest(manifest_path)
    protocol = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "hypothesis": "Relative-to-chunk-start arm targets improve first-pickup control over matched absolute targets",
                "representations": ["absolute", "relative"], "initialization": "fresh ACT, same seed and ImageNet backbone",
                "steps_per_arm": args.steps, "batch_size": 12, "chunk_size": 30, "executed_actions": 30,
                "fps": 30, "seed": 20260922, "validation_fraction": .1,
                "selection_seed": 20270101, "selection_episodes": args.selection_episodes,
                "final_test_seed": 20270201, "final_test_episodes": args.test_episodes,
                "episode_seconds": 8, "rollout_eval_frequency": 10000,
                "selection_metric": "same cube held >25mm above table for 0.5 seconds; capped latency breaks ties",
                "action_scaling": "per-representation mean/std over valid training-only pickup chunk targets",
                "pickup_manifest_sha256": file_hash(manifest_path), "pickup_frames": manifest["total_frames"],
                "source_hashes": {name: file_hash(ROOT / name) for name in (
                    "train_act.py", "rollout_act.py", "tools/act_pickup.py", "tools/evaluate_act.py",
                    "tools/lerobot_image_cache.py", "tools/run_pickup_experiment.py")}}
    write_json(output / "protocol.json", protocol)

    def status(phase, **extra):
        write_json(output / "run_status.json", {"phase": phase, "pid": os.getpid(),
                   "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra})

    def run(command, log_path, phase, representation):
        with log_path.open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status(phase, representation=representation, child_pid=child.pid, command=command, log=str(log_path))
            code = child.wait()
        if code:
            raise RuntimeError(f"{phase}/{representation} exited {code}; inspect {log_path}")

    summaries = {}
    try:
        for representation in protocol["representations"]:
            arm = output / representation
            arm.mkdir()
            command = [sys.executable, str(ROOT / "train_act.py"), "--dataset", str(DEFAULT_DATASET),
                       "--output", str(arm), "--pickup-manifest", str(manifest_path),
                       "--action-representation", representation, "--selection-objective", "pickup",
                       "--steps", str(args.steps), "--batch-size", "12", "--workers", "4",
                       "--seed", "20260922", "--chunk-size", "30", "--action-steps", "30",
                       "--rollout-action-steps", "30", "--checkpoint-freq", "10000", "--keep-checkpoints", "2",
                       "--eval-freq", "2000", "--val-batches", "0", "--early-stop-patience", "0",
                       "--rollout-eval-freq", "10000", "--rollout-eval-episodes", str(args.selection_episodes),
                       "--rollout-eval-seconds", "8", "--rollout-seed", "20270101", "--log-freq", "100"]
            run(command, arm / "train.log", "training", representation)
            command = [sys.executable, str(ROOT / "tools/evaluate_act.py"), "--checkpoint", str(arm / "best_rollout"),
                       "--output", str(arm / "final_test"), "--episodes", str(args.test_episodes),
                       "--seed", "20270201", "--seconds", "8", "--save-failures"]
            run(command, arm / "final_test.log", "final_test", representation)
            evaluation = json.loads((arm / "final_test/summary.json").read_text())
            selection = json.loads((arm / "best_rollout/selection.json").read_text())
            validation_rows = [json.loads(line) for line in (arm / "validation.jsonl").read_text().splitlines()]
            selected_validation = next(row for row in validation_rows if row["step"] == selection["step"])
            summaries[representation] = {"rates": evaluation["rates"],
                                         "selection": selection, "selected_checkpoint_validation": selected_validation,
                                         "training": json.loads((arm / "experiment.json").read_text())}
            write_json(output / "comparison.json", summaries)
        # Compare matched final-test scenes; never choose another checkpoint on these seeds.
        absolute = json.loads((output / "absolute/final_test/summary.json").read_text())["records"]
        relative = json.loads((output / "relative/final_test/summary.json").read_text())["records"]
        paired = {"both": 0, "absolute_only": 0, "relative_only": 0, "neither": 0}
        for a, r in zip(absolute, relative):
            assert a["seed"] == r["seed"]
            a, r = a["milestones"]["sustained_pickup"], r["milestones"]["sustained_pickup"]
            key = "both" if a and r else "absolute_only" if a else "relative_only" if r else "neither"
            paired[key] += 1
        summaries["paired_final_test"] = paired
        write_json(output / "comparison.json", summaries)
        status("complete", comparison=str(output / "comparison.json"))
    except BaseException as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
