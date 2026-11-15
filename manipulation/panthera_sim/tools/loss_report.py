#!/usr/bin/env python3
"""Aggregate training losses from a VLA-Adapter fine-tuning run.

Reads either an offline W&B run directory (the default when the notebook's
WANDB_ENTITY is empty) or an online run, prints a rolling-window summary and
optionally writes the raw series to CSV.

    python tools/loss_report.py VLA-Adapter/wandb/latest-run
    python tools/loss_report.py entity/robot-arm-learning-panthera/ab12cd34
    python tools/loss_report.py <run> --window 50 --csv losses.csv

The offline reader only needs the `wandb` package; nothing is uploaded and no
login is required. To pull the run off Colab first:

    from google.colab import files
    !tar czf run.tar.gz /content/RobotArmLearning/VLA-Adapter/wandb/latest-run/
    files.download("run.tar.gz")
"""

import argparse
import csv
import json
import math
import sys
from collections import deque
from pathlib import Path


def read_offline_run(path):
    """Yield (step, metrics) from a run's .wandb transaction log.

    The log is wandb's own append-only datastore, so it can be read while
    training is still writing to it -- a truncated final record just ends the
    scan early.
    """
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob("*.wandb"))
        if not candidates:
            raise SystemExit(f"No .wandb transaction log under {path}")
        path = candidates[0]

    store = DataStore()
    store.open_for_scan(str(path))
    while True:
        try:
            data = store.scan_data()
        except Exception:  # A partial record at the tail of a live run.
            break
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)
        # A logged step arrives as a HistoryRecord or, under wandb-core, as a
        # partial-history request; both carry the same item list.
        for history in (record.history, record.request.partial_history):
            if not history.item:
                continue
            metrics = {}
            for item in history.item:
                key = item.key or ".".join(item.nested_key)
                try:
                    value = json.loads(item.value_json)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[key] = float(value)
            step = metrics.pop("_step", history.step.num if history.HasField("step") else None)
            if metrics:
                yield (int(step) if step is not None else None), metrics


def read_online_run(run_path):
    """Yield (step, metrics) from an `entity/project/run_id` run on wandb.ai."""
    import wandb

    run = wandb.Api().run(run_path)
    for row in run.scan_history():
        step = row.get("_step")
        metrics = {k: float(v) for k, v in row.items()
                   if not k.startswith("_") and isinstance(v, (int, float))
                   and not isinstance(v, bool) and not math.isnan(float(v))}
        if metrics:
            yield (int(step) if step is not None else None), metrics


def rolling(values, window):
    """Trailing mean over the last `window` values, one output per input."""
    recent = deque(maxlen=window)
    for value in values:
        recent.append(value)
        yield sum(recent) / len(recent)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="Offline run directory or .wandb file, "
                                    "or an online 'entity/project/run_id'")
    parser.add_argument("--window", type=int, default=20,
                        help="Rolling window, in logged points (default: 20)")
    parser.add_argument("--rows", type=int, default=25,
                        help="Rows to print, sampled evenly (default: 25; 0 for all)")
    parser.add_argument("--metric", default="Loss",
                        help="Substring matching the metric to aggregate (default: Loss)")
    parser.add_argument("--csv", type=Path, help="Write every logged point here")
    args = parser.parse_args()

    source = Path(args.run)
    if source.exists():
        history = list(read_offline_run(source))
    elif args.run.count("/") == 2:
        history = list(read_online_run(args.run))
    else:
        raise SystemExit(f"{args.run} is neither an existing path nor 'entity/project/run_id'")

    if not history:
        raise SystemExit("No logged history yet. W&B flushes a few seconds behind training.")

    all_keys = sorted({key for _, metrics in history for key in metrics})
    matches = [key for key in all_keys if args.metric.lower() in key.lower()]
    if not matches:
        raise SystemExit(f"No metric matching {args.metric!r}. Logged: {', '.join(all_keys)}")
    key = matches[0]
    if len(matches) > 1:
        print(f"# {len(matches)} metrics match {args.metric!r}; using {key}", file=sys.stderr)

    points = [(step, metrics[key]) for step, metrics in history if key in metrics]
    steps = [step for step, _ in points]
    losses = [loss for _, loss in points]
    smoothed = list(rolling(losses, args.window))

    stride = max(1, len(points) // args.rows) if args.rows else 1
    print(f"{key}  ({len(points)} points, steps {steps[0]}-{steps[-1]}, "
          f"window {args.window})\n")
    print(f"{'step':>8}  {'value':>10}  {'rolling':>10}")
    for i in range(0, len(points), stride):
        print(f"{steps[i]:>8}  {losses[i]:>10.4f}  {smoothed[i]:>10.4f}")
    if (len(points) - 1) % stride:
        print(f"{steps[-1]:>8}  {losses[-1]:>10.4f}  {smoothed[-1]:>10.4f}")

    tail = losses[-args.window:]
    head = losses[:args.window]
    print(f"\nfirst {len(head)}: {sum(head) / len(head):.4f}"
          f"   last {len(tail)}: {sum(tail) / len(tail):.4f}"
          f"   min: {min(losses):.4f} @ step {steps[losses.index(min(losses))]}"
          f"   overall: {sum(losses) / len(losses):.4f}")

    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step"] + all_keys)
            for step, metrics in history:
                writer.writerow([step] + [metrics.get(k, "") for k in all_keys])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
