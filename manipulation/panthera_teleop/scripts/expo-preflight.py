#!/usr/bin/env python3
"""Print a local resource/input report. Never starts training or contacts AWS."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infra.rl.resources import assess_resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    parser.add_argument("--minimum-disk-gib", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("work/expo-runs"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Explicit dry-run flag; this command always only inspects")
    args = parser.parse_args()
    report = assess_resources(args.device, args.minimum_free_gib, args.output_dir, args.minimum_disk_gib)
    report["dry_run"] = True
    report["inputs"] = {}
    for name in ("checkpoint", "dataset"):
        path = getattr(args, name)
        if path is None:
            continue
        path = path.expanduser().resolve()
        exists = path.exists()
        report["inputs"][name] = {"path": str(path), "exists": exists,
                                  "kind": "directory" if path.is_dir() else "file" if path.is_file() else "other"}
        if not exists:
            report["reasons"].append(f"{name} does not exist: {path}")
        elif not (path.is_file() or path.is_dir()):
            report["reasons"].append(f"{name} must be a file or directory: {path}")
        else:
            try:
                # Confirm access without unpickling a checkpoint or loading tensors.
                if path.is_dir():
                    with os.scandir(path):
                        pass
                else:
                    with path.open("rb"):
                        pass
            except OSError as exc:
                report["reasons"].append(f"{name} is not readable: {exc}")
    report["allowed"] = not report["reasons"]
    print(json.dumps(report, indent=2))
    return 0 if report["allowed"] else 2


if __name__ == "__main__":
    sys.exit(main())
