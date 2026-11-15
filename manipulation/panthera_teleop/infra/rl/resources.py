"""Local, CUDA-free resource checks and a cooperative GPU lease.

No remote calls or torch imports occur here. Acquire ``require_idle_gpu`` before
initializing CUDA, and retain its context throughout the learner's lifetime.
The advisory lock coordinates EXPO processes; unrelated programs do not honor it.
"""
from __future__ import annotations

import csv
import fcntl
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
from contextlib import contextmanager


class ResourceError(RuntimeError):
    """Requested local resources are unavailable or cannot be verified."""


def _query(fields: str, kind: str = "gpu") -> list[list[str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResourceError(f"Cannot verify NVIDIA resources: {exc}") from exc
    return [[cell.strip() for cell in row] for row in csv.reader(result.stdout.splitlines()) if row]


def _selected_gpu(device: str) -> dict:
    if device == "cuda":
        ordinal = 0
    elif device.startswith("cuda:") and device[5:].isdigit():
        ordinal = int(device[5:])
    else:
        raise ResourceError("Device must be cpu, cuda, or cuda:<ordinal>")
    rows = _query("index,uuid,name,memory.total,memory.free,utilization.gpu")
    try:
        gpus = [dict(index=int(r[0]), uuid=r[1], name=r[2], total_gib=float(r[3]) / 1024,
                     free_gib=float(r[4]) / 1024, utilization_percent=int(r[5])) for r in rows]
    except (ValueError, IndexError) as exc:
        raise ResourceError("Invalid NVIDIA GPU query response") from exc
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if len(gpus) > 1 and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        # CUDA's default FASTEST_FIRST ordering can differ from nvidia-smi.
        if visible is None or not all(v.strip().startswith("GPU-") for v in visible.split(",")):
            raise ResourceError("Multiple GPUs: use UUID CUDA_VISIBLE_DEVICES or CUDA_DEVICE_ORDER=PCI_BUS_ID")
    if visible is not None:
        identifiers = [v.strip() for v in visible.split(",") if v.strip()]
        selected = []
        for identifier in identifiers:
            matches = [g for g in gpus if str(g["index"]) == identifier or g["uuid"] == identifier]
            if len(matches) != 1:
                raise ResourceError(f"Cannot resolve CUDA_VISIBLE_DEVICES entry {identifier!r}")
            selected.append(matches[0])
        gpus = selected
    if ordinal >= len(gpus):
        raise ResourceError(f"GPU ordinal {ordinal} is unavailable")
    gpu = gpus[ordinal]
    processes = []
    for row in _query("gpu_uuid,pid,process_name,used_gpu_memory", "compute-apps"):
        try:
            if row[0] == gpu["uuid"]:
                processes.append({"pid": int(row[1]), "name": row[2], "used_mib": row[3]})
        except (ValueError, IndexError) as exc:
            raise ResourceError("Invalid NVIDIA process query response") from exc
    gpu["compute_processes"] = processes
    return gpu


def assess_resources(device: str = "cuda:0", minimum_free_gib: float = 2.0,
                     disk_path: str | Path | None = None, minimum_disk_gib: float = 1.0) -> dict:
    """Return a JSON-safe local report without initializing CUDA or allocating VRAM."""
    if minimum_free_gib < 0 or minimum_disk_gib < 0:
        raise ValueError("Minimum resource requirements cannot be negative")
    report = {"host": socket.gethostname(), "pid": os.getpid(), "device": device,
              "scope": "local", "allowed": True, "reasons": [], "gpu": None}
    if device != "cpu":
        try:
            gpu = report["gpu"] = _selected_gpu(device)
            external = [p for p in gpu["compute_processes"] if p["pid"] != os.getpid()]
            if external:
                report["reasons"].append("GPU has existing compute processes: " + ", ".join(str(p["pid"]) for p in external))
            if gpu["free_gib"] < minimum_free_gib:
                report["reasons"].append(f"GPU needs {minimum_free_gib:.2f} GiB free; has {gpu['free_gib']:.2f}")
        except ResourceError as exc:
            report["reasons"].append(str(exc))
    path = Path(disk_path or Path.cwd()).expanduser().resolve()
    while not path.exists() and path.parent != path:
        path = path.parent
    try:
        usage = shutil.disk_usage(path)
        report["disk"] = {"path": str(path), "free_gib": usage.free / 2**30,
                          "total_gib": usage.total / 2**30}
        if report["disk"]["free_gib"] < minimum_disk_gib:
            report["reasons"].append(f"Disk needs {minimum_disk_gib:.2f} GiB free")
    except OSError as exc:
        report["reasons"].append(f"Cannot verify disk: {exc}")
    report["allowed"] = not report["reasons"]
    return report


@contextmanager
def require_idle_gpu(device: str = "cuda:0", minimum_free_gib: float = 2.0,
                     lock_dir: str | Path | None = None, disk_path: str | Path | None = None,
                     minimum_disk_gib: float = 1.0):
    """Reject other compute PIDs; hold a nonblocking per-physical-GPU flock.

    Only this process's existing CUDA context is permitted. CPU mode still checks
    disk capacity. A fresh NVIDIA query follows lock acquisition to close the
    race between cooperating EXPO learners.
    """
    report = assess_resources(device, minimum_free_gib, disk_path, minimum_disk_gib)
    if not report["allowed"]:
        raise ResourceError("; ".join(report["reasons"]))
    if device == "cpu":
        yield report
        return
    directory = Path(lock_dir or tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"panthera-expo-{report['gpu']['uuid']}.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResourceError(f"GPU already leased by another EXPO process: {path}") from exc
        try:
            report = assess_resources(device, minimum_free_gib, disk_path, minimum_disk_gib)
            if not report["allowed"]:
                raise ResourceError("; ".join(report["reasons"]))
            report["lock_path"] = str(path)
            yield report
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
