#!/usr/bin/env python3
"""Package the exact local simulator and pi05 scripts for the remote notebook."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def simulator_meshes(root):
    # The MJCF references both visual and collision meshes. Keep the complete
    # tree, including nested asset directories, rather than only render geometry.
    return sorted(p for p in (root / "sim/panthera/meshes").rglob("*") if p.is_file())


def validate_runtime_archive(archive_path):
    """Compile from a fresh extraction with no project checkout on sys.path."""
    with tempfile.TemporaryDirectory(prefix="pi05-runtime-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(root)
        manifest = json.loads((root / "runtime_manifest.json").read_text())
        for name, expected in manifest["files"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Runtime hash mismatch: {name}")
        subprocess.run([sys.executable, "-I", "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from sim.panthera_env import PantheraSim; "
            "sim=PantheraSim(); "
            "assert sim.object_names == ['cube_red', 'cube_green', 'cube_blue']; "
            "print(f'Isolated runtime compiled: {sim.model.nmesh} meshes, {len(sim.object_names)} blocks')",
            str(root)], cwd=root, check=True)


def main():
    provenance = json.loads((ROOT / "outputs/lerobot/panthera_scripted_stack_30hz/meta/provenance.json").read_text())
    expected = next(iter(provenance["sources"].values()))["render_signature"]["implementation"]
    for name in ("sim/panthera_env.py", "sim/panthera/panthera.xml", "sim/panthera/scene.xml"):
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected[name]:
            raise ValueError(f"Simulator differs from the demonstration export: {name}")
    paths = [ROOT / name for name in (
        "sim/panthera_env.py", "sim/dynamics.py", "sim/stack_task.py", "sim/panthera/scene.xml", "sim/panthera/panthera.xml",
        "teleop/render_vla_dataset.py", "teleop/dataset_contract.py",
        "tools/train_pi05_full.py", "tools/evaluate_pi05.py", "tools/pi05_checkpoint_backup.py")]
    paths += simulator_meshes(ROOT)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    manifest = {"files": hashes, "mujoco_version": "3.13.0", "fps": 30,
                "scene_matches_dataset_provenance": True,
                "cameras": {"shoulder": {"azimuth": 14., "elevation": -34., "distance": 1.2,
                                         "lookat": [.44, 0., .12]}, "wrist": "fixed wrist camera in panthera.xml"},
                "training_seeds": [json.loads(p.read_text())["seed"]
                                   for p in sorted((ROOT / "data/scripted_stack").glob("episode_*/meta.json"))]}
    output = ROOT / "outputs/pi05_artifacts/assets/pi05_runtime.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_file():
                archive.write(path, str(path.relative_to(ROOT)))
        # Explicit packages prevent collisions with other installed `tools`/`sim` modules.
        for package in ("sim", "teleop", "tools"):
            archive.writestr(f"{package}/__init__.py", "")
        archive.writestr("runtime_manifest.json", json.dumps(manifest, indent=2))
    validate_runtime_archive(output)
    print(output, output.stat().st_size, hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
