#!/usr/bin/env python3
"""Download the ACT authors' 50 published transfer-cube episodes losslessly."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import gdown
import h5py
import requests
from bs4 import BeautifulSoup
from gdown.exceptions import FileURLRetrievalError

FOLDER_ID = "1aRyoOhQwxhyt1J8XgEig4s6kzaw__LXj"
SOURCE = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
UPSTREAM_COMMIT = "742c753c0d4a5d87076c8f69e5628c79a8cc5488"


def download_public_file(file_id, raw):
    try:
        gdown.download(id=file_id, output=str(raw), quiet=True, resume=True, retries=3)
        return
    except FileURLRetrievalError:
        # Some Drive warning pages fail gdown's confirmation parsing. Submit
        # the public page's ordinary large-file download form in a fresh session.
        print(f"retrying public download confirmation: {file_id}", flush=True)
    with requests.Session() as session:
        response = session.get("https://drive.google.com/uc",
                               params={"id": file_id, "export": "download"}, timeout=60)
        response.raise_for_status()
        form = BeautifulSoup(response.text, "html.parser").find("form", id="download-form")
        if form is None or form.get("action") != "https://drive.usercontent.google.com/download":
            raise RuntimeError(f"Drive did not provide a public download form for {file_id}")
        params = {field["name"]: field.get("value", "")
                  for field in form.find_all("input", attrs={"name": True})}
        with session.get(form["action"], params=params, stream=True, timeout=(60, 120)) as response:
            response.raise_for_status()
            if "attachment" not in response.headers.get("Content-Disposition", ""):
                raise RuntimeError(f"Drive did not return a downloadable file for {file_id}")
            temporary = raw.with_suffix(".part")
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(1024**2):
                    stream.write(chunk)
            temporary.replace(raw)


def download_episode(item, destination):
    target = destination / item.path
    record_path = target.with_suffix(".source.json")
    if target.exists() and record_path.exists():
        return json.loads(record_path.read_text())
    if shutil.disk_usage(destination).free < 3 * 1024**3:
        raise RuntimeError("Less than 3 GiB free; download stopped")
    raw = target.with_suffix(".download")
    download_public_file(item.id, raw)
    hasher = hashlib.sha256()
    with raw.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    temporary = target.with_suffix(".tmp")
    with h5py.File(raw, "r") as source, h5py.File(temporary, "w") as output:
        output.attrs.update(source.attrs)
        def copy(name, obj):
            if isinstance(obj, h5py.Group):
                copied = output.require_group(name)
            else:
                options = dict(compression="gzip", compression_opts=1, shuffle=True) if obj.ndim else {}
                if obj.ndim == 4:
                    options["chunks"] = (1, *obj.shape[1:])
                copied = output.create_dataset(name, shape=obj.shape, dtype=obj.dtype, **options)
                if obj.ndim:
                    for start in range(0, len(obj), 16):
                        copied[start:start + 16] = obj[start:start + 16]
                else:
                    copied[()] = obj[()]
            copied.attrs.update(obj.attrs)
        source.visititems(copy)
    temporary.replace(target)
    record = {"file": target.name, "google_drive_id": item.id, "original_sha256": digest,
              "original_bytes": raw.stat().st_size, "stored_bytes": target.stat().st_size,
              "transform": "lossless HDF5 gzip; all arrays and attributes preserved"}
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    raw.unlink()
    print(f"downloaded {target.name}: {record['stored_bytes'] / 1024**2:.1f} MiB", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/act_reference/sim_transfer_cube_scripted"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--upstream", type=Path, default=Path("outputs/act_reference/upstream"))
    args = parser.parse_args()
    if not args.upstream.exists():
        args.upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/tonyzhaozh/act.git", str(args.upstream)], check=True)
        subprocess.run(["git", "-C", str(args.upstream), "checkout", "--detach", UPSTREAM_COMMIT], check=True)
    actual = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected upstream commit {UPSTREAM_COMMIT}, found {actual}")
    # These unused interactive-debugger imports require sqlite3, which is absent
    # from this machine's Python build. Physics and data code remain unchanged.
    for name in ("sim_env.py", "utils.py"):
        path = args.upstream / name
        path.write_text(path.read_text().replace("\nimport IPython\ne = IPython.embed\n", "\n"))
    args.output.mkdir(parents=True, exist_ok=True)
    files = [f for f in gdown.download_folder(id=FOLDER_ID, skip_download=True, quiet=True)
             if f.path.endswith(".hdf5")]
    assert {f.path for f in files} == {f"episode_{i}.hdf5" for i in range(50)}
    with ThreadPoolExecutor(args.workers) as pool:
        records = list(pool.map(lambda item: download_episode(item, args.output), files))
    (args.output / "provenance.json").write_text(json.dumps(
        {"source": SOURCE, "task": "sim_transfer_cube_scripted", "upstream_commit": UPSTREAM_COMMIT,
         "episodes": records}, indent=2) + "\n")
    (args.output / "COMPLETE").write_text("50 published episodes downloaded and losslessly compressed\n")


if __name__ == "__main__":
    main()
