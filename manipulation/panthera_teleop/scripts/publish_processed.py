#!/usr/bin/env python3
"""Publish a verified processed dataset, then verify the remote release."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main(root, update_existing=False):
    verification = json.loads((root / "verification.json").read_text())
    assert verification["passed"]
    assert json.loads((root / "processing/full_rate_verification.json").read_text())["passed"]
    manifest = json.loads((root / "processing/manifest.json").read_text())
    repo_id = manifest["repo_id"]
    api = HfApi()
    assert api.whoami()["name"] == repo_id.split("/")[0]
    # Avoid replacing any existing user dataset. Create privately until upload
    # and remote checks are complete, then publish the authorized release.
    parent_commit = None
    delete_patterns = None
    if update_existing:
        existing = api.dataset_info(repo_id)
        parent_commit = existing.sha
        url = f"https://huggingface.co/datasets/{repo_id}"
        # Preserve the previous release before updating the format tag used by
        # native LeRobot loaders. Guard the upload against concurrent changes.
        api.create_tag(repo_id, repo_type="dataset", tag=f"snapshot-{parent_commit[:12]}",
                       revision=parent_commit, exist_ok=True)
        local_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        remote_files = api.list_repo_files(repo_id, repo_type="dataset", revision=parent_commit)
        delete_patterns = [name for name in remote_files
                           if name.startswith(("data/", "meta/", "processing/alignment/"))
                           and name not in local_files]
        print(f"Updating {url} from {parent_commit}", flush=True)
    else:
        url = api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=False)
        print(f"Created staging dataset {url}", flush=True)
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=root,
        allow_patterns=["README.md", "crop_preview.jpg", "verification.json",
                        "data/**", "meta/**", "processing/**", "trimmed_full_rate/**"],
        ignore_patterns=["**/__pycache__/**", "**/*.pyc", "**/.cache/**"],
        parent_commit=parent_commit, delete_patterns=delete_patterns,
        commit_message=f"Publish {len(manifest['episodes'])} contact-trimmed Panthera demos, 20 Hz LeRobot v3")
    revision = commit.oid
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    expected = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                and p.parts[len(root.parts)] in
                ("README.md", "crop_preview.jpg", "verification.json", "data", "meta", "processing", "trimmed_full_rate")
                and "__pycache__" not in p.parts and p.suffix != ".pyc" and ".cache" not in p.parts]
    assert not (set(expected) - set(files)), sorted(set(expected) - set(files))
    assert not {name for name in files if name.startswith(("data/", "meta/", "processing/alignment/"))
                and name not in expected}, "Obsolete training files remain on the remote"
    lfs_checks = 0
    remote_info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
    for sibling in remote_info.siblings:
        if sibling.rfilename in expected and sibling.lfs:
            h = hashlib.sha256()
            with (root / sibling.rfilename).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            assert h.hexdigest() == sibling.lfs.sha256, sibling.rfilename
            lfs_checks += 1
    for name in ["README.md", "meta/info.json", "meta/stats.json", "meta/tasks.parquet",
                 "processing/manifest.json", "verification.json"]:
        remote = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=name, revision=revision))
        assert hashlib.sha256(remote.read_bytes()).digest() == hashlib.sha256((root / name).read_bytes()).digest()
    if update_existing:
        api.delete_tag(repo_id=repo_id, repo_type="dataset", tag="v3.0")
    api.create_tag(repo_id=repo_id, repo_type="dataset", tag="v3.0", revision=revision)
    assert api.dataset_info(repo_id, revision="v3.0").sha == revision
    if not update_existing:
        api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=False)
    public = HfApi(token=False).dataset_info(repo_id, revision=revision)
    assert not public.private
    result = dict(repo_id=repo_id, url=str(url), commit=revision, public=True,
                  remote_files_verified=len(expected), remote_metadata_hashes_verified=True,
                  remote_lfs_sha256_files_verified=lfs_checks)
    (root / "publication.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--update-existing", action="store_true")
    args = p.parse_args()
    main(args.root.resolve(), args.update_existing)
