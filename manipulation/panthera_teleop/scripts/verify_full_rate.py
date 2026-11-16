#!/usr/bin/env python3
"""Verify every full-rate archive against its unchanged original recording."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from postprocess_kept import source_path, read_rows, sha256, write_json, data_timestamp, data_origin


def verify(root, source, previous_root=None):
    manifest = json.loads((root / 'processing/manifest.json').read_text())
    checked_images = checked_states = original_images = 0
    reused = 0
    previous = {}
    if previous_root is not None:
        previous_manifest = json.loads((previous_root / 'processing/manifest.json').read_text())
        previous_report = json.loads((previous_root / 'processing/full_rate_verification.json').read_text())
        assert previous_report['passed'] and previous_report['episodes'] == len(previous_manifest['episodes'])
        previous = {e['id']: e for e in previous_manifest['episodes']}
        for e in manifest['episodes']:
            if e['id'] in previous:
                assert e == previous[e['id']], f"Previous annotation or source hashes changed: {e['id']}"
                archive = root / 'trimmed_full_rate' / f'{e["id"]}.tar.gz'
                assert sha256(archive) == e['trimmed_archive_sha256']
                reused += 1
        assert reused == len(previous)
        checked_images = previous_report['archived_images_exact_source_match']
        checked_states = previous_report['archived_states_exact_source_match']
        original_images = previous_report['original_image_hashes_verified']
    for e in manifest['episodes']:
        if e['id'] in previous:
            continue
        src = source_path(source, e)
        metadata = json.loads((src / 'metadata.json').read_text())
        origin = data_origin(metadata)
        start, end = [origin + round(e[k] * 1e9) for k in ['start_s', 'end_s']]
        states, frames = [read_rows(src / name) for name in ['states.jsonl', 'frames.jsonl']]
        selected_states = [s for s in states if start <= data_timestamp(s) <= end]
        selected_frames = [f for f in frames if start <= data_timestamp(f) <= end]
        digest = hashlib.sha256()
        hashes = {}
        for f in frames:
            h = sha256(src / f['path'])
            hashes[f['path']] = h
            digest.update(f['path'].encode())
            digest.update(bytes.fromhex(h))
        assert digest.hexdigest() == e['source_hashes']['ordered_original_images_sha256']
        archive = root / 'trimmed_full_rate' / f'{e["id"]}.tar.gz'
        assert sha256(archive) == e['trimmed_archive_sha256']
        prefix = e['id'] + '/'
        expected = {'metadata.json', 'states.jsonl', 'frames.jsonl'} | {f['path'] for f in selected_frames}
        seen = set()
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                if not member.isfile():
                    continue
                assert member.name.startswith(prefix)
                name = member.name[len(prefix):]
                assert name in expected and name not in seen
                seen.add(name)
                data = tar.extractfile(member).read()
                if name == 'states.jsonl':
                    assert [json.loads(l) for l in data.splitlines() if l.strip()] == selected_states
                elif name == 'frames.jsonl':
                    assert [json.loads(l) for l in data.splitlines() if l.strip()] == selected_frames
                elif name == 'metadata.json':
                    derived = json.loads(data)
                    assert derived['samples'] == len(selected_states) == e['trimmed_states']
                    assert derived['camera_frames'] == e['trimmed_camera_frames']
                    assert derived['processing']['source_hashes'] == e['source_hashes']
                else:
                    assert hashlib.sha256(data).hexdigest() == hashes[name]
        assert seen == expected
        checked_images += len(selected_frames)
        checked_states += len(selected_states)
        original_images += len(frames)
        print(f'{e["episode_index"] + 1}/{len(manifest["episodes"])} archive verified', flush=True)
    result = dict(passed=True, episodes=len(manifest['episodes']),
                  original_image_hashes_verified=original_images,
                  archived_images_exact_source_match=checked_images,
                  archived_states_exact_source_match=checked_states,
                  archive_contents_and_boundaries_verified=True,
                  newly_verified_episodes=len(manifest["episodes"]) - reused,
                  reused_verified_episodes=reused,
                  previous_verification_root=str(previous_root) if previous_root else None)
    write_json(root / 'processing/full_rate_verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--source', type=Path, default=Path('data/kept'))
    p.add_argument("--previous-root", type=Path, help="Reuse a passed prior audit when episode metadata and archive hashes are unchanged")
    a = p.parse_args()
    verify(a.root, a.source, a.previous_root)
