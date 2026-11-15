#!/usr/bin/env python3
"""Combine compatible processed batches without transforming source images again."""
import argparse
import json
from pathlib import Path
import shutil

import pyarrow.compute as pc
import pyarrow.parquet as pq
from lerobot.datasets.aggregate import aggregate_datasets
from postprocess_kept import sha256, write_json


def iter_rows(root):
    for path in sorted((root / 'data').rglob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            yield from batch.to_pylist()


def merge(roots, output, repo_id, source):
    if output.exists():
        raise FileExistsError(output)
    manifests = [json.loads((r / 'processing/manifest.json').read_text()) for r in roots]
    contract = ('fps', 'image_size', 'image_crops_xyxy', 'image_encoding', 'task',
                'action_semantics', 'state_semantics', 'max_state_alignment_s',
                'max_camera_alignment_s')
    identifiers = set()
    for root, manifest in zip(roots, manifests):
        for key in contract:
            if manifest[key] != manifests[0][key]:
                raise ValueError(f'Incompatible processing contract: {key}')
        for e in manifest['episodes']:
            if e['id'] in identifiers:
                raise ValueError(f'Duplicate episode: {e["id"]}')
            identifiers.add(e['id'])
            for name in ('metadata.json', 'states.jsonl', 'frames.jsonl'):
                if sha256(source / e['id'] / name) != e['source_hashes'][name]:
                    raise ValueError(f'Changed source: {e["id"]}/{name}')
            archive = root / 'trimmed_full_rate' / f'{e["id"]}.tar.gz'
            if sha256(archive) != e['trimmed_archive_sha256']:
                raise ValueError(f'Changed archive: {archive}')
    aggregate_datasets([m['repo_id'] for m in manifests], repo_id,
                       roots=roots, aggr_root=output)
    combined = dict(manifests[0], repo_id=repo_id, episodes=[])
    combined.pop('recording_session_local_date', None)
    combined['recording_session_local_dates'] = sorted({date for m in manifests for date in
        m.get('recording_session_local_dates', [m.get('recording_session_local_date')]) if date})
    combined['reused_processed_batches'] = [str(r.resolve()) for r in roots]
    combined['totals'] = {key: sum(m['totals'][key] for m in manifests)
                          for key in manifests[0]['totals']}
    episode_offset = frame_offset = 0
    for root, manifest in zip(roots, manifests):
        for e in manifest['episodes']:
            index = e['episode_index'] + episode_offset
            combined['episodes'].append(dict(e, episode_index=index))
            alignment = pq.read_table(root / f'processing/alignment/episode_{e["episode_index"]:06d}.parquet')
            for key, offset in [('index', frame_offset), ('episode_index', episode_offset)]:
                column = alignment.schema.get_field_index(key)
                alignment = alignment.set_column(column, key, pc.add(alignment[key], offset))
            path = output / f'processing/alignment/episode_{index:06d}.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(alignment, path, compression='zstd')
        for directory in ['trimmed_full_rate', 'processing/source_metadata']:
            shutil.copytree(root / directory, output / directory, dirs_exist_ok=True)
        episode_offset += len(manifest['episodes'])
        frame_offset += manifest['totals']['observations']
    write_json(output / 'processing/manifest.json', combined)
    annotations = {k: v for k, v in combined.items()
                   if k not in ('episodes', 'totals', 'repo_id', 'reused_processed_batches')}
    annotations['episodes'] = [e for root in roots for e in
        json.loads((root / 'processing/annotations.json').read_text())['episodes']]
    write_json(output / 'processing/annotations.json', annotations)
    # Compare the actual encoded bytes and numeric values, not just counts.
    # The merge may repack Parquet files but must never re-encode an image.
    merged_rows = iter_rows(output)
    episode_offset = frame_offset = checked = 0
    for root, manifest in zip(roots, manifests):
        for original in iter_rows(root):
            original['index'] += frame_offset
            original['episode_index'] += episode_offset
            assert original == next(merged_rows), f'Merge changed observation {checked}'
            checked += 1
        episode_offset += len(manifest['episodes'])
        frame_offset += manifest['totals']['observations']
    assert next(merged_rows, None) is None
    write_json(output / 'processing/reuse_verification.json', dict(
        passed=True, checked_observations=checked, embedded_jpeg_bytes_unchanged=True,
        numeric_rows_unchanged_except_global_indices=True,
        reused_existing_episodes=len(manifests[0]['episodes'])))
    print(json.dumps(combined['totals'], indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--repo-id', required=True)
    p.add_argument('--source', type=Path, default=Path('data/kept'))
    a = p.parse_args()
    merge(a.roots, a.output, a.repo_id, a.source)
