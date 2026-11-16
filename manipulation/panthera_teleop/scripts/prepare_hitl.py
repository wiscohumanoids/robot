#!/usr/bin/env python3
"""Select human correction intervals and successful unassisted rollouts for export.

Reads saved files only. Each interval becomes a separate LeRobot episode, so
next-state actions and action-chunk padding never cross a control handoff.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from recording import atomic_json
from hitl_recording import data_timestamp, data_origin


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_intervals(metadata, states):
    if (metadata.get('label') != 'kept' or metadata.get('complete') is not True or
            metadata.get('context', {}).get('mode') != 'human_in_the_loop'):
        return []
    if not states:
        return []
    times = [data_timestamp(s) for s in states]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('State timestamps must increase')
    sources = [s.get('control_source') for s in states]
    if any(s not in ('human', 'policy', 'hold') for s in sources):
        raise ValueError('Missing or invalid control ownership')
    if 'human' not in sources:
        if metadata.get('outcome', {}).get('success') is True and 'policy' in sources:
            # Trim waiting before/after policy execution; retain inference holds in between.
            first = sources.index('policy')
            last = len(sources) - 1 - sources[::-1].index('policy')
            return [('successful_unassisted', times[first], times[last])]
        return []
    intervals = []
    start = None
    for i, source in enumerate(sources):
        if start is not None and states[i].get('recording_segment') != states[i-1].get('recording_segment'):
            intervals.append(('human_correction', times[start], times[i-1]))
            start = None
        if source == 'human' and start is None:
            start = i
        if start is not None and (source != 'human' or i == len(sources)-1):
            end = i if source == 'human' else i-1
            intervals.append(('human_correction', times[start], times[end]))
            start = None
    return intervals


def prepare(source, output, template, minimum_seconds=.6):
    if output.exists():
        raise FileExistsError('Choose a new annotations file')
    spec = json.loads(template.read_text())
    spec['episodes'] = []
    spec['source_kind'] = 'human_gated_interventions_and_successful_unassisted_rollouts'
    spec['trim_method'] = 'Explicit control ownership; one contiguous correction per exported episode'
    spec['trim_uncertainty_s'] = 0
    spec['annotation_time_basis'] = 'Active recording clock when present, otherwise host receive clock'
    spec['sampling'] = 'Regular policy-rate grid on data_monotonic_ns when present, otherwise host monotonic time; nearest measured states and timestamp-aligned camera frames'
    spec['timing'] = 'Host receive times remain in monotonic_ns; active recording times use data_monotonic_ns, excluding holds and undoing policy safety slowdown'
    spec.pop('recording_session_local_date', None)
    skipped = []
    for directory in sorted(source.iterdir()):
        if not directory.is_dir() or not (directory / 'metadata.json').exists():
            continue
        metadata = json.loads((directory / 'metadata.json').read_text())
        states = rows(directory / 'states.jsonl')
        intervals = select_intervals(metadata, states)
        for index, (kind, start, end) in enumerate(intervals):
            if (end-start)/1e9 < minimum_seconds:
                skipped.append(dict(id=directory.name, segment=index, reason='segment_too_short'))
                continue
            origin = data_origin(metadata)
            # Do not let a mismatched image/action contract silently into the dataset.
            contract = metadata['context']['contract']
            for key in ('fps', 'image_size', 'image_crops_xyxy', 'task'):
                if spec[key] != contract[key]:
                    raise ValueError(f'{directory.name}: template differs from inference contract: {key}')
            spec['episodes'].append(dict(id=f'{directory.name}-hitl-{index:03d}',
                source_id=directory.name, source_category='kept', selection=kind,
                start_s=(start-origin)/1e9, end_s=(end-origin)/1e9,
                checkpoint=metadata['context']['checkpoint'],
                operator=metadata['context']['rollout']['operator'],
                scene=metadata['context']['rollout']['scene'], outcome=metadata.get('outcome')))
    if not spec['episodes']:
        raise ValueError('No eligible segments: keep corrections or successful unassisted rollouts first')
    spec['selection_skipped'] = skipped
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, spec)
    return spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/hitl/kept')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--template', type=Path, default=ROOT / 'config/postprocessing-gear-pin-sept30.json')
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.template)
    print(f"Selected {len(result['episodes'])} training episodes; skipped {len(result['selection_skipped'])} short segments")
