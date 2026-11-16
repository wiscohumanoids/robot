#!/usr/bin/env python3
"""Cache genuine frozen-pi0.5 proposals at exact recorded EXPO observations.

Reads local session ledgers and samples a local policy or an already loaded
frozen backend through a loopback tunnel. It never reconstructs observations
from replay actions, resizes images, or modifies original ledgers. Both decision
and next-tick observations are sampled using the same frozen base.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def exact_payload(observation, session):
    """Resolve only the exact JPEG bytes named and hashed by the recorder."""
    if not isinstance(observation, dict) or not observation.get('observation_id'):
        raise ValueError('A recorded exact observation_id is required')
    from infra.rl.contracts import CAMERAS
    images = observation.get('images', {})
    hashes = observation.get('image_sha256', {})
    if set(images) != set(CAMERAS) or set(hashes) != set(CAMERAS):
        raise ValueError('Recorded observation must identify and hash all three cropped JPEGs')
    root = Path(session).resolve(strict=True)
    encoded = {}
    for camera in CAMERAS:
        path = (root / images[camera]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('Observation image path escapes its session')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != hashes[camera]:
            raise ValueError(f'Recorded observation JPEG hash mismatch: {camera}')
        encoded[camera] = base64.b64encode(data).decode('ascii')
    return dict(observation_id=observation['observation_id'], state=observation['state'],
                images=encoded, sequence=observation['observation_id'])


def read_observations(sessions, report=None):
    """Join by explicit IDs, rejecting any changed observation under an existing ID."""
    observations = {}
    report = report if report is not None else {}
    report.update(decisions_seen=0, decisions_selected=0, skipped={})
    for directory in sessions:
        session = Path(directory).resolve(strict=True)
        ledger = session / 'decisions.jsonl'
        for number, line in enumerate(ledger.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            report['decisions_seen'] += 1
            reason = None
            if row.get('control_source', 'policy') != 'policy':
                reason = 'nonpolicy'
            elif row.get('truncated', False):
                reason = 'truncated'
            elif row.get('completed', True) is not True or row.get('next_observation') is None:
                reason = 'incomplete'
            elif row.get('contiguous_policy', True) is not True:
                reason = 'noncontiguous'
            if reason:
                report['skipped'][reason] = report['skipped'].get(reason, 0) + 1
                continue
            report['decisions_selected'] += 1
            for key in ('observation', 'next_observation'):
                observation = row.get(key)
                if observation is None:
                    raise ValueError(f'{ledger}:{number}: missing exact {key}; no candidate can be fabricated')
                payload = exact_payload(observation, session)
                identity = observation['observation_id']
                fingerprint = canonical_hash(payload)
                if identity in observations and observations[identity]['fingerprint'] != fingerprint:
                    raise ValueError(f'Observation ID reused for different input: {identity}')
                observations.setdefault(identity, dict(payload=payload, session_id=session.name,
                    source_session=str(session), fingerprint=fingerprint))
    if not observations:
        raise ValueError('No exact recorded observations found')
    return observations


def cache_candidates(sessions, checkpoint, output, seed=0, *, runtime_factory=None,
                     backend=None, manifest=None, remote_runtime_factory=None):
    """Cache eight genuine proposals per saved observation, without new motion.

    Remote draws use the backend's independent noise; no reproducible seed is
    claimed. Proposals remain native physical radians in the cache, with saved
    normalization alongside, for prepare_expo_replay to normalize exactly once.
    """
    remote = backend is not None or manifest is not None
    if remote and (not backend or not manifest or checkpoint is not None):
        raise ValueError('Remote caching requires backend and manifest, with no local checkpoint')
    if not remote and checkpoint is None:
        raise ValueError('Local caching requires an immutable checkpoint')
    if remote and runtime_factory is not None:
        raise ValueError('Use remote_runtime_factory for remote backend fixtures')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Candidate cache output must be a new directory')
    for session in sessions:
        root = Path(session).resolve(strict=True)
        if output.is_relative_to(root):
            raise ValueError('Candidate cache must be outside the original session ledger')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Seed must be an integer in [0,2**63)')
    source_report = {}
    observations = read_observations(sessions, source_report)
    if remote:
        if remote_runtime_factory is None:
            from infra.rl.remote_server import Runtime
            remote_runtime_factory = Runtime
        runtime = remote_runtime_factory(manifest, backend=backend)
    else:
        if runtime_factory is None:
            from infra.rl.server import Runtime
            from infra.rl.resources import require_idle_gpu
            runtime_factory = lambda path: Runtime(path, guard=lambda **kwargs:
                require_idle_gpu(**kwargs, disk_path=output))
        runtime = runtime_factory(checkpoint)
    # Local load() retains the idle-GPU lease until completion. The remote
    # runtime uses the existing backend and constructs no additional model.
    temporary = output.parent / f'.{output.name}.partial-{uuid.uuid4().hex}'
    try:
        if remote:
            runtime.health()
        else:
            runtime.load()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.mkdir()
        normalization = runtime.normalization
        if not normalization or runtime.normalizer_hash != canonical_hash(normalization):
            raise ValueError('Genuine saved checkpoint quantile normalization is mandatory')
        checkpoint_id = runtime.checkpoint_id if remote else runtime.adapter.checkpoint_id
        processor_id = runtime.processor_id if remote else runtime.adapter.processor_id
        if normalization.get('checkpoint_id') != checkpoint_id or normalization.get('processor_id') != processor_id:
            raise ValueError('Normalization differs from the frozen checkpoint/processor identity')
        (temporary / 'normalization.json').write_text(json.dumps(normalization, indent=2, allow_nan=False) + '\n')
        with (temporary / 'candidates.jsonl').open('x') as stream:
            for identity, record in observations.items():
                payload = dict(record['payload'])
                if remote:
                    response = runtime.sample_candidates(payload, num_candidates=8)
                else:
                    payload['seed'] = int(hashlib.sha256(f'{seed}:{identity}'.encode()).hexdigest()[:16], 16) % 2**63
                    response = runtime.infer(payload)
                provenance = response['candidates_provenance']
                if (provenance.get('kind') != 'pi05_inference' or
                        provenance.get('sampling') != 'independent' or
                        response['observation_id'] != identity or
                        provenance.get('observation_id') != identity or
                        response.get('checkpoint_id') != checkpoint_id or
                        response.get('processor_id') != processor_id or
                        response.get('normalizer_hash') != runtime.normalizer_hash):
                    raise ValueError('Inference did not preserve genuine exact-observation provenance')
                if remote and (provenance.get('seed_controlled') is not False or 'seed' in provenance):
                    raise ValueError('Remote backend cannot claim controlled/reproducible sampling seeds')
                if remote and len(response.get('candidate_limit_corrections', [])) != 8:
                    raise ValueError('Remote cache requires each draw\'s source limit-correction record')
                import numpy as np
                physical = np.asarray(response['base_actions'], dtype=np.float32)
                if physical.shape != (8, 50, 14) or not np.isfinite(physical).all() or response.get('mock', False):
                    raise ValueError('Inference must return eight genuine finite 50-step native action chunks')
                row = dict(observation_id=identity, session_id=record['session_id'],
                    source_session=record['source_session'],
                    observation_payload_sha256=record['fingerprint'],
                    base_actions=response['base_actions'], candidates_provenance=provenance,
                    checkpoint_id=response['checkpoint_id'], processor_id=response['processor_id'],
                    normalizer_hash=response['normalizer_hash'])
                if remote:
                    row.update(candidate_limit_corrections=response['candidate_limit_corrections'],
                               action_transform=provenance.get('action_transform'))
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        cache_manifest = dict(schema_version=1, kind='genuine_pi05_candidate_cache', synthetic=False,
            complete=True, observations=len(observations), num_candidates=8,
            checkpoint_id=checkpoint_id, processor_id=processor_id,
            normalization_sha256=runtime.normalizer_hash, seed=None if remote else seed,
            seed_controlled=not remote,
            sampling_backend='independent_runtime_noise' if remote else 'local_seeded_pi05',
            action_units='rad', action_representation='absolute', base_actions_normalized=False,
            source_report=source_report,
            source_sessions=[str(Path(path).resolve()) for path in sessions],
            candidates_sha256=hashlib.sha256((temporary / 'candidates.jsonl').read_bytes()).hexdigest())
        if remote:
            cache_manifest.update(backend=backend, remote_manifest_sha256=runtime.manifest_sha256,
                                  remote_manifest=str(Path(manifest).resolve()),
                                  action_transform='checkpoint_postprocess_and_source_gripper_clamp')
        (temporary / 'cache_manifest.json').write_text(json.dumps(cache_manifest, indent=2) + '\n')
        temporary.rename(output)
        return cache_manifest
    finally:
        if not remote:
            runtime.close()
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', action='append', required=True, help='Local EXPO session directory; repeat for more')
    parser.add_argument('--checkpoint', help='Existing local immutable pi0.5 checkpoint')
    parser.add_argument('--backend', help='Existing frozen HTTP loopback backend through an SSH tunnel')
    parser.add_argument('--manifest', help='Verified frozen remote-base manifest with checkpoint quantiles')
    parser.add_argument('--output', required=True, help='New candidate cache directory outside source ledgers')
    parser.add_argument('--seed', type=int, help='Local model sampling seed; remote backend cannot control seeds')
    args = parser.parse_args()
    if args.backend or args.manifest:
        if not args.backend or not args.manifest or args.checkpoint or args.seed is not None:
            parser.error('Remote mode requires --backend and --manifest; omit --checkpoint and --seed')
    elif not args.checkpoint:
        parser.error('Specify --checkpoint for local mode, or --backend and --manifest for remote mode')
    print(json.dumps(cache_candidates(args.session, args.checkpoint, args.output,
        args.seed if args.seed is not None else 0, backend=args.backend, manifest=args.manifest), indent=2))


if __name__ == '__main__':
    main()
