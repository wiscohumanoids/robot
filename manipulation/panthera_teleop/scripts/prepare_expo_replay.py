#!/usr/bin/env python3
"""Audit saved Panthera data or prepare faithful replay from opt-in decision ledgers.

These commands read saved files only. They never connect to hardware, inference,
or training services, and never edit original recordings or existing exports.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infra.rl.contracts import ActionNormalizer, ReplaySpec, normalization_fingerprint
from infra.rl.observations import ObservationEncoder
from infra.rl.replay import grouped_split, save_replay

TRAINING_OBJECTIVES = ('gear_insertion', 'gear_and_pin')
FULL_TASK = 'put the gear in the carrier and put a pin in it'


def session_objective(manifest):
    """Legacy whole-task fallback requires an explicit original task declaration."""
    objective = manifest.get('training_objective')
    legacy = objective is None
    if legacy:
        task = manifest.get('task', manifest.get('contract', {}).get('task'))
        if task != FULL_TASK:
            raise ValueError('Session requires an explicit training_objective')
        objective = 'gear_and_pin'
    if objective not in TRAINING_OBJECTIVES:
        raise ValueError('Unsupported training_objective')
    return objective, legacy


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def audit_existing(data_root):
    """Outcomes remain unknown for demos; kept is a selection category, not reward."""
    data_root = Path(data_root)
    raw = {category: len(list((data_root / category).glob('*/metadata.json')))
           for category in ('kept', 'review', 'discarded', 'pending')}
    processed, unique_sources = [], set()
    for path in sorted((data_root / 'processed').glob('*/processing/manifest.json')):
        manifest = json.loads(path.read_text())
        sources = {episode.get('source_id', episode['id']) for episode in manifest['episodes']}
        unique_sources.update(sources)
        processed.append({'path': str(path.parent.parent), 'episodes': len(sources),
                          'observations': manifest.get('totals', {}).get('observations')})
    hitl, expert_clips = [], []
    for path in sorted((data_root / 'hitl').glob('*/*/metadata.json')):
        metadata = json.loads(path.read_text())
        states = rows(path.parent / 'states.jsonl')
        sources = Counter(state.get('control_source') for state in states)
        outcome = metadata.get('outcome', {})
        item = {'source_episode_id': path.parent.name, 'source_category': path.parent.parent.name,
                'complete': metadata.get('complete'), 'task_success': outcome.get('success'),
                'human_interventions': metadata.get('human_interventions'),
                'state_counts': dict(sources), 'reason': outcome.get('reason'),
                'has_policy_clock': bool(states) and all('data_monotonic_ns' in s for s in states),
                'faithful_q_transitions': 0}
        hitl.append(item)
        if metadata.get('label') == 'kept' and metadata.get('complete') is True:
            clips, start = [], None
            for i, state in enumerate(states):
                if state.get('control_source') == 'human' and start is None:
                    start = i
                if start is not None and (state.get('control_source') != 'human' or i == len(states) - 1):
                    end = i if state.get('control_source') == 'human' else i - 1
                    a = states[start].get('data_monotonic_ns', states[start]['monotonic_ns'])
                    b = states[end].get('data_monotonic_ns', states[end]['monotonic_ns'])
                    if (b - a) / 1e9 >= .6:
                        clips.append({'duration_s': (b - a) / 1e9,
                                      'start_sequence': states[start]['sequence'],
                                      'end_sequence': states[end]['sequence']})
                    start = None
            expert_clips.extend(dict(clip, source_episode_id=path.parent.name,
                action_provenance='measured_next_pose_kinesthetic_proxy', task_success=None)
                for clip in clips)
    return {'schema_version': 1, 'kind': 'existing_data_audit', 'data_root': str(data_root.resolve()),
            'faithful_q_transitions': 0, 'raw_episode_counts': raw,
            'processed_exports': processed, 'unique_processed_demonstrations': len(unique_sources),
            'demonstration_task_success': None, 'hitl_rollouts': hitl,
            'approved_expert_correction_clips': expert_clips,
            'limitations': [
                'Ordinary demonstrations have no explicit task outcome; kept and complete do not imply success.',
                'Existing LeRobot actions are measured-next-pose imitation proxies, not executed policy waypoints.',
                'Existing HITL files lack policy data clock and genuine per-observation candidate sets.',
                'Assisted task success cannot label earlier policy prefixes successful.',
                'Processed releases overlap by original source ID; deduplicate before splitting.',
                'No existing export defines an evaluation holdout; October 2 camera placement changed.',
                'Review/discarded/incomplete/fault sources need explicit approval and faithful action evidence.']}


def verify_observation(observation, session):
    if not observation or not observation.get('observation_id'):
        raise ValueError('Observation identity is missing')
    for name, relative in observation.get('images', {}).items():
        path = (session / relative).resolve()
        if not path.is_relative_to(session.resolve()):
            raise ValueError('Observation image escapes session')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if observation.get('image_sha256', {}).get(name) != digest:
            raise ValueError('Observation JPEG hash mismatch')


def prepare(ledger_root, normalization, output, candidate_cache=None, gamma=.99, C=17, N=8,
            training_objective=None):
    """Require full approved, uninterrupted, unassisted episodes and genuine candidates."""
    ledger_root, output = Path(ledger_root), Path(output)
    spec = ReplaySpec(gamma=gamma, C=C, N=N, checkpoint_id=normalization['checkpoint_id'],
                      processor_id=normalization['processor_id'])
    if C > 50:
        raise ValueError('Executed prefix cannot exceed the frozen pi0.5 horizon of 50 waypoints')
    action_normalizer = ActionNormalizer.from_stats(normalization.get('action'))
    encoder = ObservationEncoder(ActionNormalizer.from_stats(normalization.get('state')))
    if training_objective is not None and training_objective not in TRAINING_OBJECTIVES:
        raise ValueError('Unsupported requested training_objective')
    sessions = [ledger_root] if (ledger_root / 'decisions.jsonl').exists() else sorted(ledger_root.iterdir())
    declared_sessions = {}
    objectives = set()
    for session in sessions:
        if not (session / 'decisions.jsonl').exists():
            continue
        manifest = json.loads((session / 'manifest.json').read_text())
        objective, legacy = session_objective(manifest)
        declared_sessions[session] = (manifest, objective, legacy)
        objectives.add(objective)
    if len(objectives) > 1:
        raise ValueError('Mixed training objectives cannot share one replay export')
    if objectives:
        declared = next(iter(objectives))
        if training_objective is not None and training_objective != declared:
            raise ValueError('Requested training objective differs from session declaration')
        training_objective = declared
    for manifest, _, _ in declared_sessions.values():
        if type(manifest.get('fixed_prefix_length')) is not int or manifest['fixed_prefix_length'] != C:
            raise ValueError('Session executed-prefix length differs from requested replay prefix; chunk lengths cannot mix')
    cache = {}
    if candidate_cache:
        candidate_cache = Path(candidate_cache)
        cache_manifest = json.loads((candidate_cache / 'cache_manifest.json').read_text())
        if (cache_manifest.get('kind') != 'genuine_pi05_candidate_cache' or
                cache_manifest.get('synthetic') is not False or cache_manifest.get('complete') is not True or
                cache_manifest.get('normalization_sha256') != normalization_fingerprint(normalization) or
                cache_manifest.get('checkpoint_id') != spec.checkpoint_id or
                cache_manifest.get('processor_id') != spec.processor_id):
            raise ValueError('Candidate cache is incomplete or differs from the pinned checkpoint/normalization')
        if cache_manifest.get('candidates_sha256') != hashlib.sha256((candidate_cache / 'candidates.jsonl').read_bytes()).hexdigest():
            raise ValueError('Candidate cache file hash mismatch')
        for item in rows(candidate_cache / 'candidates.jsonl'):
            key = (item.get('session_id'), item['observation_id'])
            if key in cache:
                raise ValueError('Duplicate candidate cache observation identity')
            cache[key] = item
    arrays = defaultdict(list)
    provenance, rejected = [], []
    for session in sessions:
        if session not in declared_sessions:
            continue
        manifest, objective, legacy = declared_sessions[session]
        endings = {}
        for ending in rows(session / 'episodes.jsonl'):
            if ending.get('event') == 'end':
                source = ending['source_episode_id']
                if source in endings:
                    raise ValueError('Duplicate episode ending')
                endings[source] = ending
        decisions_by_source = defaultdict(list)
        for decision in rows(session / 'decisions.jsonl'):
            decisions_by_source[decision['source_episode_id']].append(decision)
        for source, decisions in decisions_by_source.items():
            staged = []
            staged_source_metadata = []
            try:
                ending = endings.get(source, {})
                ending_objective = ending.get('training_objective', 'gear_and_pin' if legacy else None)
                if ending_objective != objective:
                    raise ValueError('Episode ending training objective differs from session declaration')
                if manifest.get('simulated') is not False:
                    raise ValueError('Session simulation status missing or simulated')
                if (manifest.get('fixed_prefix_length') != C or manifest.get('blend_actions') != 0 or
                        manifest.get('action_representation') != 'absolute' or manifest.get('action_units') != 'rad'):
                    raise ValueError('Session native action/executed-prefix contract differs from replay')
                fps = manifest.get('contract', {}).get('fps', 20)
                if type(fps) not in (int, float) or not np.isfinite(fps) or fps != 20:
                    raise ValueError('Session policy fps must match the pinned 20 Hz contract')
                if ending.get('complete') is not True or type(ending.get('success')) is not bool:
                    raise ValueError('Complete explicit operator task outcome is missing')
                if ending.get('intervened') is not False or ending.get('human_interventions') != 0 or ending.get('interrupted') is not False:
                    raise ValueError('Assisted/interrupted episode is ineligible for initial faithful replay')
                if ending.get('label') not in ('kept', 'review') or (ending.get('label') == 'review' and ending.get('review_approved') is not True):
                    raise ValueError('Episode requires kept or explicitly approved review provenance')
                if ending.get('reward_provenance') != 'operator_terminal_outcome':
                    raise ValueError('Operator reward provenance is missing')
                if ending.get('final_decision_id') != decisions[-1].get('decision_id'):
                    raise ValueError('Terminal decision identity mismatch')
                seen_decisions = set()
                for index, decision in enumerate(decisions):
                    decision_objective = decision.get('training_objective', 'gear_and_pin' if legacy else None)
                    if decision_objective != objective:
                        raise ValueError('Decision training objective differs from session declaration')
                    if decision.get('decision_id') in seen_decisions:
                        raise ValueError('Duplicate decision identity')
                    seen_decisions.add(decision.get('decision_id'))
                    if (decision.get('completed') is not True or decision.get('contiguous_policy') is not True or
                            decision.get('control_source') != 'policy' or decision.get('truncated') is not False or
                            decision.get('actual_prefix_length') != C):
                        raise ValueError('Decision is incomplete, crosses ownership, or has the wrong executed prefix')
                    if decision.get('checkpoint_id') != spec.checkpoint_id or decision.get('processor_id') != spec.processor_id:
                        raise ValueError('Pinned checkpoint/processor differs from decision')
                    # Prove execution against the original JSON numbers. Training
                    # float32 rounding (e.g. 0.8 or 1.1 radians) can exceed the
                    # recorder's 1e-8 proof tolerance; normalize only afterward.
                    executed = np.asarray(decision.get('executed_actions'), dtype=np.float64)
                    emitted = np.asarray(decision.get('last_emitted_target'), dtype=np.float64)
                    policy_time = decision.get('last_emitted_policy_time_s')
                    if (executed.shape != (C, 14) or emitted.shape != (14,) or
                            not np.isfinite(executed).all() or not np.isfinite(emitted).all() or
                            type(policy_time) not in (int, float) or not np.isfinite(policy_time) or
                            policy_time + 1e-8 < C / fps or
                            decision.get('emitted_command_count', 0) < C or not np.allclose(
                                emitted, executed[-1], atol=1e-8, rtol=0)):
                        raise ValueError('Actual final emitted waypoint proof is missing')
                    obs, next_obs = decision.get('observation'), decision.get('next_observation')
                    verify_observation(obs, session)
                    verify_observation(next_obs, session)
                    if next_obs.get('monotonic_ns', 0) <= obs.get('monotonic_ns', 0):
                        raise ValueError('Next observation does not follow execution')
                    candidates = []
                    candidate_evidence = {}
                    for field, observation in [('base_actions', obs), ('next_base_actions', next_obs)]:
                        entry = cache.get((manifest['session_id'], observation['observation_id']))
                        if entry is None:
                            # Direct candidates need their observation IDs explicitly tied to this decision.
                            entry = dict(decision, base_actions=decision.get(field))
                            if decision.get(field + '_observation_id') != observation['observation_id']:
                                raise ValueError(f'{field}: exact-observation genuine candidate cache missing')
                        cp = entry.get('candidates_provenance', {})
                        if (cp.get('kind') != 'pi05_inference' or cp.get('sampling') != 'independent' or
                                cp.get('observation_id') != observation['observation_id'] or
                                cp.get('checkpoint_id') != spec.checkpoint_id or
                                cp.get('processor_id') != spec.processor_id or
                                entry.get('checkpoint_id') != spec.checkpoint_id or entry.get('processor_id') != spec.processor_id):
                            raise ValueError('Genuine current pi05 candidate provenance is missing')
                        if entry.get('normalizer_hash') != normalization_fingerprint(normalization):
                            raise ValueError('Candidate normalization differs from the exact checkpoint statistics')
                        if candidate_cache:
                            from scripts.cache_expo_candidates import canonical_hash, exact_payload
                            if entry.get('observation_payload_sha256') != canonical_hash(exact_payload(observation, session)):
                                raise ValueError('Candidate cache is attached to a different state or camera observation')
                        values = np.asarray(entry.get('base_actions'), dtype=np.float32)
                        if values.ndim != 3 or values.shape[0] != N or values.shape[1] < C or values.shape[2] != 14:
                            raise ValueError('Candidate count/horizon/dimension mismatch')
                        candidates.append(action_normalizer.normalize(values[:, :C]))
                        candidate_evidence[field] = dict(
                            observation_id=observation['observation_id'],
                            candidates_provenance=cp,
                            candidate_limit_corrections=entry.get('candidate_limit_corrections'),
                            action_transform=entry.get('action_transform', cp.get('action_transform')),
                            observation_payload_sha256=entry.get('observation_payload_sha256'))
                    terminal = index == len(decisions) - 1
                    rewards = np.zeros(C, dtype=np.float32)
                    if terminal:
                        rewards[-1] = float(ending['success'])
                    staged.append(dict(obs=encoder.encode(obs, session), next_obs=encoder.encode(next_obs, session),
                        actions=action_normalizer.normalize(executed), base_actions=candidates[0], next_base_actions=candidates[1],
                        rewards=rewards, reward_return=float(np.dot(rewards, np.power(gamma, np.arange(C)))),
                        bootstrap_discount=0.0 if terminal else gamma ** C, terminated=terminal,
                        source_episode_ids=source, editor_train_valid=True, transition_valid=True))
                    response = decision.get('response', {})
                    staged_source_metadata.append(dict(candidate_evidence=candidate_evidence,
                        executed_limit_corrections=decision.get('limit_corrections', response.get('limit_corrections')),
                        selected_candidate=response.get('selected_candidate'),
                        selected_is_edited=response.get('selected_is_edited'),
                        policy_bundle_id=decision.get('policy_bundle_id', response.get('policy_bundle_id'))))
                for index, item in enumerate(staged):
                    for key, value in item.items():
                        arrays[key].append(value)
                    provenance.append(dict(source_episode_id=source, complete=True, simulation=False,
                        control_source='policy', contiguous_policy=True, task_success=ending['success'],
                        source_category=ending['label'], review_approved=ending.get('review_approved', False),
                        intervened=False, human_interventions=0, action_provenance='executed_policy_waypoints',
                        training_objective=objective,
                        candidate_provenance='pi05_inference', checkpoint_id=spec.checkpoint_id,
                        processor_id=spec.processor_id, session_id=manifest['session_id'],
                        decision_id=decisions[index]['decision_id'], **staged_source_metadata[index]))
            except (ValueError, TypeError, KeyError, OSError) as exc:
                rejected.append({'source_episode_id': source, 'session_id': session.name, 'reason': str(exc)})
    report = {'eligible_transitions': len(provenance), 'rejected_sources': rejected}
    if not provenance:
        raise ValueError('Zero faithful transitions. ' + json.dumps(report))
    result = {}
    for key, value in arrays.items():
        dtype = str if key == 'source_episode_ids' else bool if key in ('terminated', 'editor_train_valid', 'transition_valid') else np.float32
        result[key] = np.asarray(value, dtype=dtype)
    train, evaluation = grouped_split(result['source_episode_ids'])
    metadata = dict(schema_version=1, kind='faithful_policy_replay', training_eligible=True, synthetic=False,
        training_objective=training_objective,
        spec=spec.as_dict(), normalization=normalization, normalization_sha256=normalization_fingerprint(normalization),
        transitions=provenance, sources=sorted(set(result['source_episode_ids'].tolist())),
        splits={'train_indices': train.tolist(), 'eval_indices': evaluation.tolist()}, audit=report,
        observation_encoding='Native state checkpoint QUANTILES + RGB block means 16x16 in camera/CHW order',
        reward_semantics=('Sparse operator terminal outcome: gear fully seated gives 1, failure gives 0; '
                          'no pin requirement; zero before final executed terminal waypoint'
                          if training_objective == 'gear_insertion' else
                          'Sparse operator terminal full gear-and-pin task outcome; '
                          'zero before final executed terminal waypoint'))
    save_replay(output, result, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--data-root', type=Path, default=ROOT / 'data')
    audit.add_argument('--output', type=Path, required=True)
    export = sub.add_parser('prepare')
    export.add_argument('--ledger-root', type=Path, required=True)
    export.add_argument('--normalization', type=Path, required=True)
    export.add_argument('--candidate-cache', type=Path)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--gamma', type=float, default=.99)
    export.add_argument('--prefix-length', type=int, default=17,
                        help='Executed waypoints per decision (default 17); must match every session')
    export.add_argument('--candidates', type=int, default=8)
    export.add_argument('--training-objective', choices=TRAINING_OBJECTIVES, default='gear_insertion')
    args = parser.parse_args()
    if args.command == 'audit':
        value = audit_existing(args.data_root)
        write_new_json(args.output, value)
        print(json.dumps({'unique_processed_demonstrations': value['unique_processed_demonstrations'],
                          'hitl_rollouts': len(value['hitl_rollouts']), 'faithful_q_transitions': 0}, indent=2))
    else:
        value = prepare(args.ledger_root, json.loads(args.normalization.read_text()), args.output,
                        args.candidate_cache, args.gamma, args.prefix_length, args.candidates,
                        args.training_objective)
        print(json.dumps(value['audit'], indent=2))


if __name__ == '__main__':
    main()
