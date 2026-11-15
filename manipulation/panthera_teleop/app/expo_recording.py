"""Opt-in EXPO decision ledger; all filesystem work belongs to its writer.

This is deliberately separate from the measured-next-pose HITL/SFT store.
Missing candidate sets or rewards remain missing, never synthesized here.
"""
import base64
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import uuid


class ExplorationRolloutStore:
    def __init__(self, root, manifest, queue_size=1024):
        self.root = Path(root)
        self.session_id = uuid.uuid4().hex
        self.path = self.root / self.session_id
        self.events = queue.Queue(maxsize=queue_size)
        self.failure = None
        self.active = False
        self.closed = False
        self.status = dict(completed_decisions=0, source_episode_id=None,
                           last_outcome=None, interrupted=False, intervened=False)
        self.manifest = dict(manifest, schema_version=1, session_id=self.session_id,
                             training_objective=manifest.get('training_objective', 'gear_and_pin'),
                             action_representation='absolute', action_units='rad',
                             fixed_prefix_length=manifest.get('fixed_prefix_length', 1), blend_actions=0,
                             reward_policy='Explicit operator episode outcome; rewards absent until validated export',
                             timing='Original host time; safety retiming is preserved')
        if self.manifest['training_objective'] not in ('gear_insertion', 'gear_and_pin'):
            raise ValueError('Unknown EXPO training objective')
        if type(self.manifest['fixed_prefix_length']) is not int or not 1 <= self.manifest['fixed_prefix_length'] <= 50:
            raise ValueError('EXPO prefix length must be an integer from 1 to 50')
        self.thread = threading.Thread(target=self._worker, name='expo-writer', daemon=True)
        self.thread.start()

    def snapshot(self):
        return dict(self.status, enabled=True, active=self.active, error=self.failure,
                    session_id=self.session_id, path=str(self.path), closed=self.closed)

    def submit(self, kind, payload):
        if self.failure:
            raise RuntimeError(f'EXPO recording failed: {self.failure}')
        if self.closed:
            raise RuntimeError('EXPO recorder is closed')
        try:
            self.events.put_nowait((kind, payload))
        except queue.Full:
            self.failure = 'Recorder queue full; collection stopped to avoid silent data loss'
            raise RuntimeError(self.failure)

    def begin(self, details, monotonic_ns):
        if details.get('training_objective', self.manifest['training_objective']) != self.manifest['training_objective']:
            raise ValueError('Trial objective differs from the immutable session objective')
        self.submit('begin', dict(details=details, monotonic_ns=monotonic_ns))
        self.active = True

    def requested(self, decision_index, state, images, monotonic_ns, camera_metadata=None, observation_id=None):
        if self.active:
            observation_id = observation_id or uuid.uuid4().hex
            self.submit('request', dict(decision_index=decision_index,
                observation=dict(state=list(state), images=dict(images), monotonic_ns=monotonic_ns,
                                 cameras=camera_metadata or {}, observation_id=observation_id)))
            return observation_id

    def accepted(self, response, executed_actions):
        if self.active:
            self.submit('accept', dict(response=response, executed_actions=executed_actions))

    def responded(self, response):
        if self.active:
            self.submit('response', dict(response=response))

    def command(self, target, emitted_ns, policy_time_s):
        if self.active:
            self.submit('command', dict(target=list(target), emitted_monotonic_ns=emitted_ns,
                                        policy_time_s=policy_time_s))

    def completed(self, state, images, monotonic_ns, camera_metadata=None):
        if self.active:
            observation_id = uuid.uuid4().hex
            self.submit('complete', dict(state=list(state), images=dict(images),
                monotonic_ns=monotonic_ns, cameras=camera_metadata or {}, observation_id=observation_id))
            return observation_id

    def boundary(self, reason):
        if self.active:
            self.submit('boundary', dict(reason=reason, monotonic_ns=time.monotonic_ns()))

    def end(self, outcome):
        if outcome.get('training_objective', self.manifest['training_objective']) != self.manifest['training_objective']:
            raise ValueError('Trial objective differs from the immutable session objective')
        if self.active:
            self.submit('end', dict(outcome, monotonic_ns=time.monotonic_ns()))
            self.active = False

    def close(self, reason='session_stopped'):
        if self.closed:
            return
        try:
            if self.active and not self.failure:
                self.end(dict(success=None, complete=False, label='review', reason=reason))
            if not self.failure:
                self.submit('close', {})
        except RuntimeError as exc:
            # A saturated logging queue must not prevent deliberate SDK cleanup.
            self.failure = self.failure or str(exc)
        self.closed = True
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            self.failure = self.failure or 'EXPO writer did not finish'

    def _observation(self, value):
        observation = dict(value)
        identifier = value.get('observation_id') or uuid.uuid4().hex
        if not isinstance(identifier, str) or len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('Expected generated observation UUID')
        observation['observation_id'] = identifier
        images, hashes = {}, {}
        for name, encoded in value['images'].items():
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError('Invalid camera name')
            jpeg = base64.b64decode(encoded, validate=True)
            relative = Path('images') / identifier / f'{name}.jpg'
            destination = self.path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(jpeg)
            images[name] = str(relative)
            hashes[name] = hashlib.sha256(jpeg).hexdigest()
        observation.update(images=images, image_sha256=hashes)
        return observation

    @staticmethod
    def _write(stream, payload):
        stream.write(json.dumps(payload, allow_nan=False) + '\n')
        stream.flush()

    def _worker(self):
        pending = episode = last_decision = None
        segment = 0
        try:
            self.path.mkdir(parents=True, exist_ok=False)
            (self.path / 'manifest.json').write_text(json.dumps(self.manifest, indent=2, allow_nan=False) + '\n')
            with (self.path / 'decisions.jsonl').open('w') as decisions, \
                 (self.path / 'episodes.jsonl').open('w') as episodes, \
                 (self.path / 'commands.jsonl').open('w') as commands:
                def abort(reason):
                    nonlocal pending, segment
                    if pending is not None:
                        pending.update(completed=False, contiguous_policy=False, truncated=True,
                                       rejection_reason=reason)
                        self._write(decisions, pending)
                        pending = None
                    segment += 1
                while True:
                    try:
                        kind, payload = self.events.get(timeout=.2)
                    except queue.Empty:
                        if self.failure:
                            break
                        continue
                    if kind == 'close':
                        break
                    if kind == 'begin':
                        if episode is not None:
                            raise RuntimeError('An EXPO episode is already active')
                        episode = dict(source_episode_id=uuid.uuid4().hex, event='begin',
                                       training_objective=self.manifest['training_objective'],
                                       **payload, intervened=False, interrupted=False, human_interventions=0)
                        last_decision = None
                        segment = 0
                        self.status.update(completed_decisions=0, source_episode_id=episode['source_episode_id'],
                                           interrupted=False, intervened=False, last_outcome=None)
                        self._write(episodes, episode)
                    elif kind == 'request' and episode is not None:
                        if pending is not None:
                            raise RuntimeError('Previous EXPO decision lacks completion feedback')
                        pending = dict(schema_version=1, source_episode_id=episode['source_episode_id'],
                            session_id=self.session_id, decision_id=uuid.uuid4().hex,
                            training_objective=self.manifest['training_objective'],
                            simulated=self.manifest.get('simulated', False),
                            decision_index=payload['decision_index'], recording_segment=segment,
                            observation=self._observation(payload['observation']),
                            next_observation=None, executed_actions=None, rewards=None,
                            terminated=False, truncated=False, completed=False,
                            contiguous_policy=True, control_source='policy', actual_prefix_length=0,
                            emitted_command_count=0, last_emitted_target=None, last_emitted_policy_time_s=None,
                            checkpoint_id=self.manifest.get('checkpoint_id'),
                            processor_id=self.manifest.get('processor_id'))
                    elif kind in ('response', 'accept') and pending is not None:
                        response = payload['response']
                        pending['response'] = response
                        if kind == 'accept':
                            if len(payload['executed_actions']) != self.manifest['fixed_prefix_length']:
                                raise RuntimeError('EXPO accepted prefix differs from the fixed session prefix length')
                            pending.update(executed_actions=payload['executed_actions'],
                                           actual_prefix_length=len(payload['executed_actions']))
                        for field in ('base_actions', 'next_base_actions', 'candidates_provenance',
                                      'checkpoint_id', 'processor_id', 'policy_bundle_id',
                                      'normalizer_hash', 'rl_model_id', 'seed', 'explore'):
                            if field in response:
                                pending[field] = response[field]
                        if 'base_actions' in response:
                            pending['base_actions_observation_id'] = pending['observation']['observation_id']
                    elif kind == 'command' and pending is not None:
                        pending['emitted_command_count'] += 1
                        pending['last_emitted_target'] = payload['target']
                        pending['last_emitted_policy_time_s'] = payload['policy_time_s']
                        self._write(commands, dict(payload, decision_id=pending['decision_id'],
                                                   source_episode_id=episode['source_episode_id']))
                    elif kind == 'complete' and pending is not None:
                        prefix_length = self.manifest['fixed_prefix_length']
                        if not pending['executed_actions'] or pending['actual_prefix_length'] != prefix_length:
                            raise RuntimeError('EXPO completion lacks the full fixed accepted prefix')
                        if (pending['last_emitted_target'] is None or any(abs(a-b) > 1e-8 for a, b in
                                zip(pending['last_emitted_target'], pending['executed_actions'][-1]))):
                            raise RuntimeError('EXPO completion lacks the final emitted waypoint')
                        fps = self.manifest.get('contract', {}).get('fps', 20)
                        if (pending['last_emitted_policy_time_s'] is None or
                                pending['last_emitted_policy_time_s'] + 1e-8 < prefix_length / fps):
                            raise RuntimeError('EXPO completion lacks the full emitted policy duration')
                        pending.update(completed=True, next_observation=self._observation(payload),
                            safety_execution='Bounded smoothed waypoint prefix; emitted commands in commands.jsonl')
                        last_decision = pending['decision_id']
                        self._write(decisions, pending)
                        self.status['completed_decisions'] += 1
                        pending = None
                    elif kind == 'boundary' and episode is not None:
                        episode['interrupted'] = True
                        if payload['reason'] in ('intervene', 'resume'):
                            episode['intervened'] = True
                            if payload['reason'] == 'intervene':
                                episode['human_interventions'] += 1
                        abort(payload['reason'])
                        self.status.update(interrupted=episode['interrupted'], intervened=episode['intervened'])
                        self._write(episodes, dict(payload, event='boundary',
                                                   source_episode_id=episode['source_episode_id']))
                    elif kind == 'end' and episode is not None:
                        if pending is not None:
                            if (pending['emitted_command_count'] == 0 and
                                    payload.get('complete', True) is True and
                                    type(payload.get('success')) is bool):
                                # Finish may cancel the next prediction before it ever
                                # commands an arm. Preserve that request as provenance,
                                # while the preceding completed decision remains terminal.
                                self._write(episodes, dict(event='unexecuted_request',
                                    source_episode_id=episode['source_episode_id'],
                                    cancelled_before_execution=True, decision=pending))
                                pending = None
                            else:
                                episode['interrupted'] = True
                                abort(payload.get('reason', 'episode_finished_before_completion'))
                        self._write(episodes, dict(payload, event='end',
                            training_objective=self.manifest['training_objective'],
                            complete=payload.get('complete', True),
                            source_episode_id=episode['source_episode_id'], final_decision_id=last_decision,
                            intervened=episode['intervened'], interrupted=episode['interrupted'],
                            human_interventions=episode['human_interventions'],
                            reward_provenance='operator_terminal_outcome' if
                                type(payload.get('success')) is bool and payload.get('complete', True) else None))
                        self.status.update(source_episode_id=None,
                            last_outcome=dict(label=payload.get('label'), success=payload.get('success'),
                                complete=payload.get('complete', True), interrupted=episode['interrupted'],
                                intervened=episode['intervened']))
                        episode = None
        except BaseException as exc:
            self.failure = str(exc)
