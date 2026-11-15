#!/usr/bin/env python3
"""Optional local EXPO inference, isolated from the production pi0.5 service.

No hardware APIs, remote checkpoint downloads, or training override are exposed.
The base-only mode also serves genuine candidates for offline cache generation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import base64
import io
import re
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads((ROOT / 'infra/inference/contract.json').read_text())
NAMES = [name for side in ('left', 'right') for name in
         [*[f'{side}.joint{i}' for i in range(1, 7)], f'{side}.gripper_rad']]
NUM_CANDIDATES = 8
DEFAULT_CHUNK_SIZE = 1
HORIZON = 50
TRAINING_OBJECTIVES = frozenset(('gear_insertion', 'gear_and_pin'))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def to_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    import numpy as np
    return np.asarray(value)


class Runtime:
    """Lazy runtime. Injected factories let CPU tests exercise the full contract."""

    def __init__(self, checkpoint, rl_checkpoint=None, device='cuda', *,
                 guard=None, adapter_factory=None, learner_factory=None,
                 normalizer_factory=None, explore=False, seed=0, prefix_length=None):
        if prefix_length is not None and (type(prefix_length) is not int or not 1 <= prefix_length <= HORIZON):
            raise ValueError('Prefix length must be an integer from 1 to 50')
        self.requested_prefix_length = prefix_length
        self.chunk_size = prefix_length if prefix_length is not None else DEFAULT_CHUNK_SIZE
        self.checkpoint = str(checkpoint)
        self.rl_checkpoint = str(rl_checkpoint) if rl_checkpoint else None
        self.device = device
        self.status = 'unloaded'
        self.error = None
        self.lock = threading.Lock()
        self.guard = guard
        self.adapter_factory = adapter_factory
        self.learner_factory = learner_factory
        self.normalizer_factory = normalizer_factory
        self.rl_model_id = None
        self.adapter = self.learner = self.normalizer = None
        self.resource_lease = None
        self.resource_report = None
        self.normalization = None
        self.normalizer_hash = None
        self.encoder = None
        self.policy_bundle_id = None
        self.training_objective = None
        self.training_reward_semantics = None
        self.run_config = None
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('Run seed must be an integer in [0,2**63)')
        self.seed_rng = random.Random(seed)
        self.explore = bool(explore)

    def load(self):
        """Fail closed before any model is constructed if the GPU is occupied."""
        try:
            self.status = 'checking_resources'
            if self.guard is None:
                from infra.rl.resources import require_idle_gpu
                self.guard = require_idle_gpu
            self.resource_lease = self.guard(device=self.device)
            self.resource_report = self.resource_lease.__enter__()
            checkpoint = Path(self.checkpoint).resolve(strict=True)
            if not checkpoint.is_dir():
                raise ValueError('Base checkpoint must be a local directory')
            self.checkpoint = str(checkpoint)
            if self.rl_checkpoint:
                self.rl_checkpoint = str(Path(self.rl_checkpoint).resolve(strict=True))
                self.rl_model_id = file_sha256(self.rl_checkpoint)
            self.status = 'loading'
            if self.adapter_factory is None:
                from infra.rl.pi05_adapter import PI05Adapter
                self.adapter_factory = PI05Adapter
            self.adapter = self.adapter_factory(self.checkpoint, device=self.device)
            self.adapter.load()
            if self.normalizer_factory is None:
                from infra.rl.contracts import ActionNormalizer, normalization_fingerprint
                from infra.rl.observations import ObservationEncoder
                self.normalization = self.adapter.normalization
                self.normalizer = ActionNormalizer.from_stats(self.normalization['action'])
                self.encoder = ObservationEncoder(ActionNormalizer.from_stats(self.normalization['state']))
                self.normalizer_hash = normalization_fingerprint(self.normalization)
            else:
                self.normalizer = self.normalizer_factory(self.checkpoint)
                self.normalizer_hash = getattr(self.normalizer, 'hash', None)
            if self.rl_checkpoint:
                self._validate_rl_provenance()
                if self.learner_factory is None:
                    from infra.rl.learner import EXPOLearner
                    self.learner_factory = EXPOLearner.load
                self.learner = self.learner_factory(self.rl_checkpoint, device=self.device)
                checkpoint_metadata = getattr(self.learner, 'checkpoint_metadata', {})
                if (checkpoint_metadata.get('training_objective') != self.training_objective or
                        checkpoint_metadata.get('reward_semantics') != self.training_reward_semantics):
                    raise ValueError('Loaded RL checkpoint objective/reward differs from its run and replay')
                loaded_config = json.loads(json.dumps(vars(self.learner.config), allow_nan=False))
                if loaded_config != self.run_config:
                    raise ValueError('Loaded RL checkpoint config differs from the completed run')
                if self.learner.config.chunk_size != self.chunk_size:
                    raise ValueError('Serving prefix must match the trained EXPO checkpoint chunk length')
                if self.learner.config.action_dim != len(NAMES):
                    raise ValueError('EXPO checkpoint must use the 14 physical actions')
                if self.learner.config.obs_dim != self.encoder.obs_dim:
                    raise ValueError('EXPO checkpoint observation encoder mismatch')
                expected_lower, expected_upper = self._expected_action_bounds()
                if (list(self.learner.config.action_lower or ()) != expected_lower or
                        list(self.learner.config.action_upper or ()) != expected_upper):
                    raise ValueError('EXPO checkpoint physical safety bounds mismatch')
                if file_sha256(self.rl_checkpoint) != self.rl_model_id:
                    raise ValueError('EXPO checkpoint changed during loading')
            identity = dict(checkpoint_id=self.adapter.checkpoint_id,
                            processor_id=self.adapter.processor_id,
                            normalization_sha256=self.normalizer_hash,
                            rl_model_id=self.rl_model_id, training_objective=self.training_objective,
                            C=self.chunk_size, N=NUM_CANDIDATES)
            self.policy_bundle_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            self.status = 'ready'
        except Exception as exc:
            self.error = str(exc)
            self.status = 'failed'
            self.close()
            raise

    def _validate_rl_provenance(self):
        """Only a completed real replay run for this exact frozen base can serve."""
        from infra.rl.contracts import normalization_fingerprint
        checkpoint = Path(self.rl_checkpoint)
        report = json.loads((checkpoint.parent / 'run.json').read_text())
        if report.get('synthetic') is not False or report.get('mode') != 'frozen_base_expo_residual_rl':
            raise ValueError('Serving requires a real frozen-base EXPO run; synthetic checkpoints are ineligible')
        step = re.fullmatch(r'checkpoint-(\d+)\.pt', checkpoint.name)
        if not step or report.get('completed_steps', -1) < int(step.group(1)):
            raise ValueError('RL checkpoint must belong to a completed, recorded training run')
        metadata = report.get('replay_metadata', {})
        if (metadata.get('synthetic') is not False or metadata.get('training_eligible') is not True or
                metadata.get('kind') != 'faithful_policy_replay'):
            raise ValueError('RL checkpoint has no training-eligible real replay provenance')
        objective = report.get('training_objective')
        if (not isinstance(objective, str) or objective not in TRAINING_OBJECTIVES or
                metadata.get('training_objective') != objective):
            raise ValueError('RL run and replay require the same explicit training objective')
        from infra.rl.replay import REWARD_SEMANTICS
        if report.get('reward_semantics') != REWARD_SEMANTICS[objective]:
            raise ValueError('RL run reward semantics differ from the training objective')
        self.training_objective = objective
        self.training_reward_semantics = REWARD_SEMANTICS[objective]
        config = report.get('config', {})
        trained_chunk_size = config.get('chunk_size')
        if type(trained_chunk_size) is not int or not 1 <= trained_chunk_size <= HORIZON:
            raise ValueError('Trained EXPO chunk length must be an integer from 1 to 50')
        if self.requested_prefix_length is not None and self.requested_prefix_length != trained_chunk_size:
            raise ValueError('Requested prefix differs from the trained EXPO checkpoint chunk length')
        self.chunk_size = trained_chunk_size
        spec = metadata.get('spec', {})
        expected = dict(C=self.chunk_size, N=NUM_CANDIDATES, obs_dim=self.encoder.obs_dim,
                        checkpoint_id=self.adapter.checkpoint_id,
                        processor_id=self.adapter.processor_id)
        if any(spec.get(key) != value for key, value in expected.items()):
            raise ValueError('RL checkpoint replay specification differs from the deployed base')
        if (metadata.get('normalization_sha256') != self.normalizer_hash or
                normalization_fingerprint(metadata.get('normalization')) != self.normalizer_hash):
            raise ValueError('RL checkpoint quantile snapshot differs from the deployed processor')
        if any(config.get(key) != value for key, value in
               dict(chunk_size=self.chunk_size, action_dim=len(NAMES), obs_dim=self.encoder.obs_dim).items()):
            raise ValueError('RL run config differs from the serving contract')
        lower, upper = self._expected_action_bounds()
        if config.get('action_lower') != lower or config.get('action_upper') != upper:
            raise ValueError('RL run gripper bounds differ from the deployed processor')
        self.run_config = config

    def _expected_action_bounds(self):
        import numpy as np
        normalized_lower = self.normalizer.normalize(np.zeros(len(NAMES), dtype=np.float32))
        normalized_upper = self.normalizer.normalize(np.full(len(NAMES), 2., dtype=np.float32))
        lower, upper = [None] * len(NAMES), [None] * len(NAMES)
        for index in (6, 13):
            lower[index], upper[index] = float(normalized_lower[index]), float(normalized_upper[index])
        return lower, upper

    def close(self):
        if self.resource_lease is not None:
            self.resource_lease.__exit__(None, None, None)
            self.resource_lease = None

    def health(self):
        return dict(status=self.status, error=self.error, checkpoint=self.checkpoint,
                    checkpoint_id=getattr(self.adapter, 'checkpoint_id', None),
                    processor_id=getattr(self.adapter, 'processor_id', None),
                    normalizer_hash=self.normalizer_hash,
                    rl_model_id=self.rl_model_id, mode='expo' if self.rl_checkpoint else 'base',
                    training_objective=self.training_objective,
                    policy_bundle_id=self.policy_bundle_id,
                    num_candidates=NUM_CANDIDATES, chunk_size=self.chunk_size,
                    explore=self.explore,
                    mock=False, contract=CONTRACT, names=NAMES,
                    resources=self.resource_report)

    def infer(self, payload):
        if self.status != 'ready':
            raise RuntimeError(f'Runtime is {self.status}')
        if not isinstance(payload, dict) or 'sequence' not in payload:
            raise ValueError('Expected an inference request with sequence')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Inference busy')
        try:
            import numpy as np
            seed = payload.get('seed', self.seed_rng.randrange(2**63))
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError('Candidate seed must be an integer in [0,2**63)')
            candidates = self.adapter.infer_candidates(payload, n=NUM_CANDIDATES, seed=seed)
            physical = to_numpy(candidates.physical)
            normalized = to_numpy(candidates.normalized)
            if (physical.ndim != 3 or physical.shape[0] != NUM_CANDIDATES or
                    physical.shape[1] != HORIZON or physical.shape[2] != len(NAMES) or
                    normalized.shape != physical.shape or
                    not np.isfinite(physical).all() or not np.isfinite(normalized).all()):
                raise RuntimeError('Invalid pi0.5 candidate batch')
            selected_index, edited, scores = 0, False, None
            actions = physical[0, :self.chunk_size].copy()
            if self.learner is not None:
                from PIL import Image
                decoded = {}
                for name, encoded in payload['images'].items():
                    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                        decoded[name] = np.asarray(image).copy()
                obs = self.encoder.encode(dict(state=payload['state'], images=decoded))
                selection = self.learner.select_actions(obs[None],
                    normalized[None, :, :self.chunk_size], deterministic=not self.explore)
                selected_index = int(to_numpy(selection['selected'])[0])
                edited = bool(to_numpy(selection['is_edited'])[0])
                scores = to_numpy(selection['scores'])[0].tolist()
                actions = to_numpy(self.adapter.denormalize_actions(selection['actions'][0]))
            if actions.shape != (self.chunk_size, len(NAMES)) or not np.isfinite(actions).all():
                raise RuntimeError('Invalid selected physical action prefix')
            from infra.inference.action_limits import clamp_gripper_actions
            commanded, corrections = clamp_gripper_actions(actions)
            observation_id = payload.get('observation_id', candidates.observation_id)
            provenance = dict(kind='pi05_inference', checkpoint_id=candidates.checkpoint_id,
                              processor_id=candidates.processor_id, sampling='independent',
                              observation_id=observation_id, observation_hash=candidates.observation_id,
                              seed=candidates.seed)
            return dict(sequence=payload['sequence'], actions=commanded.tolist(),
                        selected_physical_chunk=actions.tolist(),
                        current_physical_chunk=physical[0, :self.chunk_size].tolist(),
                        base_actions=physical.tolist(), candidates_provenance=provenance,
                        observation_id=observation_id,
                        checkpoint=self.checkpoint, checkpoint_id=candidates.checkpoint_id,
                        processor_id=candidates.processor_id,
                        normalizer_hash=self.normalizer_hash,
                        rl_model_id=self.rl_model_id, selected_candidate=selected_index,
                        training_objective=self.training_objective,
                        policy_bundle_id=self.policy_bundle_id,
                        selected_is_edited=edited, candidate_scores=scores,
                        num_candidates=NUM_CANDIDATES, chunk_size=self.chunk_size,
                        explore=self.explore, seed=seed,
                        fps=CONTRACT['fps'], mock=False, limit_corrections=corrections)
        finally:
            self.lock.release()

def make_handler(runtime):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send(200, runtime.health()) if self.path == '/health' else self.send(404, {})

        def do_POST(self):
            if self.path != '/infer':
                return self.send(404, {})
            if runtime.status != 'ready':
                return self.send(503, dict(error=runtime.status))
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 2_000_000:
                    raise ValueError('Invalid body size')
                self.connection.settimeout(5)
                self.send(200, runtime.infer(json.loads(self.rfile.read(size))))
            except (ValueError, KeyError, TypeError) as exc:
                self.send(400, dict(error=str(exc)))
            except Exception as exc:
                self.send(503, dict(error=str(exc)))
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Existing local frozen pi0.5 checkpoint')
    parser.add_argument('--rl-checkpoint', help='Optional locally trained EXPO learner; its complete chunk length is required')
    parser.add_argument('--prefix-length', type=int,
                        help='Executed prefix 1..50; default 1 in base mode or the trained learner chunk length')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--seed', type=int, default=0, help='Reproducible run seed; successive base draws remain independent')
    parser.add_argument('--explore', action='store_true', help='Sample stochastic editor proposals during collection')
    args = parser.parse_args()
    if args.port == 8765:
        parser.error('Port 8765 belongs to the existing production service')
    runtime = Runtime(args.checkpoint, args.rl_checkpoint, explore=args.explore, seed=args.seed,
                      prefix_length=args.prefix_length)
    runtime.load()
    try:
        ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(runtime)).serve_forever()
    finally:
        runtime.close()


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
