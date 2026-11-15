"""Strict, portable, provenance-bearing replay. NPZ files contain no pickles."""
import json
from pathlib import Path

import numpy as np

from .contracts import ActionNormalizer, ReplaySpec, normalization_fingerprint

REQUIRED_ARRAYS = ('obs', 'next_obs', 'actions', 'base_actions', 'next_base_actions',
                   'reward_return', 'bootstrap_discount', 'rewards', 'terminated',
                   'source_episode_ids')

TRAINING_OBJECTIVES = ('gear_insertion', 'gear_and_pin')
LEGACY_TASK = 'put the gear in the carrier and put a pin in it'
REWARD_SEMANTICS = {
    'gear_insertion': 'Terminal reward 1 iff the gear is fully seated in the carrier; no pin is required. Otherwise 0.',
    'gear_and_pin': 'Terminal reward 1 iff the gear is seated in the carrier and the pin is inserted. Otherwise 0.',
}


def _declares_legacy_task(metadata):
    contract = metadata.get('contract')
    return (metadata.get('task') == LEGACY_TASK or
            isinstance(contract, dict) and contract.get('task') == LEGACY_TASK)


def training_objective_from_metadata(metadata):
    """Require an explicit reward task; recognize only the exact old task text."""
    if not isinstance(metadata, dict):
        raise ValueError('Replay objective metadata must be a mapping')
    objective = metadata.get('training_objective')
    if 'training_objective' not in metadata and _declares_legacy_task(metadata):
        objective = 'gear_and_pin'
    if objective not in TRAINING_OBJECTIVES:
        raise ValueError('Replay requires training_objective gear_insertion or gear_and_pin')
    return objective


def validate_replay(arrays, metadata, expected_spec=None):
    if not isinstance(metadata, dict) or metadata.get('schema_version') != 1:
        raise ValueError('Unsupported replay metadata schema')
    if (metadata.get('kind') != 'faithful_policy_replay' or
            metadata.get('training_eligible') is not True or metadata.get('synthetic') is not False):
        raise ValueError('Only explicitly eligible, nonsynthetic faithful policy replay can train')
    objective = training_objective_from_metadata(metadata)
    spec = ReplaySpec(**metadata['spec'])
    if spec.C > 50:
        raise ValueError('Replay chunk exceeds the frozen pi0.5 horizon of 50 waypoints')
    if expected_spec is not None and spec != expected_spec:
        raise ValueError('Replay checkpoint, processor, or dimensions differ from expected spec')
    normalization = metadata.get('normalization', {})
    ActionNormalizer.from_stats(normalization.get('action'))
    ActionNormalizer.from_stats(normalization.get('state'))
    if normalization.get('checkpoint_id') != spec.checkpoint_id or normalization.get('processor_id') != spec.processor_id:
        raise ValueError('Normalization identity differs from replay checkpoint/processor')
    if metadata.get('normalization_sha256') != normalization_fingerprint(normalization):
        raise ValueError('Normalization fingerprint mismatch')
    if any(name not in arrays for name in REQUIRED_ARRAYS):
        raise ValueError('Replay is missing required arrays')
    for key, value in arrays.items():
        if np.asarray(value).dtype.hasobject:
            raise ValueError(f'{key}: object arrays and pickle payloads are forbidden')
    actions = np.asarray(arrays['actions'])
    if actions.ndim != 3:
        raise ValueError('actions must be [M,C,14]')
    M = len(actions)
    if not M:
        raise ValueError('Faithful replay contains no valid transitions')
    shapes = {'obs': (M, spec.obs_dim), 'next_obs': (M, spec.obs_dim),
              'actions': (M, spec.C, 14), 'base_actions': (M, spec.N, spec.C, 14),
              'next_base_actions': (M, spec.N, spec.C, 14), 'rewards': (M, spec.C),
              'reward_return': (M,), 'bootstrap_discount': (M,),
              'terminated': (M,), 'source_episode_ids': (M,)}
    for key, shape in shapes.items():
        array = np.asarray(arrays[key])
        if array.shape != shape or array.dtype.hasobject:
            raise ValueError(f'{key}: expected shape {shape}, no object arrays')
        if key == 'source_episode_ids':
            if array.dtype.kind not in ('U', 'S'):
                raise ValueError('Source IDs must be plain strings')
        elif key == 'terminated':
            if array.dtype.kind != 'b':
                raise ValueError('terminated must be boolean')
        elif array.dtype.kind != 'f' or not np.isfinite(array).all():
            raise ValueError(f'{key}: require finite floating-point data')
    expected_return = (arrays['rewards'] * np.power(spec.gamma, np.arange(spec.C))).sum(axis=1)
    expected_discount = (spec.gamma ** spec.C) * (~arrays['terminated'])
    if not np.allclose(arrays['reward_return'], expected_return, atol=1e-6, rtol=1e-6):
        raise ValueError('reward_return must sum gamma**i times actual prefix rewards')
    if not np.allclose(arrays['bootstrap_discount'], expected_discount, atol=1e-6, rtol=1e-6):
        raise ValueError('bootstrap_discount must be gamma**C times nontermination')
    provenance = metadata.get('transitions', [])
    if len(provenance) != M:
        raise ValueError('Every transition requires provenance')
    for i, row in enumerate(provenance):
        row_objective = row.get('training_objective')
        if 'training_objective' not in row and _declares_legacy_task(metadata):
            row_objective = 'gear_and_pin'
        if row_objective != objective:
            raise ValueError('Every transition must declare the same training_objective as its replay')
        if row.get('source_episode_id') != str(arrays['source_episode_ids'][i]):
            raise ValueError('Source grouping/provenance mismatch')
        if (row.get('complete') is not True or row.get('simulation') is not False or
                row.get('control_source') != 'policy' or row.get('contiguous_policy') is not True):
            raise ValueError('Incomplete, simulated, or noncontiguous policy transitions are ineligible')
        if type(row.get('task_success')) is not bool:
            raise ValueError('Explicit operator task outcome is mandatory')
        if row.get('intervened') is not False or row.get('human_interventions') != 0:
            raise ValueError('Assisted episodes cannot assign task outcome credit to policy prefixes')
        if row.get('source_category') not in ('kept', 'review'):
            raise ValueError('Source category must be kept or explicitly approved review')
        if row.get('source_category') == 'review' and row.get('review_approved') is not True:
            raise ValueError('Review sources require explicit approval')
        if row.get('action_provenance') != 'executed_policy_waypoints':
            raise ValueError('Faithful replay requires executed policy waypoints, not measured-next-pose proxies')
        if row.get('candidate_provenance') != 'pi05_inference':
            raise ValueError('Genuine pi05 inference candidates are mandatory')
        if row.get('checkpoint_id') != spec.checkpoint_id or row.get('processor_id') != spec.processor_id:
            raise ValueError('Transition checkpoint/processor identity mismatch')
        rewards = np.asarray(arrays['rewards'][i])
        expected = np.zeros(spec.C, dtype=np.float32)
        if arrays['terminated'][i]:
            expected[-1] = float(row['task_success'])
        if not np.array_equal(rewards, expected):
            raise ValueError('Sparse operator reward belongs only on the final executed terminal waypoint')
    for source in np.unique(arrays['source_episode_ids']):
        indices = np.flatnonzero(arrays['source_episode_ids'] == source)
        if len({provenance[i]['task_success'] for i in indices}) != 1:
            raise ValueError('Task outcome provenance must agree throughout an original episode')
        terminal = indices[arrays['terminated'][indices]]
        if len(terminal) != 1 or terminal[0] != indices[-1]:
            raise ValueError('Each complete original episode needs exactly one final terminal transition')
    if 'splits' in metadata:
        splits = metadata['splits']
        train = np.asarray(splits.get('train_indices', []), dtype=np.int64)
        evaluation = np.asarray(splits.get('eval_indices', []), dtype=np.int64)
        indices = np.concatenate([train, evaluation])
        if indices.ndim != 1 or not np.array_equal(np.sort(indices), np.arange(M)):
            raise ValueError('Replay splits must partition every transition exactly once')
        if set(arrays['source_episode_ids'][train]) & set(arrays['source_episode_ids'][evaluation]):
            raise ValueError('Original source episodes leak between training and evaluation')
    for key in ('editor_train_valid', 'transition_valid'):
        if key in arrays and (arrays[key].shape != (M,) or arrays[key].dtype.kind != 'b'):
            raise ValueError(f'{key} must be a boolean vector')
    return spec


def save_replay(path, arrays, metadata):
    validate_replay(arrays, metadata)
    path = Path(path)
    if path.exists():
        raise FileExistsError('Choose a new replay path; existing datasets are preserved')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload['_metadata_json'] = np.frombuffer(json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **payload)


def load_replay(path, expected_spec=None):
    with np.load(path, allow_pickle=False) as archive:
        if '_metadata_json' not in archive:
            raise ValueError('Replay metadata is missing')
        encoded = archive['_metadata_json']
        if encoded.dtype != np.uint8 or encoded.ndim != 1:
            raise ValueError('Malformed replay metadata encoding')
        metadata = json.loads(encoded.tobytes().decode())
        arrays = {key: archive[key] for key in archive.files if key != '_metadata_json'}
    validate_replay(arrays, metadata, expected_spec)
    # Legacy files remain readable only with explicit original task evidence;
    # expose the resolved identity to downstream training and bundle binding.
    metadata['training_objective'] = training_objective_from_metadata(metadata)
    return arrays, metadata


def grouped_split(source_episode_ids, eval_fraction=.2, seed=0):
    """All corrections/crops/re-exports from an original episode share one split."""
    ids = np.asarray(source_episode_ids)
    if ids.ndim != 1 or ids.dtype.kind not in ('U', 'S') or not 0 <= eval_fraction < 1:
        raise ValueError('Expected source IDs and eval_fraction in [0,1)')
    groups = np.unique(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    count = int(round(len(groups) * eval_fraction))
    if eval_fraction > 0 and len(groups) > 1:
        count = max(1, min(len(groups) - 1, count))
    selected = np.isin(ids, groups[:count])
    return np.flatnonzero(~selected), np.flatnonzero(selected)
