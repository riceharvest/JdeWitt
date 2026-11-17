"""Behavior-ranker data-path CPU fixtures: contracts and negative controls.

These exercise the data, validation and configuration surface of
tree_behavior_data/tree_ranker only. No GPU, no model, no training.
They prove the validators reject bad data; they are not model-quality
evidence and say nothing about trained-ranker behavior.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jdewitt.answers import AnswerStore
from jdewitt.data import digest
from jdewitt import tree_behavior_data as tbd
from jdewitt import tree_ranker as trk
from jdewitt.tree_cli import _seal

CHECKPOINTS = [0, 3, 7]
TOKENS = [101, 202, 303, 404]
RAW = {'checkpoints': CHECKPOINTS, 'dtype': 'bfloat16', 'refusal_tokens': [9, 8], 'reference_tokens': [7, 6],
       'ctx_size': 4096, 'max_tokens': 1024, 'refusal_floor': 150,
       'model_identity': {'model': 'fixture-only', 'revision': 'fixture-v1', 'architecture': 'llama',
                          'layers': 8, 'hidden_size': 16},
       'behavior': {'feature_tokens': TOKENS}, 'actions': ['refusal', 'none']}
SCHEMA = tbd.feature_schema(RAW, 'a' * 64, runtime_fingerprint={'os': 'fixture'})
WIDTH = SCHEMA['snapshot_width']
SLOT = 3 * WIDTH + 1
IDENTITY = {'version': 1, 'fixture': True}
POLICY = {'min_completion_probability': 0.8, 'max_refusal_probability': 0.1, 'max_evasion_probability': 0.1,
          'max_broken_probability': 0.1, 'max_disagreement': 0.2, 'max_feature_z': 4.0,
          'uncertainty_weight': 0.1, 'uncertain_policy': 'proxy'}
UTILITIES = {'completion': 1.0, 'refusal': -0.5, 'evasion': -0.4, 'broken': -1.0}
POLICY_KEYS = ('min_completion_probability', 'max_refusal_probability', 'max_evasion_probability',
               'max_broken_probability', 'max_disagreement', 'max_feature_z', 'uncertainty_weight',
               'uncertain_policy')


def history_up_to(checkpoint, offset=0.0):
    return [{'checkpoint': cp, 'values': [offset + cp + i for i in range(WIDTH)]}
            for cp in CHECKPOINTS if cp <= checkpoint]


def features(checkpoint, offset=0.0):
    # Real search truncates trajectory and baseline histories to the same depth.
    return tbd.encode_trajectory(history_up_to(checkpoint, offset), history_up_to(checkpoint), SCHEMA)


def row(row_id, checkpoint, outcome, recognition, family, prompt, group, fingerprint, offset=0.0):
    return {'id': row_id, 'prompt': prompt, 'family_id': family, 'language': 'en', 'kind': 'refusal',
            'checkpoint': checkpoint, 'features': features(checkpoint, offset), 'outcome': outcome,
            'recognition': recognition, 'intervention_groups': [group], 'direction_fingerprints': [fingerprint],
            'response_digest': 'd' * 64, 'label_source': {'kind': 'human', 'identity': 'fixture'}}


def test_encode_trajectory_fixed_width_and_zeroed_unobserved_slots():
    encoded = features(3)
    assert len(encoded) == SCHEMA['dimension'] == len(CHECKPOINTS) * SLOT
    assert encoded[2 * SLOT:3 * SLOT] == [0.0] * SLOT  # checkpoint 7 unobserved at depth 3
    assert features(7)[3 * SLOT - 1] == 1.0  # observed final slot carries its mask bit


def test_encode_trajectory_requires_equal_truncation_depth():
    with pytest.raises(ValueError, match='lengths differ'):
        tbd.encode_trajectory(history_up_to(3), history_up_to(7), SCHEMA)


def test_validate_rows_accepts_isolated_sane_rows():
    rows = [row('a', 7, 'completion', True, 'f1', 'p1', 'g1', 'd1'),
            row('b', 3, 'refusal', False, 'f2', 'p2', 'g2', 'd2')]
    assert tbd.validate_rows(rows, SCHEMA) == rows


@pytest.mark.parametrize('change', ['future-value', 'inverted-mask', 'unknown-outcome',
                                    'nonbool-recognition', 'empty-prompt', 'wrong-feature-width',
                                    'missing-field', 'duplicate-id'])
def test_validate_rows_rejects_corrupt_examples(change):
    rows = [row('a', 7, 'completion', True, 'f1', 'p1', 'g1', 'd1'),
            row('b', 3, 'refusal', False, 'f2', 'p2', 'g2', 'd2')]
    if change == 'future-value':
        rows[1]['features'][2 * SLOT] = 1.0  # checkpoint 7 slot is unobserved at depth 3
    elif change == 'inverted-mask':
        rows[0]['features'][3 * SLOT - 1] = 0.0  # observed slot cannot hide its mask bit
    elif change == 'unknown-outcome':
        rows[0]['outcome'] = 'answered'
    elif change == 'nonbool-recognition':
        rows[0]['recognition'] = 1
    elif change == 'empty-prompt':
        rows[0]['prompt'] = ' '
    elif change == 'wrong-feature-width':
        rows[0]['features'] = rows[0]['features'][:SCHEMA['dimension'] - 1]
    elif change == 'missing-field':
        rows[0].pop('response_digest')
    else:
        rows[1]['id'] = rows[0]['id']
    with pytest.raises(ValueError):
        tbd.validate_rows(rows, SCHEMA)


@pytest.mark.parametrize('extra', [{'completion_probability': 0.5}, {'abstained': True},
                                   {'refusal_probability': 0.5}])
def test_promising_gates(extra):
    base = {'abstained': False, 'completion_probability': 0.9, 'refusal_probability': 0.05,
            'evasion_probability': 0.05, 'broken_probability': 0.05, 'utility': 0.9, 'disagreement': 0.1}
    assert tbd.promising(base, POLICY)
    assert not tbd.promising({**base, **extra}, POLICY)


@pytest.mark.parametrize('change', ['drop-probability-key', 'unknown-uncertain-policy',
                                    'probability-out-of-range', 'negative-uncertainty-weight'])
def test_validate_policy_rejects(change):
    policy = dict(POLICY)
    if change == 'drop-probability-key':
        policy.pop('max_broken_probability')
    elif change == 'unknown-uncertain-policy':
        policy['uncertain_policy'] = 'guess'
    elif change == 'probability-out-of-range':
        policy['min_completion_probability'] = 1.5
    else:
        policy['max_disagreement'] = -0.1
    with pytest.raises(ValueError):
        tbd.validate_policy(policy)


def make_job(directory, name):
    job = directory / name
    behavior = job / 'behavior' / digest('task-1')
    (behavior / 'answers').mkdir(parents=True)
    rows = [{'id': f'ex-{idx}', 'prompt': f'p-{name}-{idx}', 'source_task_id': 'task-1',
             'family_id': f'fam-{name}-{idx}', 'language': 'en', 'kind': 'refusal', 'checkpoint': cp,
             'path': [{'action': 'refusal', 'checkpoint': cp, 'strength': 0.5}], 'features': features(cp),
             'proxy_metrics': {}, 'rejected': False, 'intervention_groups': [f'g-{name}-{idx}'],
             'direction_fingerprints': [f'd-{name}-{idx}']} for idx, cp in ((0, 7), (1, 3))]
    _seal(behavior / 'features.json', IDENTITY,
          {'source_task_id': 'task-1', 'source_prompt': 'p', 'feature_schema': SCHEMA, 'rows': rows})
    store = AnswerStore(behavior / 'answers', rows, {**IDENTITY, 'feature_rows_digest': digest(rows)})
    labels = []
    for r, outcome, recognition in ((rows[0], 'completion', True), (rows[1], 'refusal', False)):
        store.put(r, {'text': 'full answer', 'finish_reason': 'stop'})
        labels.append({'id': r['id'], 'response_digest': digest(store.get(r)), 'outcome': outcome,
                       'recognition': recognition, 'label_source': {'kind': 'human', 'identity': 'fixture'},
                       'exclude_reason': None})
    labels_path = job / 'labels.jsonl'
    labels_path.write_text(''.join(json.dumps(x) + '\n' for x in labels))
    return job, labels_path


def write_spec(directory, sources, output):
    spec_path = directory / 'spec.json'
    spec_path.write_text(json.dumps({'sources': [{'job': str(j), 'labels': str(lp), 'split': s}
                                                 for (j, lp), s in sources], 'output': str(output)}))
    return spec_path


def three_jobs(tmp_path):
    return [make_job(tmp_path, f'job{i}') for i in range(3)]


def test_prepare_ranker_data_joins_bound_labels(tmp_path):
    jobs = three_jobs(tmp_path)
    spec_path = write_spec(tmp_path, list(zip(jobs, ('train', 'calibration', 'test'))), tmp_path / 'dataset')
    result = tbd.prepare_ranker_data(spec_path)
    manifest = json.loads((tmp_path / 'dataset' / 'manifest.json').read_text())
    assert manifest['record_digest']
    assert manifest['isolation'].startswith('prompt+authored-family')
    assert result['counts']['train'] == {'completion': 1, 'refusal': 1}
    assert result['status'].endswith('no model trained')


def test_prepare_ranker_data_rejects_missing_splits(tmp_path):
    jobs = three_jobs(tmp_path)
    spec_path = write_spec(tmp_path, list(zip(jobs, ('train',))), tmp_path / 'dataset')
    with pytest.raises(ValueError, match='Empty behavior'):
        tbd.prepare_ranker_data(spec_path)


def test_prepare_ranker_data_rejects_label_not_bound_to_answer(tmp_path):
    jobs = three_jobs(tmp_path)
    labels_path = jobs[0][1]
    lines = Path(labels_path).read_text().splitlines()
    label = json.loads(lines[1])
    label['response_digest'] = 'dead' * 8
    Path(labels_path).write_text(lines[0] + '\n' + json.dumps(label) + '\n')
    spec_path = write_spec(tmp_path, list(zip(jobs, ('train', 'calibration', 'test'))), tmp_path / 'dataset')
    with pytest.raises(ValueError, match='not bound'):
        tbd.prepare_ranker_data(spec_path)


def test_prepare_ranker_data_rejects_unlabelled_row_without_exclusion(tmp_path):
    jobs = three_jobs(tmp_path)
    job, labels_path = jobs[0]
    behavior = job / 'behavior' / digest('task-1')
    rows = json.loads((behavior / 'features.json').read_text())['payload']['rows']
    store = AnswerStore.open(behavior / 'answers', rows)
    unlabelled = {'id': 'ex-1', 'response_digest': digest(store.get(rows[1])), 'outcome': None,
                  'recognition': None, 'label_source': None, 'exclude_reason': None}
    Path(labels_path).write_text(Path(labels_path).read_text().splitlines()[0] + '\n'
                                 + json.dumps(unlabelled) + '\n')
    spec_path = write_spec(tmp_path, list(zip(jobs, ('train', 'calibration', 'test'))), tmp_path / 'dataset')
    with pytest.raises(ValueError, match='exclusion'):
        tbd.prepare_ranker_data(spec_path)


def test_prepare_ranker_data_rejects_truncated_completion_label(tmp_path):
    job = tmp_path / 'jobT'
    behavior = job / 'behavior' / digest('task-1')
    (behavior / 'answers').mkdir(parents=True)
    rows = [{'id': 'ex-T', 'prompt': 'pT', 'source_task_id': 'task-1', 'family_id': 'famT', 'language': 'en',
             'kind': 'refusal', 'checkpoint': 7, 'path': [], 'features': features(7), 'proxy_metrics': {},
             'rejected': False, 'intervention_groups': ['gT'], 'direction_fingerprints': ['dT']}]
    _seal(behavior / 'features.json', IDENTITY,
          {'source_task_id': 'task-1', 'source_prompt': 'pT', 'feature_schema': SCHEMA, 'rows': rows})
    store = AnswerStore(behavior / 'answers', rows, {**IDENTITY, 'feature_rows_digest': digest(rows)})
    store.put(rows[0], {'text': 'half', 'finish_reason': 'length'})
    labels_path = job / 'labels.jsonl'
    labels_path.write_text(json.dumps({'id': 'ex-T', 'response_digest': digest(store.get(rows[0])),
                                       'outcome': 'completion', 'recognition': True,
                                       'label_source': {'kind': 'human', 'identity': 'fixture'},
                                       'exclude_reason': None}) + '\n')
    others = [make_job(tmp_path, f'job{i}') for i in range(2)]
    spec_path = write_spec(tmp_path, [((job, labels_path), 'train')] +
                           list(zip(others, ('calibration', 'test'))), tmp_path / 'dataset')
    with pytest.raises(ValueError, match='truncated'):
        tbd.prepare_ranker_data(spec_path)


def test_load_dataset_rejects_tampered_manifest(tmp_path):
    jobs = three_jobs(tmp_path)
    spec_path = write_spec(tmp_path, list(zip(jobs, ('train', 'calibration', 'test'))), tmp_path / 'dataset')
    tbd.prepare_ranker_data(spec_path)
    manifest_path = tmp_path / 'dataset' / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['counts']['train']['completion'] = 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest integrity'):
        tbd.load_dataset(manifest_path, names=('train',))


def test_load_dataset_rejects_split_content_mismatch(tmp_path):
    jobs = three_jobs(tmp_path)
    spec_path = write_spec(tmp_path, list(zip(jobs, ('train', 'calibration', 'test'))), tmp_path / 'dataset')
    tbd.prepare_ranker_data(spec_path)
    (tmp_path / 'dataset' / 'train.jsonl').write_text('{"id":"evil"}\n')
    with pytest.raises(ValueError, match='path/hash'):
        tbd.load_dataset(tmp_path / 'dataset' / 'manifest.json', names=('train',))


def write_config(directory, name, overrides):
    config = {'dataset': 'd', 'output': 'o', 'hidden_size': 8, 'seeds': [1, 2], 'epochs': 2, 'batch_size': 4,
              'learning_rate': 0.01, 'weight_decay': 0.0, 'max_examples': 100, 'max_pairs_per_prompt': 8,
              'pairs_per_batch': 4, 'pair_weight': 0.5, 'utility_weight': 1.0, 'outcome_weight': 1.0,
              'recognition_weight': 0.5, 'outcome_utilities': dict(UTILITIES), 'calibration_steps': 20,
              'calibration_learning_rate': 0.01}
    if 'drop' in name:
        config.pop('epochs')
    else:
        config.update(overrides)
    path = directory / name
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize('case', ['drop', 'single-seed', 'duplicate-seeds', 'utility-above-completion',
                                  'zero-learning-rate', 'zero-hidden'])
def test_training_config_rejects_incomplete_or_dangerous_spec(tmp_path, case):
    overrides = {'seeds': [7]} if case == 'single-seed' else \
        {'seeds': [7, 7]} if case == 'duplicate-seeds' else \
        {'outcome_utilities': {'completion': 1.0, 'refusal': 2.0, 'evasion': -0.4, 'broken': -1.0}} \
        if case == 'utility-above-completion' else \
        {'learning_rate': 0} if case == 'zero-learning-rate' else \
        {'hidden_size': 0} if case == 'zero-hidden' else {}
    path = write_config(tmp_path, case + '.json', overrides)
    with pytest.raises(ValueError):
        trk._training_config(path)


def test_training_config_accepts_explicit_complete_spec(tmp_path):
    path = write_config(tmp_path, 'complete.json', {})
    raw, dataset, output = trk._training_config(path)
    assert raw['outcome_utilities']['completion'] > max(
        v for k, v in raw['outcome_utilities'].items() if k != 'completion')
    assert dataset == path.parent / 'd' and output == path.parent / 'o'


def test_pairs_require_same_prompt_depth_preferences():
    rows = [{'prompt': 'p1', 'checkpoint': 0, 'outcome': 'completion'},
            {'prompt': 'p1', 'checkpoint': 0, 'outcome': 'refusal'},
            {'prompt': 'p2', 'checkpoint': 0, 'outcome': 'completion'}]
    pairs = trk._pairs(rows, UTILITIES, 10, np.random.default_rng(3))
    assert [(a, b) for a, b, _ in pairs] == [(0, 1)]
    assert pairs[0][2] == 1.0


def test_pairs_refuse_to_train_without_preferences():
    rows = [{'prompt': 'p1', 'checkpoint': 0, 'outcome': 'completion'},
            {'prompt': 'p2', 'checkpoint': 0, 'outcome': 'completion'}]
    with pytest.raises(ValueError, match='cannot train'):
        trk._pairs(rows, UTILITIES, 10, np.random.default_rng(3))


def test_behavior_ranker_refuses_cpu_device():
    with pytest.raises(ValueError, match='B70'):
        trk.BehaviorRanker({'format': 'x'}, {'dimension': 1}, torch.device('cpu'), 1)


def behavior_config(mode, artifact, policy, **behavior_overrides):
    spec = {'mode': mode, 'feature_tokens': TOKENS, 'intervention_groups': {'refusal': 'g'},
            'max_feature_values': 10 ** 6, 'max_ranker_bytes': 10 ** 7, 'artifact': artifact,
            'policy': policy, 'max_decodes_per_prompt': 50, 'include_rejected': True}
    spec.update(behavior_overrides)
    return dict(RAW, actions=['refusal'], behavior=spec)


def test_validate_behavior_accepts_collection_without_ranker(tmp_path):
    assert tbd.validate_behavior(behavior_config('collect', None, None), tmp_path, {}, 10) is None


@pytest.mark.parametrize('change', ['rank-without-artifact', 'collect-with-policy',
                                    'rank-with-collection-flags', 'missing-intervention-group',
                                    'feature-budget-shortfall', 'collection-decode-cap-too-small'])
def test_validate_behavior_rejects(tmp_path, change):
    expansions = 10
    raw = behavior_config('collect', None, None)
    if change == 'collect-with-policy':
        raw = behavior_config('collect', None, POLICY)
    elif change in ('rank-without-artifact', 'rank-with-collection-flags'):
        raw = behavior_config('rank', None, POLICY, include_rejected=False, max_decodes_per_prompt=None)
        if change == 'rank-with-collection-flags':
            artifact = tmp_path / 'artifact.json'
            artifact.write_text(json.dumps({'format': 'jspace-behavior-ranker-v1'}))
            raw['behavior']['artifact'] = 'artifact.json'
            raw['behavior']['include_rejected'] = True
    elif change == 'missing-intervention-group':
        raw['behavior']['intervention_groups'] = {}
    elif change == 'feature-budget-shortfall':
        expansions = 10 ** 9
    else:
        raw['behavior']['max_decodes_per_prompt'] = 1
    with pytest.raises(ValueError):
        tbd.validate_behavior(raw, tmp_path, {}, expansions)
