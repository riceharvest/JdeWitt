# SPDX-License-Identifier: AGPL-3.0-or-later
"""Trajectory observations, optional learned ranking, and full-answer collection.

Import is metadata-only. All model operations happen only inside an explicitly
invoked B70 tree job. Independent annotations are never inferred here.
"""
import json
import math
from pathlib import Path

from .answers import AnswerStore, atomic_json
from .data import digest
from .tree_behavior_data import encode_trajectory, feature_schema, checked_sealed


class TrajectoryReadout:
    def __init__(self, base, raw, lens_sha256, artifact_path=None, runtime_fingerprint=None):
        import torch
        self.base, self.spec = base, raw['behavior']
        self.schema = feature_schema(raw, lens_sha256, runtime_fingerprint)
        self.policy = self.spec['policy']
        tokens = self.spec['feature_tokens']
        if any(t >= len(base.head) for t in tokens):
            raise ValueError('Behavior feature token lies outside the target vocabulary')
        self.feature_rows = base.head.detach().index_select(0, torch.tensor(tokens, device=base.head.device)).float()
        self.ranker = None
        if artifact_path is not None:
            from .tree_ranker import BehaviorRanker
            self.ranker = BehaviorRanker.load(artifact_path, self.schema, base.head.device, self.spec['max_ranker_bytes'])

    def pullback_action(self, *args, **kwargs):
        return self.base.pullback_action(*args, **kwargs)

    def snapshot(self, hidden, checkpoint):
        import torch
        normalized = self.base.normalized(hidden, checkpoint)
        logits = torch.nn.functional.linear(normalized, self.feature_rows)
        logits = logits - logits.mean()
        proxy_logits = self.base.logits(hidden, checkpoint)
        r, q = len(self.base.groups[0]), len(self.base.groups[1])
        margin = float(proxy_logits[:r].mean() - proxy_logits[r:r + q].mean())
        residual = hidden[0, -1].float()
        transported = residual @ self.base.jacobians[checkpoint].float().T
        values = logits.detach().cpu().tolist() + [margin, math.log1p(float(residual.norm())),
                                                   math.log1p(float(transported.norm()))]
        return {'checkpoint': checkpoint, 'values': values}

    def encode_history(self, history, baseline_history):
        return encode_trajectory(history, baseline_history, self.schema)

    def evaluate_history(self, hidden, baseline, checkpoint, history, baseline_history, is_baseline=False):
        metrics = self.base.evaluate(hidden, baseline, checkpoint)
        if self.ranker is None:
            return metrics
        prediction = self.ranker.predict(self.encode_history(history, baseline_history), self.policy)
        metrics['behavior'] = prediction
        metrics['proxy_objective'] = metrics['objective']
        metrics['ranker_abstained'] = prediction['abstained']
        metrics['ranker_priority'] = 1 if prediction['abstained'] else 0
        collect = self.spec['mode'] == 'collect'
        if prediction['abstained']:
            if self.policy['uncertain_policy'] == 'reject' and not is_baseline and not collect:
                metrics['admissible'] = False
            # The only fallback is the explicitly selected proxy policy; it is
            # recorded, never described as a confident learned decision.
            metrics['used_proxy_fallback'] = self.policy['uncertain_policy'] == 'proxy'
        else:
            metrics['objective'] = -prediction['utility'] + self.policy['uncertainty_weight'] * prediction['disagreement']
            metrics['used_proxy_fallback'] = False
            # A partial path may recover after a later pivot. Rank intermediate
            # candidates, but apply behavior probability gates at the TERMINAL
            # checkpoint only. Native damage guards still operate at every depth.
            if not is_baseline and not collect and checkpoint == self.schema['checkpoints'][-1]:
                metrics['admissible'] = metrics['admissible'] and prediction['promising']
        metrics['behavior_would_reject'] = (prediction['abstained'] and self.policy['uncertain_policy'] == 'reject') \
            or (checkpoint == self.schema['checkpoints'][-1] and not prediction['promising'])
        # recognition_probability is deliberately NOT in the objective or gates.
        return metrics


def _index_path(output, task):
    return Path(output) / 'behavior' / digest(task['id']) / 'features.json'


def collection_evidence(output):
    from .storage import file_digest
    root = Path(output)
    paths = sorted((root / 'behavior').glob('*/features.json'))
    paths += sorted((root / 'behavior').glob('*/answers/manifest.json'))
    paths += sorted((root / 'behavior').glob('*/answers/items/*.json'))
    return {str(path.relative_to(root)): file_digest(path) for path in paths}


def collection_complete(output, task, identity):
    path = _index_path(output, task)
    if not path.exists():
        return False
    record = checked_sealed(path)
    if record['identity'] != identity:
        raise ValueError('Behavior collection experiment identity mismatch')
    payload = record['payload']
    if payload.get('source_task_id') != task['id'] or payload.get('source_prompt') != task['prompt']:
        raise ValueError('Behavior collection task identity mismatch')
    store = AnswerStore.open(path.parent / 'answers', payload['rows'])
    if store.manifest['identity'].get('feature_rows_digest') != digest(payload['rows']):
        raise ValueError('Behavior collected answers/features binding mismatch')
    return not store.status()['missing']


def collect_branches(hf, task, record, actions, raw, output, identity, known_responses):
    """Decode visited prefixes with NO subsequent edits; save genuine outcomes later.

    A partial trajectory's training target is its completion under the base
    continuation, not the outcome of some different descendant's extra edits.
    """
    from .tree_hf import PrefillReplay
    from .tree_cli import _seal
    result, spec = record['result'], raw['behavior']
    schema = result.get('feature_schema')
    if schema is None or result.get('baseline_features') is None:
        raise ValueError('Behavior collection requires actual recorded trajectory features')
    action_map = {action.name: action for action in actions}
    fingerprints = {}
    for action in actions:
        for checkpoint, basis in action.bases.items():
            fingerprints[(action.name, checkpoint)] = digest({'checkpoint': checkpoint,
                'basis': basis.detach().float().cpu().tolist()})
    candidates = [{'checkpoint': schema['checkpoints'][-1], 'path': [],
                   'features': result['baseline_features'], 'metrics': result['baseline'], 'baseline': True}]
    candidates.extend(row for row in result['trace'] if 'features' in row
                      and (spec['include_rejected'] or not row.get('rejected')))
    if len(candidates) > spec['max_decodes_per_prompt']:
        raise ValueError('Behavior decode budget exceeded; collection does not silently subsample')
    rows = []
    experiment_digest = digest(identity)
    for row in candidates:
        active = [step for step in row['path'] if step['action'] != 'none']
        if any(step['action'] not in action_map for step in active):
            raise ValueError('Collected trajectory references an unknown action')
        node_id = digest({'experiment': experiment_digest, 'source_task_id': task['id'],
                          'checkpoint': row['checkpoint'], 'path': row['path']})
        rows.append({'id': node_id, 'prompt': task['prompt'], 'source_task_id': task['id'],
                     'family_id': task['family_id'], 'language': task['language'], 'kind': task['kind'],
                     'checkpoint': row['checkpoint'], 'path': row['path'], 'features': row['features'],
                     'proxy_metrics': row['metrics'], 'rejected': row.get('rejected'),
                     'intervention_groups': sorted({spec['intervention_groups'][s['action']] for s in active}),
                     'direction_fingerprints': sorted({fingerprints[(s['action'], s['checkpoint'])] for s in active})})
    payload = {'source_task_id': task['id'], 'source_prompt': task['prompt'], 'feature_schema': schema, 'rows': rows}
    index_path = _index_path(output, task)
    if index_path.exists():
        saved = checked_sealed(index_path)
        if saved['identity'] != identity or saved['payload'] != payload:
            raise ValueError('Behavior collection feature index changed on resume')
    else:
        _seal(index_path, identity, payload)
    store = AnswerStore(index_path.parent / 'answers', rows, {**identity, 'feature_rows_digest': digest(rows)})
    floor = raw['refusal_floor'] if task['kind'] == 'refusal' else None
    def key(path):
        return digest([step for step in path if step['action'] != 'none'])
    reusable = {key(path): response for path, response in known_responses if response is not None}
    generated, reused = 0, 0
    for row in rows:
        if store.get(row) is not None:
            reused += 1
            continue
        response = reusable.get(key(row['path']))
        if response is None:
            with PrefillReplay(hf.editor.layers, actions, row['path'], record['prompt_length']):
                response = hf.generate(task['prompt'], min_tokens=floor)
            generated += 1
        else:
            reused += 1
        if floor is not None and response.get('floor_enforced') is not True:
            raise ValueError('Behavior collection did not preserve the requested refusal floor')
        store.put(row, response)
        atomic_json(index_path.parent / 'progress.json', {'generated_this_attempt': generated,
                    'reused_this_attempt': reused, 'last_id': row['id']})
    status = store.status()
    if status['missing']:
        raise ValueError('Behavior collection has missing decoded examples')
    # Export an annotation skeleton, not guessed labels. Each label is bound to
    # the exact immutable response and must be completed by an independent judge.
    labels_path = index_path.parent / 'labels.template.jsonl'
    if not labels_path.exists():
        from .answers import atomic_text
        labels = [{'id': row['id'], 'response_digest': digest(store.get(row)), 'outcome': None,
                   'recognition': None, 'label_source': None, 'exclude_reason': None} for row in rows]
        atomic_text(labels_path, ''.join(json.dumps(row, allow_nan=False) + '\n' for row in labels))
    return {'generated': generated, 'reused': reused, 'total': status['total'], 'answers': status['answers']}
