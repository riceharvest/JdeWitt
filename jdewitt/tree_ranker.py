# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pairwise trajectory ranker with independent behavior and recognition heads.

Training, calibration and prediction require B70. Nothing is trained on import.
This is not a refusal judge: labels come from independent completed-answer checks.
"""
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .answers import atomic_json
from .data import canonical_prompt, digest
from .storage import file_digest
from .tree_behavior_data import OUTCOMES, finite_number, load_dataset, promising, validate_policy


class TrajectoryNetwork(nn.Module):
    def __init__(self, dimension, hidden_size, device):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, hidden_size, device=device, dtype=torch.float32),
                                    nn.Tanh(), nn.Linear(hidden_size, 6, device=device, dtype=torch.float32))

    def forward(self, features):
        return self.layers(features)


def _require_xpu(device):
    if device.type != 'xpu':
        raise ValueError('Behavior ranker model operations require B70; CPU/CUDA fallback is forbidden')


class BehaviorRanker:
    def __init__(self, artifact, schema, device, max_bytes):
        _require_xpu(device)
        if (not isinstance(artifact, dict) or artifact.get('format') != 'jspace-behavior-ranker-v1'
                or artifact.get('feature_schema') != schema
                or artifact.get('record_digest') != digest({k: v for k, v in artifact.items() if k != 'record_digest'})):
            raise ValueError('Behavior ranker artifact integrity/feature-schema mismatch')
        hidden = artifact.get('hidden_size')
        members = artifact.get('members')
        dimension = schema['dimension']
        if type(hidden) is not int or hidden < 1 or not isinstance(members, list) or len(members) < 2:
            raise ValueError('Behavior ranker needs a valid multi-member ensemble')
        # Network weights, normalization and ensemble prediction scratch only;
        # caller budgets feature tensors and target-model workspaces separately.
        bound = 4 * (len(members) * (dimension * hidden + hidden + hidden * 6 + 6) + 2 * dimension)
        if type(max_bytes) is not int or max_bytes < bound:
            raise ValueError(f'Behavior ranker resident parameter budget too small: {bound}')
        self.artifact, self.device = artifact, device
        self.mean = torch.tensor(artifact['normalization']['mean'], dtype=torch.float32, device=device)
        self.scale = torch.tensor(artifact['normalization']['scale'], dtype=torch.float32, device=device)
        if (self.mean.shape != (dimension,) or self.scale.shape != (dimension,)
                or not bool(torch.isfinite(self.mean).all()) or not bool(torch.isfinite(self.scale).all())
                or not bool((self.scale > 0).all())):
            raise ValueError('Invalid behavior ranker normalization')
        self.members = []
        for state in members:
            model = TrajectoryNetwork(dimension, hidden, device)
            expected = model.state_dict()
            if set(state) != set(expected):
                raise ValueError('Behavior ranker member parameter names differ')
            tensors = {name: torch.tensor(value, dtype=torch.float32, device=device) for name, value in state.items()}
            if any(tensors[k].shape != expected[k].shape or not bool(torch.isfinite(tensors[k]).all()) for k in tensors):
                raise ValueError('Behavior ranker member shape/finite mismatch')
            model.load_state_dict(tensors, strict=True)
            self.members.append(model.eval())
        self.temperature = artifact['temperature']
        self.recognition_temperature = artifact['recognition_temperature']
        if any(not finite_number(t) or t <= 0 for t in (self.temperature, self.recognition_temperature)):
            raise ValueError('Behavior ranker temperatures must be finite and positive')

    @classmethod
    def load(cls, path, schema, device, max_bytes):
        return cls(json.loads(Path(path).read_text()), schema, device, max_bytes)

    @torch.inference_mode()
    def predict_many(self, features, policy):
        validate_policy(policy)
        x = torch.tensor(features, device=self.device, dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not bool(torch.isfinite(x).all()):
            raise ValueError('Invalid behavior ranker feature batch')
        z = (x - self.mean) / self.scale
        outputs = torch.stack([model(z) for model in self.members])
        probabilities = F.softmax(outputs[..., :4] / self.temperature, dim=-1)
        mean = probabilities.mean(0)
        # Disagreement is not a formal uncertainty bound. Keep it separate from
        # calibrated probabilities and abstain on excessive feature extrapolation.
        disagreement = probabilities.std(0, unbiased=False).amax(-1)
        feature_z = z.abs().amax(-1)
        utilities = outputs[..., 5].mean(0)
        recognition = torch.sigmoid(outputs[..., 4] / self.recognition_temperature).mean(0)
        result = []
        for index in range(len(features)):
            row = {name + '_probability': float(mean[index, j]) for j, name in enumerate(OUTCOMES)}
            row.update(utility=float(utilities[index]), recognition_probability=float(recognition[index]),
                       disagreement=float(disagreement[index]), feature_z=float(feature_z[index]))
            if not all(finite_number(v) for v in row.values()):
                raise ValueError('Nonfinite behavior ranker predictions')
            row['abstained'] = (row['disagreement'] > policy['max_disagreement']
                                or row['feature_z'] > policy['max_feature_z'])
            row['promising'] = promising(row, policy)
            result.append(row)
        return result

    def predict(self, features, policy):
        return self.predict_many([features], policy)[0]


def _seen(rows):
    return {'families': sorted({r['family_id'] for r in rows}),
            'prompts': sorted({digest(canonical_prompt(r['prompt'])) for r in rows}),
            'intervention_groups': sorted({g for r in rows for g in r['intervention_groups']}),
            'direction_fingerprints': sorted({g for r in rows for g in r['direction_fingerprints']})}


def _assert_unseen(rows, seen):
    current = _seen(rows)
    for key in current:
        if set(current[key]) & set(seen[key]):
            raise ValueError(f'Behavior held-out data overlaps training/calibration {key}')


def _pairs(rows, utilities, limit, rng):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(canonical_prompt(row['prompt']), row['checkpoint'])].append(index)
    pairs = []
    for indices in groups.values():
        # Reservoir sampling bounds pair storage without preferring the first
        # decoded branch. All comparisons remain within the SAME prompt/depth.
        selected, count = [], 0
        for a, b in itertools.combinations(indices, 2):
            gap = utilities[rows[a]['outcome']] - utilities[rows[b]['outcome']]
            if gap == 0:
                continue
            item = (a, b, 1. if gap > 0 else -1.)
            count += 1
            if len(selected) < limit:
                selected.append(item)
            else:
                replace = int(rng.integers(count))
                if replace < limit:
                    selected[replace] = item
        pairs.extend(selected)
    if not pairs:
        raise ValueError('No same-prompt/depth outcome-preference pairs; cannot train a pairwise ranker')
    return pairs


def _training_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    fields = {'dataset', 'output', 'hidden_size', 'seeds', 'epochs', 'batch_size', 'learning_rate',
              'weight_decay', 'max_examples', 'max_pairs_per_prompt', 'pairs_per_batch',
              'pair_weight', 'utility_weight', 'outcome_weight', 'recognition_weight',
              'outcome_utilities', 'calibration_steps', 'calibration_learning_rate'}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError('Ranker training requires all explicit dataset/model/loss/optimizer/calibration settings')
    if any(not isinstance(raw[k], str) or not raw[k].strip() for k in ('dataset', 'output')):
        raise ValueError('Ranker training requires actual dataset/output paths')
    for key in ('hidden_size', 'epochs', 'batch_size', 'max_examples', 'max_pairs_per_prompt',
                'pairs_per_batch', 'calibration_steps'):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError(f'Ranker {key} must be a positive integer')
    seeds = raw['seeds']
    if (not isinstance(seeds, list) or len(seeds) < 2 or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Choose at least two distinct nonnegative ranker ensemble seeds')
    for key in ('learning_rate', 'calibration_learning_rate', 'pair_weight', 'utility_weight',
                'outcome_weight', 'recognition_weight'):
        if not finite_number(raw[key]) or raw[key] <= 0:
            raise ValueError(f'Ranker {key} must be finite and positive')
    if not finite_number(raw['weight_decay']) or raw['weight_decay'] < 0:
        raise ValueError('Ranker weight_decay must be finite and nonnegative')
    utilities = raw['outcome_utilities']
    if (not isinstance(utilities, dict) or set(utilities) != set(OUTCOMES)
            or not all(finite_number(v) for v in utilities.values())
            or utilities['completion'] <= max(utilities[k] for k in OUTCOMES if k != 'completion')):
        raise ValueError('Explicit outcome utilities must rank genuine completion above every failure outcome')
    return raw, (path.parent / raw['dataset']).resolve(), (path.parent / raw['output']).resolve()


def _temperature(logits, targets, steps, lr, binary=False):
    # Fit on CALIBRATION only. Test outcomes never set temperatures or thresholds.
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        t = log_temperature.clamp(-5, 5).exp()
        if binary:
            probability = torch.sigmoid(logits / t).mean(0).clamp(1e-7, 1 - 1e-7)
            loss = F.binary_cross_entropy(probability, targets)
        else:
            probabilities = F.softmax(logits / t, dim=-1).mean(0).clamp_min(1e-12)
            loss = F.nll_loss(probabilities.log(), targets)
        loss.backward()
        optimizer.step()
    return float(log_temperature.detach().clamp(-5, 5).exp())


def train_ranker(spec_path):
    raw, dataset_path, output = _training_config(spec_path)
    if output.exists() or output == dataset_path or output in dataset_path.parents:
        raise FileExistsError('Ranker output must be a new path outside its input dataset')
    manifest, splits = load_dataset(dataset_path, names=('train', 'calibration'))
    train, calibration = splits['train'], splits['calibration']
    if len(train) + len(calibration) > raw['max_examples']:
        raise ValueError('Ranker example budget exceeded; no automatic row truncation')
    _assert_unseen(calibration, _seen(train))
    expected_checkpoints = set(manifest['feature_schema']['checkpoints'][1:])
    for name, rows in splits.items():
        if ({r['outcome'] for r in rows} != set(OUTCOMES) or {r['recognition'] for r in rows} != {True, False}
                or {r['checkpoint'] for r in rows} != expected_checkpoints):
            raise ValueError(f'{name} must cover all behavior classes, recognition labels and evaluated checkpoint depths')
    # Metadata/data parsing above does not initialize a device. Only this explicit
    # training command takes B70 ownership. No target model is loaded here.
    from .hf import require_b70
    from .pipeline import configure_compiler_paths
    configure_compiler_paths()
    device = require_b70()
    x = torch.tensor([r['features'] for r in train], device=device, dtype=torch.float32)
    xc = torch.tensor([r['features'] for r in calibration], device=device, dtype=torch.float32)
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    x, xc = (x - mean) / scale, (xc - mean) / scale
    y = torch.tensor([OUTCOMES.index(r['outcome']) for r in train], device=device)
    yc = torch.tensor([OUTCOMES.index(r['outcome']) for r in calibration], device=device)
    recognition = torch.tensor([float(r['recognition']) for r in train], device=device)
    rc = torch.tensor([float(r['recognition']) for r in calibration], device=device)
    utility = torch.tensor([raw['outcome_utilities'][r['outcome']] for r in train], device=device)
    family_counts = Counter(r['family_id'] for r in train)
    weights = torch.tensor([1. / family_counts[r['family_id']] for r in train], device=device)
    pairs = _pairs(train, raw['outcome_utilities'], raw['max_pairs_per_prompt'], np.random.default_rng(raw['seeds'][0]))
    pa = torch.tensor([p[0] for p in pairs], device=device)
    pb = torch.tensor([p[1] for p in pairs], device=device)
    ps = torch.tensor([p[2] for p in pairs], device=device)
    pair_families = Counter(train[p[0]]['family_id'] for p in pairs)
    pw = torch.tensor([1. / pair_families[train[p[0]]['family_id']] for p in pairs], device=device)
    models, logs = [], []
    for seed in raw['seeds']:
        torch.manual_seed(seed)
        model = TrajectoryNetwork(x.shape[1], raw['hidden_size'], device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=raw['learning_rate'], weight_decay=raw['weight_decay'])
        rng = np.random.default_rng(seed)
        last_loss = None
        for _ in range(raw['epochs']):
            order = rng.permutation(len(train))
            for start in range(0, len(order), raw['batch_size']):
                idx = torch.tensor(order[start:start + raw['batch_size']], device=device)
                predicted = model(x[idx])
                row_loss = (raw['outcome_weight'] * F.cross_entropy(predicted[:, :4], y[idx], reduction='none')
                            + raw['recognition_weight'] * F.binary_cross_entropy_with_logits(predicted[:, 4], recognition[idx], reduction='none')
                            + raw['utility_weight'] * F.smooth_l1_loss(predicted[:, 5], utility[idx], reduction='none'))
                local_weights = weights[idx]
                loss = (row_loss * local_weights).sum() / local_weights.sum()
                pi = torch.tensor(rng.integers(len(pairs), size=raw['pairs_per_batch']), device=device)
                difference = model(x[pa[pi]])[:, 5] - model(x[pb[pi]])[:, 5]
                pair_loss = F.softplus(-ps[pi] * difference)
                loss = loss + raw['pair_weight'] * (pair_loss * pw[pi]).sum() / pw[pi].sum()
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite behavior ranker training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach())
        models.append(model.eval())
        logs.append({'seed': seed, 'last_training_loss': last_loss})
    with torch.no_grad():
        logits = torch.stack([model(xc) for model in models])
    temperature = _temperature(logits[..., :4].detach(), yc, raw['calibration_steps'], raw['calibration_learning_rate'])
    recognition_temperature = _temperature(logits[..., 4].detach(), rc, raw['calibration_steps'],
                                           raw['calibration_learning_rate'], binary=True)
    artifact = {'format': 'jspace-behavior-ranker-v1', 'status': 'trained-calibrated-development-only',
                'feature_schema': manifest['feature_schema'], 'hidden_size': raw['hidden_size'],
                'normalization': {'mean': mean.detach().cpu().tolist(), 'scale': scale.detach().cpu().tolist()},
                'members': [{k: v.detach().cpu().tolist() for k, v in model.state_dict().items()} for model in models],
                'temperature': temperature, 'recognition_temperature': recognition_temperature,
                'training': {'spec': raw, 'dataset_sha256': file_digest(dataset_path),
                             'split_hashes': {k: manifest['files'][k]['sha256'] for k in ('train', 'calibration')},
                             'seen': _seen(train + calibration), 'logs': logs, 'pairs': len(pairs)},
                'not_checked': ['unseen-family/direction false confidence', 'tree optimization against this ranker',
                                'complete-answer generalization', 'capability retention']}
    artifact['record_digest'] = digest(artifact)
    atomic_json(output, artifact)
    return {'artifact': str(output), 'status': artifact['status'], 'members': len(models), 'pairs': len(pairs),
            'test_split_consumed': False}


def audit_ranker(spec_path):
    """Independent false-confidence report, not automatic acceptance/retraining."""
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    required = {'dataset', 'artifact', 'output', 'policy', 'batch_size', 'max_ranker_bytes',
                'min_promising', 'max_false_confidence_rate'}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError('Ranker audit needs explicit dataset/artifact/policy/budgets/acceptance limits')
    if any(not isinstance(spec[k], str) or not spec[k].strip() for k in ('dataset', 'artifact', 'output')):
        raise ValueError('Ranker audit requires actual dataset/artifact/output paths')
    validate_policy(spec['policy'])
    for key in ('batch_size', 'max_ranker_bytes', 'min_promising'):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(f'Audit {key} must be a positive integer')
    if not finite_number(spec['max_false_confidence_rate']) or not 0 <= spec['max_false_confidence_rate'] <= 1:
        raise ValueError('Audit false-confidence bound must be within [0,1]')
    dataset, artifact_path, output = [(spec_path.parent / spec[k]).resolve() for k in ('dataset', 'artifact', 'output')]
    if output.exists() or output in {dataset, artifact_path, spec_path}:
        raise FileExistsError('Ranker audit needs a new output path')
    manifest, splits = load_dataset(dataset, names=('test',))
    rows = splits['test']
    if not rows:
        raise ValueError('Empty ranker audit set')
    artifact = json.loads(artifact_path.read_text())
    _assert_unseen(rows, artifact['training']['seen'])
    from .hf import require_b70
    from .pipeline import configure_compiler_paths
    configure_compiler_paths()
    ranker = BehaviorRanker(artifact, manifest['feature_schema'], require_b70(), spec['max_ranker_bytes'])
    predictions = []
    for start in range(0, len(rows), spec['batch_size']):
        predictions.extend(ranker.predict_many([r['features'] for r in rows[start:start + spec['batch_size']]], spec['policy']))
    selected = [i for i, pred in enumerate(predictions) if pred['promising']]
    false = [i for i in selected if rows[i]['outcome'] != 'completion']
    rate = len(false) / len(selected) if selected else None
    by_family = {}
    for family in sorted({r['family_id'] for r in rows}):
        indices = [i for i, row in enumerate(rows) if row['family_id'] == family and predictions[i]['promising']]
        failures = sum(rows[i]['outcome'] != 'completion' for i in indices)
        by_family[family] = {'promising': len(indices), 'false_confident': failures,
                             'rate': failures / len(indices) if indices else None}
    pair_correct, pair_total = 0., 0
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(canonical_prompt(row['prompt']), row['checkpoint'])].append(i)
    utilities = artifact['training']['spec']['outcome_utilities']
    for indices in groups.values():
        for a, b in itertools.combinations(indices, 2):
            gap = utilities[rows[a]['outcome']] - utilities[rows[b]['outcome']]
            if gap == 0:
                continue
            difference = predictions[a]['utility'] - predictions[b]['utility']
            pair_correct += .5 if difference == 0 else float(gap * difference > 0)
            pair_total += 1
    result = {'status': 'passed-held-out-screen' if len(selected) >= spec['min_promising'] and rate is not None
              and rate <= spec['max_false_confidence_rate'] else 'failed-held-out-screen',
              'artifact_sha256': file_digest(artifact_path), 'dataset_sha256': file_digest(dataset),
              'policy': spec['policy'], 'total': len(rows), 'promising_count': len(selected),
              'false_confidence_count': len(false), 'false_confidence_rate': rate,
              'false_confident_outcomes': dict(Counter(rows[i]['outcome'] for i in false)),
              'by_family': by_family, 'preference_pairs': pair_total,
              'pairwise_accuracy': pair_correct / pair_total if pair_total else None,
              'recognition_accuracy': sum((pred['recognition_probability'] >= .5) == row['recognition']
                                          for pred, row in zip(predictions, rows)) / len(rows),
              'examples': [{'id': row['id'], 'outcome': row['outcome'], 'prediction': pred}
                           for row, pred in zip(rows, predictions)],
              'warning': 'Do not use these test errors as training feedback; create fresh development collection instead.'}
    result['record_digest'] = digest(result)
    atomic_json(output, result)
    return result
