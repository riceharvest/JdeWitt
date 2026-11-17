# SPDX-License-Identifier: AGPL-3.0-or-later
"""Behavior-feature contracts and independently labelled data. No model imports.

Labels must describe completed decoded behavior, never proxy scores or keywords.
The recognition label is independent and is never used as the ranking reward.
"""
from collections import Counter
import json
import math
from pathlib import Path

from .answers import AnswerStore, atomic_json, atomic_text
from .data import canonical_prompt, digest
from .storage import file_digest

OUTCOMES = ('completion', 'refusal', 'evasion', 'broken')
SPLITS = ('train', 'calibration', 'test')


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def feature_schema(raw, lens_sha256, runtime_fingerprint=None):
    tokens = raw['behavior']['feature_tokens']
    width = len(tokens) + 3
    return {'format': 'jspace-trajectory-features-v1', 'model_identity': raw['model_identity'],
            'lens_sha256': lens_sha256, 'checkpoints': raw['checkpoints'], 'feature_tokens': tokens,
            'source_dtype': raw['dtype'], 'refusal_tokens': raw['refusal_tokens'],
            'reference_tokens': raw['reference_tokens'], 'runtime_fingerprint': runtime_fingerprint,
            'generation_contract': {key: raw[key] for key in ('ctx_size', 'max_tokens', 'refusal_floor')},
            'snapshot': 'centered-selected-logits,refusal-margin,log1p-residual-norm,log1p-transport-norm',
            'encoding': 'per-checkpoint:snapshot,delta-from-baseline,delta-from-previous,observed-mask',
            'snapshot_width': width, 'dimension': len(raw['checkpoints']) * (3 * width + 1)}


def encode_trajectory(history, baseline_history, schema):
    """Fixed width, no future observations, no action-name/strength label leakage."""
    width = schema['snapshot_width']
    if len(history) != len(baseline_history) or not history:
        raise ValueError('Behavior history/baseline lengths differ')
    output, previous = [], None
    for index, checkpoint in enumerate(schema['checkpoints']):
        if index >= len(history):
            output.extend([0.] * (3 * width + 1))
            continue
        item, baseline = history[index], baseline_history[index]
        if item['checkpoint'] != checkpoint or baseline['checkpoint'] != checkpoint:
            raise ValueError('Behavior trajectory checkpoint order differs from its schema')
        row, base = item['values'], baseline['values']
        if len(row) != width or len(base) != width or not all(finite_number(x) for x in row + base):
            raise ValueError('Invalid/nonfinite behavior snapshot')
        delta = [0.] * width if previous is None else [a - b for a, b in zip(row, previous)]
        output.extend(row + [a - b for a, b in zip(row, base)] + delta + [1.])
        previous = row
    return output


def validate_policy(policy):
    probabilities = ('min_completion_probability', 'max_refusal_probability',
                     'max_evasion_probability', 'max_broken_probability')
    extra = {'max_disagreement', 'max_feature_z', 'uncertainty_weight', 'uncertain_policy'}
    if not isinstance(policy, dict) or set(policy) != set(probabilities) | extra:
        raise ValueError('Behavior ranker policy requires explicit probability/uncertainty/abstention settings')
    if any(not finite_number(policy[k]) or not 0 <= policy[k] <= 1 for k in probabilities):
        raise ValueError('Behavior policy probabilities must be finite and within [0,1]')
    if any(not finite_number(policy[k]) or policy[k] < 0 for k in extra - {'uncertain_policy'}):
        raise ValueError('Behavior uncertainty bounds/weights must be finite and nonnegative')
    if policy['uncertain_policy'] not in {'reject', 'proxy'}:
        raise ValueError('uncertain_policy must explicitly choose reject or proxy')


def validate_behavior(raw, base, paths, required_expansions):
    spec = raw.get('behavior')
    fields = {'mode', 'feature_tokens', 'intervention_groups', 'max_feature_values', 'max_ranker_bytes',
              'artifact', 'policy', 'max_decodes_per_prompt', 'include_rejected'}
    if not isinstance(spec, dict) or set(spec) != fields or spec['mode'] not in {'collect', 'rank'}:
        raise ValueError('Invalid explicit behavior feature/ranker configuration')
    tokens = spec['feature_tokens']
    if (not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens)
            or len(set(tokens)) != len(tokens)):
        raise ValueError('behavior.feature_tokens needs explicit distinct target-token IDs')
    groups = spec['intervention_groups']
    if (not isinstance(groups, dict) or set(groups) != set(raw['actions'])
            or any(not isinstance(v, str) or not v.strip() for v in groups.values())):
        raise ValueError('Declare an authored intervention family for every named action')
    for key in ('max_feature_values', 'max_ranker_bytes'):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(f'behavior.{key} requires an explicit positive integer')
    dimension = len(raw['checkpoints']) * (3 * (len(tokens) + 3) + 1)
    if (required_expansions + 1) * dimension > spec['max_feature_values']:
        raise ValueError('Behavior feature budget cannot retain the complete configured search')
    if type(spec['include_rejected']) is not bool:
        raise ValueError('include_rejected must be a genuine boolean')
    if spec['mode'] == 'collect':
        cap = spec['max_decodes_per_prompt']
        if type(cap) is not int or cap < required_expansions + 1:
            raise ValueError('Behavior collection decode cap must cover baseline and every finite visited node')
    elif spec['max_decodes_per_prompt'] is not None or spec['include_rejected']:
        raise ValueError('Rank-only mode cannot request branch collection')
    if spec['artifact'] is None:
        if spec['mode'] != 'collect' or spec['policy'] is not None:
            raise ValueError('Rank mode requires a trained artifact; collection without one cannot set ranker policy')
    else:
        if not isinstance(spec['artifact'], str) or not spec['artifact'].strip():
            raise ValueError('Behavior ranker requires an artifact path')
        validate_policy(spec['policy'])
        paths['behavior_artifact'] = (base / spec['artifact']).resolve()
        if not paths['behavior_artifact'].is_file():
            raise ValueError('Missing behavior ranker artifact')
        artifact = json.loads(paths['behavior_artifact'].read_text())
        if not isinstance(artifact, dict) or artifact.get('format') != 'jspace-behavior-ranker-v1':
            raise ValueError('Unsupported behavior ranker artifact format')


def promising(prediction, policy):
    return (not prediction['abstained']
            and prediction['completion_probability'] >= policy['min_completion_probability']
            and all(prediction[k + '_probability'] <= policy['max_' + k + '_probability']
                    for k in ('refusal', 'evasion', 'broken')))


def checked_sealed(path):
    record = json.loads(Path(path).read_text())
    if (not isinstance(record, dict) or set(record) != {'identity', 'payload', 'record_digest'}
            or record['record_digest'] != digest({k: v for k, v in record.items() if k != 'record_digest'})):
        raise ValueError('Behavior evidence record integrity mismatch')
    return record


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_rows(rows, schema):
    ids = set()
    for row in rows:
        required = {'id', 'prompt', 'family_id', 'language', 'kind', 'checkpoint', 'features', 'outcome',
                    'recognition', 'intervention_groups', 'direction_fingerprints', 'response_digest', 'label_source'}
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError('Behavior training row fields differ from the required schema')
        if any(not isinstance(row[k], str) or not row[k].strip()
               for k in ('id', 'prompt', 'family_id', 'language', 'kind', 'response_digest')):
            raise ValueError('Behavior examples require actual prompt/family/language identities')
        if row['id'] in ids:
            raise ValueError('Duplicate behavior example ID')
        ids.add(row['id'])
        if (type(row['checkpoint']) is not int or row['checkpoint'] not in schema['checkpoints']
                or not isinstance(row['features'], list) or len(row['features']) != schema['dimension']
                or not all(finite_number(x) for x in row['features'])):
            raise ValueError('Behavior example feature/checkpoint shape mismatch')
        slot_width = 3 * schema['snapshot_width'] + 1
        for index, checkpoint in enumerate(schema['checkpoints']):
            slot = row['features'][index * slot_width:(index + 1) * slot_width]
            observed = checkpoint <= row['checkpoint']
            if slot[-1] != float(observed) or (not observed and any(v != 0 for v in slot)):
                raise ValueError('Behavior example exposes future/unobserved checkpoint features')
        if row['outcome'] not in OUTCOMES or type(row['recognition']) is not bool:
            raise ValueError('Explicit behavioral outcome and independent boolean recognition label required')
        for key in ('intervention_groups', 'direction_fingerprints'):
            if (not isinstance(row[key], list) or len(set(row[key])) != len(row[key])
                    or any(not isinstance(v, str) or not v.strip() for v in row[key])):
                raise ValueError('Invalid intervention provenance in behavior example')
        label_source = row['label_source']
        if (not isinstance(label_source, dict) or set(label_source) != {'kind', 'identity'}
                or label_source['kind'] not in {'human', 'full-answer-judge'}
                or not isinstance(label_source['identity'], (dict, str)) or not label_source['identity']):
            raise ValueError('Independent label-source provenance required')
    return rows


def validate_isolation(splits):
    """Disjoint prompts, authored topic families AND intervention families/bases."""
    for index, left in enumerate(SPLITS):
        if not splits.get(left):
            raise ValueError(f'Empty behavior {left} split')
        for right in SPLITS[:index]:
            for field in ('family_id', 'prompt', 'intervention_groups', 'direction_fingerprints'):
                def values(rows):
                    if field == 'prompt':
                        return {canonical_prompt(row[field]) for row in rows}
                    if field in {'intervention_groups', 'direction_fingerprints'}:
                        return {value for row in rows for value in row[field]}
                    return {row[field] for row in rows}
                if values(splits[left]) & values(splits[right]):
                    raise ValueError(f'Behavior {field} overlap between {left} and {right}')


def prepare_ranker_data(spec_path):
    """Join collected states to explicit independent full-answer annotations."""
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    if (not isinstance(spec, dict) or set(spec) != {'sources', 'output'} or not isinstance(spec['sources'], list)
            or not isinstance(spec['output'], str) or not spec['output'].strip()):
        raise ValueError('Behavior data spec requires sources and output')
    output = (spec_path.parent / spec['output']).resolve()
    if output.exists():
        raise FileExistsError('Use a new behavior dataset output; no existing data is rewritten')
    splits, schema, evidence = {key: [] for key in SPLITS}, None, []
    input_paths = [spec_path]
    for source in spec['sources']:
        if (not isinstance(source, dict) or set(source) != {'job', 'labels', 'split'} or source['split'] not in SPLITS
                or any(not isinstance(source[k], str) or not source[k].strip() for k in ('job', 'labels'))):
            raise ValueError('Each behavior source needs explicit job/labels/split')
        job = (spec_path.parent / source['job']).resolve()
        labels_path = (spec_path.parent / source['labels']).resolve()
        input_paths.extend([job, labels_path])
        labels = {}
        for label in read_jsonl(labels_path):
            if not isinstance(label, dict) or label.get('id') in labels or not isinstance(label.get('id'), str):
                raise ValueError('Invalid/duplicate behavior label IDs')
            labels[label['id']] = label
        consumed = set()
        indexes = sorted((job / 'behavior').glob('*/features.json'))
        if not indexes:
            raise ValueError('No collected behavior features in the requested job')
        for index_path in indexes:
            sealed = checked_sealed(index_path)
            payload = sealed['payload']
            local_schema = payload['feature_schema']
            schema = local_schema if schema is None else schema
            if schema != local_schema:
                raise ValueError('Behavior sources use different model/lens/trajectory feature schemas')
            rows = payload['rows']
            store = AnswerStore.open(index_path.parent / 'answers', rows)
            if store.manifest['identity'].get('feature_rows_digest') != digest(rows):
                raise ValueError('Behavior answers are not bound to their collected features')
            for row in rows:
                response = store.get(row)
                if response is None:
                    raise ValueError('Collected behavior example has no saved full response')
                label = labels.get(row['id'])
                if label is None:
                    raise ValueError('Every collected example needs an explicit annotation or explicit exclusion')
                consumed.add(row['id'])
                if label.get('response_digest') != digest(response):
                    raise ValueError('Behavior label is not bound to the exact decoded answer')
                if label.get('outcome') is None:
                    if not isinstance(label.get('exclude_reason'), str) or not label['exclude_reason'].strip():
                        raise ValueError('Unlabelled behavior data requires an explicit exclusion reason')
                    continue
                if label.get('outcome') == 'completion' and response.get('finish_reason') != 'stop':
                    raise ValueError('A truncated/nonterminal answer cannot be labelled genuine completion')
                # Recognition must be independently assessed, not inferred from
                # the refusal outcome or copied from the prompt's harm category.
                item = {key: row[key] for key in ('id', 'prompt', 'family_id', 'language', 'kind', 'checkpoint',
                        'features', 'intervention_groups', 'direction_fingerprints')}
                item.update({key: label.get(key) for key in ('outcome', 'recognition', 'response_digest', 'label_source')})
                splits[source['split']].append(item)
            evidence.append({'index': str(index_path), 'sha256': file_digest(index_path),
                             'answer_generation_digest': store.manifest['generation_digest']})
        if consumed != set(labels):
            raise ValueError('Behavior labels contain IDs absent from the requested collection')
        evidence.append({'labels': str(labels_path), 'sha256': file_digest(labels_path), 'split': source['split']})
    if schema is None:
        raise ValueError('No behavior schema/records were collected')
    for rows in splits.values():
        validate_rows(rows, schema)
    validate_isolation(splits)
    if any(output == p or output in p.parents for p in input_paths):
        raise ValueError('Behavior dataset output overlaps its input evidence')
    output.mkdir(parents=True)
    files, counts = {}, {}
    for name, rows in splits.items():
        path = output / (name + '.jsonl')
        atomic_text(path, ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))
        files[name] = {'file': path.name, 'sha256': file_digest(path)}
        counts[name] = dict(Counter(row['outcome'] for row in rows))
    manifest = {'format': 'jspace-behavior-dataset-v1', 'feature_schema': schema, 'files': files,
                'counts': counts, 'evidence': evidence, 'isolation': 'prompt+authored-family+intervention-family+exact-basis'}
    manifest['record_digest'] = digest(manifest)
    atomic_json(output / 'manifest.json', manifest)
    return {'dataset': str(output / 'manifest.json'), 'counts': counts,
            'status': 'independently-labelled-data; no model trained'}


def load_dataset(path, names=SPLITS):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if (manifest.get('format') != 'jspace-behavior-dataset-v1'
            or manifest.get('record_digest') != digest({k: v for k, v in manifest.items() if k != 'record_digest'})):
        raise ValueError('Behavior dataset manifest integrity mismatch')
    splits = {}
    for name in names:
        entry = manifest['files'][name]
        target = (path.parent / entry['file']).resolve()
        if target.parent != path.parent or file_digest(target) != entry['sha256']:
            raise ValueError('Behavior dataset split path/hash mismatch')
        splits[name] = validate_rows(read_jsonl(target), manifest['feature_schema'])
    return manifest, splits
