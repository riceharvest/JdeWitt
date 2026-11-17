# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tree configuration and provenance checks without importing a model runtime."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from .architectures import MAX_STRENGTH
from .data import load_suite
from .storage import file_digest

SEGMENT_PATHS = {'llama': 'model', 'qwen3': 'model', 'qwen3_moe': 'model',
                 'qwen3_5_text': 'model', 'qwen3_5_moe_text': 'model',
                 'qwen3_5': 'model.language_model', 'qwen3_5_moe': 'model.language_model'}


@dataclass(frozen=True)
class TreeSettings:
    checkpoints: tuple
    beam_width: int
    finalists: int
    max_expansions: int
    max_state_bytes: int
    strengths: tuple

    def validate(self, n_layers, n_actions):
        cps = self.checkpoints
        if (not isinstance(cps, (list, tuple)) or len(cps) < 3
                or any(type(c) is not int or not 0 <= c < n_layers for c in cps)
                or list(cps) != sorted(set(cps)) or cps[-1] != n_layers - 1):
            raise ValueError('Tree checkpoints must be increasing block outputs, at least three, ending at the last layer')
        for key in ('beam_width', 'finalists', 'max_expansions', 'max_state_bytes'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'Tree {key} needs an explicit positive integer')
        if self.finalists > self.beam_width:
            raise ValueError('Tree finalists cannot exceed beam_width')
        if (not isinstance(self.strengths, (list, tuple)) or not self.strengths
                or any(type(s) not in (int, float) or not math.isfinite(s) or not 0 < s <= MAX_STRENGTH
                       for s in self.strengths) or len(set(self.strengths)) != len(self.strengths)):
            raise ValueError('Tree strengths need distinct finite positive values within the operator bound')
        if n_actions < 1:
            raise ValueError('Tree needs at least one named intervention direction')
        choices = 1 + n_actions * len(self.strengths)
        width, required = 1, 0
        for _ in cps[1:]:
            required += width * choices
            width = min(self.beam_width, width * choices)
        if required > self.max_expansions:
            raise ValueError(f'Tree expansion budget cannot cover the complete configured search: {required}')
        return required


def validate_model_identity(identity):
    if (not isinstance(identity, dict) or set(identity) != {'model', 'revision', 'architecture', 'layers', 'hidden_size'}
            or any(not isinstance(identity[k], str) or not identity[k].strip()
                   for k in ('model', 'revision', 'architecture'))
            or any(type(identity[k]) is not int or identity[k] < 1 for k in ('layers', 'hidden_size'))):
        raise ValueError('Explicit model_identity requires model/revision/architecture/layers/hidden_size')
    if identity['architecture'] not in SEGMENT_PATHS:
        raise ValueError('No segment adapter for the declared architecture')


def _tokens(ids):
    return (isinstance(ids, list) and bool(ids) and all(type(i) is int and i >= 0 for i in ids)
            and len(set(ids)) == len(ids))


def read_tree_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    required = {'model_identity', 'dtype', 'lens_manifest', 'suite', 'suite_role', 'output', 'ctx_size',
                'max_tokens', 'refusal_floor', 'checkpoints', 'beam_width', 'finalists', 'max_expansions',
                'max_state_bytes', 'max_lens_bytes', 'strengths', 'refusal_tokens', 'reference_tokens',
                'protected_tokens', 'actions', 'max_protected_kl', 'max_norm_change', 'drift_weight'}
    if not isinstance(raw, dict) or not required <= set(raw) or set(raw) - required - {'direction_bank', 'behavior'}:
        raise ValueError(f'Tree configuration requires exactly {sorted(required)}')
    validate_model_identity(raw['model_identity'])
    if raw['dtype'] not in {'bfloat16', 'float16'} or raw['suite_role'] != 'development':
        raise ValueError('Tree requires original-precision dtype and an explicitly development-only suite')
    for key in ('ctx_size', 'max_tokens', 'max_lens_bytes'):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError(f'Tree {key} requires an explicit positive integer')
    if raw['max_tokens'] >= raw['ctx_size']:
        raise ValueError('Tree context must exceed the output reserve')
    floor = raw['refusal_floor']
    if floor is not None and (type(floor) is not int or not 0 < floor <= raw['max_tokens']):
        raise ValueError('Tree refusal_floor must be null or a positive output-budget integer')
    groups = [raw[k] for k in ('refusal_tokens', 'reference_tokens', 'protected_tokens')]
    if any(not _tokens(g) for g in groups) or len(groups[2]) < 2:
        raise ValueError('Explicit token groups required; protected_tokens needs at least two IDs')
    if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i)):
        raise ValueError('Readout token groups must be disjoint')
    actions = raw['actions']
    def bank_reference(value):
        return (isinstance(value, dict) and set(value) == {'bank'}
                and isinstance(value['bank'], str) and bool(value['bank'].strip()))
    if (not isinstance(actions, dict) or not actions
            or any(not isinstance(k, str) or not k.strip() or k == 'none'
                   or not (_tokens(v) or bank_reference(v)) for k, v in actions.items())):
        raise ValueError('Tree actions need nonempty unique names and explicit token-ID groups')
    needs_bank = any(bank_reference(value) for value in actions.values())
    if needs_bank != ('direction_bank' in raw):
        raise ValueError('direction_bank must be supplied exactly when actions reference learned bases')
    for key in ('max_protected_kl', 'max_norm_change', 'drift_weight'):
        if type(raw[key]) not in (float, int) or not math.isfinite(raw[key]) or raw[key] < 0:
            raise ValueError(f'Tree {key} requires an explicit finite nonnegative value')
    settings = TreeSettings(**{k: tuple(raw[k]) if isinstance(raw[k], list) else raw[k]
                               for k in TreeSettings.__dataclass_fields__})
    required_expansions = settings.validate(raw['model_identity']['layers'], len(actions))
    paths = {}
    for key in ('lens_manifest', 'suite', 'output'):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f'Tree {key} requires a path')
        paths[key] = (path.parent / raw[key]).resolve()
    if needs_bank:
        if not isinstance(raw['direction_bank'], str) or not raw['direction_bank'].strip():
            raise ValueError('direction_bank requires a file path')
        paths['direction_bank'] = (path.parent / raw['direction_bank']).resolve()
        if not paths['direction_bank'].is_file():
            raise ValueError('Missing learned direction bank')
    manifest = json.loads(paths['lens_manifest'].read_text())
    required_manifest = {'format', 'lens_file', 'sha256', 'model_identity', 'source_layers', 'n_prompts'}
    if (not isinstance(manifest, dict) or set(manifest) != required_manifest
            or manifest['format'] != 'upstream-jlens-v1' or manifest['model_identity'] != raw['model_identity']
            or type(manifest['n_prompts']) is not int or manifest['n_prompts'] < 1
            or not isinstance(manifest['source_layers'], list)
            or any(type(i) is not int or not 0 <= i < raw['model_identity']['layers'] for i in manifest['source_layers'])
            or len(set(manifest['source_layers'])) != len(manifest['source_layers'])
            or not set(settings.checkpoints[:-1]) <= set(manifest['source_layers'])
            or not isinstance(manifest['sha256'], str) or re.fullmatch('[0-9a-f]{64}', manifest['sha256']) is None
            or not isinstance(manifest['lens_file'], str) or not manifest['lens_file'].strip()):
        raise ValueError('Tree lens manifest model/provenance/checkpoint mismatch')
    lens_file = (paths['lens_manifest'].parent / manifest['lens_file']).resolve()
    if file_digest(lens_file) != manifest['sha256']:
        raise ValueError('Tree lens artifact hash mismatch')
    suite = load_suite(paths['suite'])
    if any(row.get('split') in {'holdout', 'test', 'certification'} for row in suite):
        raise ValueError('Tree tuning cannot consume a marked holdout/test/certification suite')
    if 'behavior' in raw:
        from .tree_behavior_data import validate_behavior
        validate_behavior(raw, path.parent, paths, required_expansions)
        if any(not isinstance(row.get('family_id'), str) or not row['family_id'].strip()
               or row.get('family_origin') == 'synthetic-row-group' for row in suite):
            raise ValueError('Behavior data requires real authored topic-family IDs, not synthetic row groups')
    # Include fp32 J matrices, selected readout rows and all compact directions.
    d = raw['model_identity']['hidden_size']
    # A bank may carry full-rank bases: budget the maximum supported rank rather
    # than inspecting tensor payloads in this runtime-free metadata command.
    direction_elements = sum(d * d if bank_reference(value) else d for value in actions.values())
    lens_bound = 4 * (len(settings.checkpoints) * d * d + sum(map(len, groups)) * d
                      + direction_elements * len(settings.checkpoints[:-1]))
    if 'behavior' in raw:
        lens_bound += 4 * len(raw['behavior']['feature_tokens']) * d
    if lens_bound > raw['max_lens_bytes']:
        raise ValueError(f'Tree lens budget too small; resident readout bound is {lens_bound} bytes')
    # Outputs must never contain/replace their own inputs on a resumed job.
    for source in (path, paths['suite'], paths['lens_manifest'], lens_file,
                   *([paths['direction_bank']] if needs_bank else []),
                   *([paths['behavior_artifact']] if 'behavior_artifact' in paths else [])):
        if source == paths['output'] or paths['output'] in source.parents:
            raise ValueError('Tree output overlaps its configuration/input assets')
    return raw, paths, settings, manifest, lens_file, suite, required_expansions, lens_bound


def validate_tree(path):
    raw, paths, settings, manifest, lens_file, suite, expansions, lens_bound = read_tree_config(path)
    return {'status': 'metadata-valid', 'experimental': True, 'suite_role': 'development',
            'prompts': len(suite), 'required_expansions_per_prompt': expansions,
            'resident_lens_bound_bytes': lens_bound, 'output': str(paths['output']),
            'architecture': raw['model_identity']['architecture'], 'checkpoints': list(settings.checkpoints),
            'lens_sha256': manifest['sha256'], 'lens_origin': 'operator-attested; upstream file has no model ID',
            'not_checked': ['lens tensor contents', 'model/tokenizer availability', 'B70 fit/parity',
                            'proxy calibration on edited states', 'answer quality', 'capability preservation']}
