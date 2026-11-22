# SPDX-License-Identifier: AGPL-3.0-or-later
"""Published JSON Schemas for on-disk artifacts and a dependency-free shape validator.

The schemas describe the versioned contracts this tool writes, so a third party
can check an artifact offline. They mirror the writers in release.py and
answers.py; when a writer version bumps, both the const version and the schema
bump together or nothing is silent.
"""
import json
from pathlib import Path

SCHEMAS = {
    'files.json': {
        'description': 'Integrity index written beside every release bundle.',
        'type': 'object',
        'required': ['version', 'release_identity', 'files', 'record_digest'],
        'properties': {
            'version': {'const': 1},
            'release_identity': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
            'files': {'type': 'object', 'additionalProperties': {
                'type': 'object', 'required': ['sha256', 'bytes'],
                'properties': {'sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
                               'bytes': {'type': 'integer'}}}},
            'record_digest': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}},
    },
    'evidence.json': {
        'description': 'Chained claims ledger of a release.',
        'type': 'object',
        'required': ['version', 'release_identity', 'entries', 'ledger_digest'],
        'properties': {
            'version': {'const': 1},
            'release_identity': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
            'entries': {'type': 'array', 'items': {
                'type': 'object',
                'required': ['seq', 'claim', 'evidence', 'verified_by', 'at', 'previous_digest', 'digest'],
                'properties': {'seq': {'type': 'integer'}, 'claim': {'type': 'string'},
                               'evidence': {'type': 'string'}, 'verified_by': {'type': 'string'},
                               'at': {'type': 'string'}, 'previous_digest': {'type': 'string'},
                               'digest': {'type': 'string'}}}},
            'ledger_digest': {'type': 'string'}},
    },
    'signature.json': {
        'description': 'Ed25519 signature over a release reproduction index.',
        'type': 'object',
        'required': ['version', 'algorithm', 'public_key_pem', 'payload', 'signature_base64'],
        'properties': {'version': {'const': 1}, 'algorithm': {'const': 'ed25519'},
                       'public_key_pem': {'type': 'string'}, 'payload': {'type': 'object'},
                       'signature_base64': {'type': 'string'}}},
    'scoring-manifest.json': {
        'description': 'Identity binding a scoring directory to one generation and one scorer.',
        'type': 'object',
        'required': ['version', 'suite_digest', 'generation_digest', 'identity', 'sandbox_image', 'scoring_digest'],
        'properties': {'version': {'const': 2}, 'suite_digest': {'type': 'string'},
                       'generation_digest': {'type': 'string'}, 'identity': {'type': 'object'},
                       'sandbox_image': {'type': 'string'}, 'scoring_digest': {'type': 'string'}}},
    'run-status.command.json': {
        'description': 'run-status output contract.',
        'type': 'object',
        'required': ['output', 'trial_counts', 'trials', 'cancel_requested', 'release_present', 'note'],
        'properties': {'output': {'type': 'string'}, 'trial_counts': {'type': 'object'},
                       'trials': {'type': 'array'}, 'note': {'type': 'string'}}},
    'studies.command.json': {
        'description': 'studies output contract.',
        'type': 'object',
        'required': ['root', 'study_count', 'studies', 'note'],
        'properties': {'root': {'type': 'string'}, 'study_count': {'type': 'integer'},
                       'studies': {'type': 'array'}, 'note': {'type': 'string'}}},
    'describe-run.command.json': {
        'description': 'describe-run output contract.',
        'type': 'object',
        'required': ['config', 'status', 'suites', 'budget_estimates_upper_bounds', 'note'],
        'properties': {'config': {'type': 'string'}, 'status': {'type': 'string'},
                       'suites': {'type': 'object'}, 'split_family_collisions': {'type': 'array'},
                       'budget_estimates_upper_bounds': {'type': 'object'}, 'note': {'type': 'string'}}},
    'validate-auto.command.json': {
        'description': 'validate-auto / abliterate dry-run output contract.',
        'type': 'object',
        'required': ['status', 'trial_budget', 'candidates', 'models_loaded', 'output_created'],
        'properties': {'status': {'const': 'metadata-valid'}, 'trial_budget': {'type': 'integer'},
                       'candidates': {'type': 'array'}, 'holdout_policy': {'type': 'string'}}},
    'methods.command.json': {
        'description': 'methods output contract.',
        'type': 'object',
        'required': ['scope', 'methods', 'edit_surface']},
    'architectures.command.json': {
        'description': 'architectures output contract.',
        'type': 'object',
        'required': ['scope', 'architectures']},
    'score-compare.command.json': {
        'description': 'score-compare output contract.',
        'type': 'object',
        'required': ['suite', 'before', 'after', 'pass_on', 'shared_prompts', 'flip_count',
                     'flips', 'list_cap', 'before_summary', 'after_summary', 'note'],
        'properties': {'effect_size_cohens_h': {'type': 'number'}, 'suite': {'type': 'string'}, 'before': {'type': 'string'},
                       'after': {'type': 'string'}, 'pass_on': {'type': 'string'},
                       'shared_prompts': {'type': 'integer'}, 'flip_count': {'type': 'integer'},
                       'flips': {'type': 'array'}, 'list_cap': {'type': 'integer'},
                       'before_summary': {'type': 'object'}, 'after_summary': {'type': 'object'},
                       'family_bootstrap': {'type': 'object'}, 'paired_permutation': {'type': 'object'},
                       'note': {'type': 'string'}}},
    'judge-ensemble.command.json': {
        'description': 'judge-ensemble output contract.',
        'type': 'object',
        'required': ['suite', 'directories', 'min_agreement', 'prompts_scored_by_any_judge',
                     'consensus_count', 'contested_count', 'consensus', 'contested', 'note'],
        'properties': {'suite': {'type': 'string'}, 'directories': {'type': 'object'},
                       'consensus_count': {'type': 'integer'}, 'contested_count': {'type': 'integer'},
                       'consensus': {'type': 'array'}, 'contested': {'type': 'array'},
                       'note': {'type': 'string'}}},
    'calibration-report.command.json': {
        'description': 'calibration-report output contract.',
        'type': 'object',
        'required': ['scope', 'suite', 'rows_in_suite', 'rows_with_consensus', 'per_judge',
                     'pairwise_agreement', 'note'],
        'properties': {'scope': {'type': 'string'}, 'suite': {'type': 'string'},
                       'rows_in_suite': {'type': 'integer'}, 'rows_with_consensus': {'type': 'integer'},
                       'per_judge': {'type': 'object'}, 'pairwise_agreement': {'type': 'object'},
                       'authored_ledger': {'type': 'object'}, 'note': {'type': 'string'}}},
    'direction-export.command.json': {
        'description': 'direction-export output contract.',
        'type': 'object',
        'required': ['output', 'plan_key', 'format', 'tensors', 'metadata_entries',
                     'data_base', 'file_size', 'note'],
        'properties': {'output': {'type': 'string'}, 'plan_key': {'type': 'string'},
                       'format': {'type': 'string', 'const': 'direction-gguf-v1'},
                       'tensors': {'type': 'integer'}, 'metadata_entries': {'type': 'integer'},
                       'data_base': {'type': 'integer'}, 'file_size': {'type': 'integer'},
                       'note': {'type': 'string'}}},
    'gguf-delta-pack.command.json': {
        'description': 'gguf-delta-pack output contract.',
        'type': 'object',
        'required': ['output', 'tensors', 'metadata_entries', 'data_base', 'file_size'],
        'properties': {'output': {'type': 'string'}, 'tensors': {'type': 'integer'},
                       'metadata_entries': {'type': 'integer'},
                       'data_base': {'type': 'integer'}, 'file_size': {'type': 'integer'}}},
    'suites-compare.command.json': {
        'description': 'suites-compare output contract.',
        'type': 'object',
        'required': ['families', 'tests_in_family', 'holm_min_adjusted_p', 'note'],
        'properties': {'families': {'type': 'array'}, 'tests_in_family': {'type': 'integer'},
                       'holm_min_adjusted_p': {'type': 'number'}, 'note': {'type': 'string'}}},
    'studies-compare.command.json': {
        'description': 'studies-compare output contract.',
        'type': 'object',
        'required': ['databases', 'param_stats', 'note'],
        'properties': {'databases': {'type': 'array'}, 'param_stats': {'type': 'object'},
                       'note': {'type': 'string'}}},
    'quants-audit.command.json': {
        'description': 'quants-audit output contract.',
        'type': 'object',
        'required': ['file', 'dtypes_audited', 'per_dtype', 'decode_only_dtypes',
                     'port_decode_only_dtypes', 'unsupported_dtypes', 'note'],
        'properties': {'file': {'type': 'string'}, 'dtypes_audited': {'type': 'array'},
                       'per_dtype': {'type': 'object'}, 'decode_only_dtypes': {'type': 'object'},
                       'port_decode_only_dtypes': {'type': 'object'},
                       'unsupported_dtypes': {'type': 'object'}, 'note': {'type': 'string'}}},
    'verdict-audit.command.json': {
        'description': 'verdict-audit output contract.',
        'type': 'object',
        'required': ['directory', 'rows_total', 'compared', 'agreed', 'agreement_rate',
                     'skipped_states', 'dangerous_disagreements', 'dangerous_count',
                     'other_disagreement_kinds', 'other_disagreement_examples',
                     'check_method', 'note'],
        'properties': {'directory': {'type': 'string'}, 'rows_total': {'type': 'integer'},
                       'compared': {'type': 'integer'}, 'agreed': {'type': 'integer'},
                       'skipped_states': {'type': 'object'},
                       'dangerous_disagreements': {'type': 'array'}, 'dangerous_count': {'type': 'integer'},
                       'other_disagreement_kinds': {'type': 'object'},
                       'other_disagreement_examples': {'type': 'array'},
                       'check_method': {'type': 'string',
                                        'const': 'explicit-wording-v1'},
                       'note': {'type': 'string'}}},
    'suite-stats.command.json': {
        'description': 'suite-stats output contract.',
        'type': 'object',
        'required': ['suite', 'prompts', 'kinds', 'languages', 'families', 'note'],
        'properties': {'suite': {'type': 'string'}, 'prompts': {'type': 'integer'},
                       'kinds': {'type': 'object'}, 'languages': {'type': 'object'},
                       'families': {'type': 'object'}}},
    'suite-diff.command.json': {
        'description': 'suite-diff output contract.',
        'type': 'object',
        'required': ['before', 'after', 'digest_before', 'digest_after', 'counts', 'added',
                     'removed', 'changed', 'note'],
        'properties': {'before': {'type': 'string'}, 'after': {'type': 'string'},
                       'digest_before': {'type': 'string'}, 'digest_after': {'type': 'string'},
                       'counts': {'type': 'object'}, 'added': {'type': 'array'},
                       'removed': {'type': 'array'}, 'changed': {'type': 'array'}}},
    'study-metrics.command.json': {
        'description': 'study-metrics output contract.',
        'type': 'object',
        'required': ['study', 'trials', 'best', 'state_counts', 'note'],
        'properties': {'study': {'type': 'string'}, 'trials': {'type': 'integer'},
                       'best': {'type': 'object'}, 'state_counts': {'type': 'object'},
                       'note': {'type': 'string'}}},
    'family-decay.command.json': {
        'description': 'family-decay output contract.',
        'type': 'object',
        'required': ['fitted', 'held', 'global_decay', 'novel_families', 'shared_families', 'note'],
        'properties': {'fitted': {'type': 'object'}, 'held': {'type': 'object'},
                       'novel_families': {'type': 'array'}, 'shared_families': {'type': 'array'},
                       'warning': {'type': 'string'}, 'note': {'type': 'string'}}},
    'run-events.command.json': {
        'description': 'run-events output contract.',
        'type': 'object', 'required': ['events'],
        'properties': {'events': {'type': 'array', 'items': {'type': 'object', 'required': ['kind']}}}},
    'plan-list.command.json': {
        'description': 'plan-list output contract (array of plan rows).',
        'type': 'array', 'items': {'type': 'object', 'required': ['key', 'measured']}},
    'hub-census.command.json': {
        'description': 'hub-census output contract.',
        'type': 'object',
        'required': ['scope', 'window', 'census_at', 'quarter', 'repositories_checked',
                     'model_types_seen', 'adapter_worklist', 'untyped_repositories',
                     'supported_model_types', 'unresolved_gaps', 'author_errors',
                     'derived_artifact_repositories'],
        'properties': {'scope': {'type': 'string'}, 'window': {'type': 'object'},
                       'census_at': {'type': 'string', 'pattern': r'^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'},
                       'quarter': {'type': 'string', 'pattern': r'^[0-9]{4}-Q[1-4]$'},
                       'repositories_checked': {'type': 'integer'},
                       'derived_artifact_repositories': {'type': 'integer'},
                       'model_types_seen': {'type': 'object'},
                       'adapter_worklist': {'type': 'array', 'items': {
                           'type': 'object',
                           'required': ['model_type', 'downloads', 'example_models'],
                           'properties': {'model_type': {'type': 'string'},
                                          'downloads': {'type': 'integer'},
                                          'example_models': {'type': 'array'}}}},
                       'untyped_repositories': {'type': 'array', 'items': {
                           'type': 'object',
                           'required': ['label', 'downloads', 'example_models'],
                           'properties': {'label': {'type': 'string'},
                                          'downloads': {'type': 'integer'},
                                          'example_models': {'type': 'array'}}}},
                       'supported_model_types': {'type': 'array'}, 'unresolved_gaps': {'type': 'array'},
                       'author_errors': {'type': 'array'}}},
    'run-ls.command.json': {
        'description': 'run-ls output contract.',
        'type': 'object',
        'required': ['output', 'entries', 'unexpected_entries', 'note'],
        'properties': {'output': {'type': 'string'}, 'entries': {'type': 'array'},
                       'unexpected_entries': {'type': 'array'}, 'note': {'type': 'string'}}},
    'release-diff.command.json': {
        'description': 'release-diff output contract.',
        'type': 'object',
        'required': ['left', 'right', 'release_identities', 'files', 'claims',
                     'ledger_digests', 'signatures', 'note'],
        'properties': {'left': {'type': 'string'}, 'right': {'type': 'string'},
                       'release_identities': {'type': 'object'}, 'files': {'type': 'object'},
                       'claims': {'type': 'object'}, 'ledger_digests': {'type': 'object'},
                       'signatures': {'type': 'object'}, 'note': {'type': 'string'}}},
    'gguf-tensor-diff.command.json': {
        'description': 'gguf-tensor-diff output contract.',
        'type': 'object',
        'required': ['before', 'after', 'tensor_count_before', 'tensor_count_after',
                     'only_in_before', 'only_in_after', 'summary', 'rel_rms',
                     'differing_tensors', 'note'],
        'properties': {'before': {'type': 'string'}, 'after': {'type': 'string'},
                       'tensor_count_before': {'type': 'integer'},
                       'tensor_count_after': {'type': 'integer'},
                       'only_in_before': {'type': 'array'}, 'only_in_after': {'type': 'array'},
                       'summary': {'type': 'object'},
                       'rel_rms': {'type': 'object'},
                       'differing_tensors': {'type': 'array'}, 'note': {'type': 'string'}}},
    'safetensors-diff.command.json': {
        'description': 'safetensors-diff output contract.',
        'type': 'object',
        'required': ['before', 'after', 'tensor_count_before', 'tensor_count_after',
                     'only_in_before', 'only_in_after', 'summary', 'rel_rms',
                     'differing_tensors', 'note'],
        'properties': {'before': {'type': 'string'}, 'after': {'type': 'string'},
                       'tensor_count_before': {'type': 'integer'},
                       'tensor_count_after': {'type': 'integer'},
                       'only_in_before': {'type': 'array'}, 'only_in_after': {'type': 'array'},
                       'summary': {'type': 'object'},
                       'rel_rms': {'type': 'object'},
                       'differing_tensors': {'type': 'array'}, 'note': {'type': 'string'}}},
    'pt-diff.command.json': {
        'description': 'pt-diff output contract.',
        'type': 'object',
        'required': ['before', 'after', 'states', 'summary', 'differing',
                     'missing_in_after', 'missing_in_before', 'note'],
        'properties': {'before': {'type': 'string'}, 'after': {'type': 'string'},
                       'states': {'type': 'object'}, 'summary': {'type': 'object'},
                       'differing': {'type': 'array'},
                       'missing_in_after': {'type': 'array'},
                       'missing_in_before': {'type': 'array'},
                       'note': {'type': 'string'}}},
    'fp8-edit.command.json': {
        'description': 'fp8-edit output contract.',
        'type': 'object',
        'required': ['input', 'output', 'format', 'tensor', 'scales', 'elements',
                     'scale_mode', 'axis', 'block', 'note'],
        'properties': {'input': {'type': 'string'}, 'output': {'type': 'string'},
                       'format': {'type': 'string'}, 'tensor': {'type': 'string'},
                       'scales': {'type': 'string'}, 'elements': {'type': 'integer'},
                       'scale_mode': {'type': 'string'},
                       'axis': {}, 'block': {}, 'note': {'type': 'string'}}},
    'fp4-edit.command.json': {
        'description': 'fp4-edit output contract.',
        'type': 'object',
        'required': ['input', 'output', 'format', 'tensor', 'scales', 'elements',
                     'block', 'global_scale', 'note'],
        'properties': {'input': {'type': 'string'}, 'output': {'type': 'string'},
                       'format': {'type': 'string'}, 'tensor': {'type': 'string'},
                       'scales': {'type': 'string'}, 'elements': {'type': 'integer'},
                       'block': {'type': 'integer'}, 'global_scale': {},
                       'note': {'type': 'string'}}},
    'nf4-edit.command.json': {
        'description': 'nf4-edit output contract.',
        'type': 'object',
        'required': ['input', 'output', 'tensor', 'scales', 'elements', 'block',
                     'scale_dtype', 'note'],
        'properties': {'input': {'type': 'string'}, 'output': {'type': 'string'},
                       'tensor': {'type': 'string'}, 'scales': {'type': 'string'},
                       'elements': {'type': 'integer'}, 'block': {'type': 'integer'},
                       'scale_dtype': {'type': 'string'}, 'note': {'type': 'string'}}},
    'run-doctor.command.json': {
        'description': 'run-doctor output contract.',
        'type': 'object',
        'required': ['output', 'healthy', 'problems', 'trial_counts',
                     'run_lock_held', 'cancel_requested', 'note'],
        'properties': {'output': {'type': 'string'}, 'healthy': {'type': 'boolean'},
                       'problems': {'type': 'array', 'items': {
                           'type': 'object',
                           'required': ['code', 'detail', 'action'],
                           'properties': {'code': {'type': 'string'},
                                          'detail': {'type': 'string'},
                                          'action': {'type': 'string'}}}},
                       'trial_counts': {}, 'run_lock_held': {},
                       'cancel_requested': {'type': 'boolean'},
                       'note': {'type': 'string'}}},
    }

EMIT_SCHEMAS = {
    'run-status': 'run-status.command.json',
    'studies': 'studies.command.json',
    'describe-run': 'describe-run.command.json',
    'validate-auto': 'validate-auto.command.json',
    'abliterate': 'validate-auto.command.json',
    'methods': 'methods.command.json',
    'architectures': 'architectures.command.json',
    'score-compare': 'score-compare.command.json',
    'judge-ensemble': 'judge-ensemble.command.json',
    'calibration-report': 'calibration-report.command.json',
    'verdict-audit': 'verdict-audit.command.json',
    'direction-export': 'direction-export.command.json',
    'gguf-delta-pack': 'gguf-delta-pack.command.json',
    'suites-compare': 'suites-compare.command.json',
    'studies-compare': 'studies-compare.command.json',
    'quants-audit': 'quants-audit.command.json',
    'suite-stats': 'suite-stats.command.json',
    'suite-diff': 'suite-diff.command.json',
    'study-metrics': 'study-metrics.command.json',
    'family-decay': 'family-decay.command.json',
    'run-events': 'run-events.command.json',
    'plan-list': 'plan-list.command.json',
    'hub-census': 'hub-census.command.json',
    'run-ls': 'run-ls.command.json',
    'run-doctor': 'run-doctor.command.json',
    'release-diff': 'release-diff.command.json',
    'gguf-tensor-diff': 'gguf-tensor-diff.command.json',
    'safetensors-diff': 'safetensors-diff.command.json',
    'pt-diff': 'pt-diff.command.json',
    'fp8-edit': 'fp8-edit.command.json',
    'fp4-edit': 'fp4-edit.command.json',
    'nf4-edit': 'nf4-edit.command.json',
}


def enforce(command, value):
    """Command-output contract enforcement at emit time; raise if the writer drifted."""
    name = EMIT_SCHEMAS[command]
    problems = validate_against(value, SCHEMAS[name])
    if problems:
        raise ValueError(f'{command} output violates published schema {name}: '
                         + '; '.join(problems[:3]) + (f' ({len(problems)} total)' if len(problems) > 3 else ''))


def validate_against(value, schema, path='$'):
    """Small validator covering the subset used above; returns a list of problems."""
    problems = []
    kind = schema.get('type')
    if kind == 'object':
        if not isinstance(value, dict):
            return [f'{path}: expected object']
        for key in schema.get('required', []):
            if key not in value:
                problems.append(f'{path}.{key}: required and missing')
        for key, item in value.items():
            rule = schema.get('properties', {}).get(key, schema.get('additionalProperties'))
            if rule is not None and not isinstance(rule, bool):
                problems.extend(validate_against(item, rule, f'{path}.{key}'))
    elif kind == 'array':
        if not isinstance(value, list):
            return [f'{path}: expected array']
        rule = schema.get('items')
        if rule is not None:
            for index, item in enumerate(value):
                problems.extend(validate_against(item, rule, f'{path}[{index}]'))
    elif kind == 'string':
        if not isinstance(value, str):
            problems.append(f'{path}: expected string')
        elif 'pattern' in schema:
            import re
            if not re.fullmatch(schema['pattern'], value):
                problems.append(f'{path}: does not match {schema["pattern"]}')
    elif kind == 'integer':
        if type(value) is not int:
            problems.append(f'{path}: expected integer')
    if 'const' in schema and value != schema['const']:
        problems.append(f'{path}: expected constant {schema["const"]!r}, got {value!r}')
    return problems


def infer_schema_name(path):
    path = Path(path)
    if path.name in SCHEMAS:
        return path.name
    # An artifact persisted by a command keeps that command's name on its directory,
    # so `verification/hub-census/2026-11-22.json` is checked without repeating --name.
    emitted = EMIT_SCHEMAS.get(path.parent.name)
    if emitted:
        return emitted
    raise ValueError(f'No published schema for {path.name!r}; available: {sorted(SCHEMAS)}')


def validate_artifact(path, name=None):
    """Check one artifact against its published schema. File reads only."""
    path = Path(path)
    name = name or infer_schema_name(path)
    if name not in SCHEMAS:
        raise ValueError(f'No published schema for {name!r}; available: {sorted(SCHEMAS)}')
    document = json.loads(path.read_text())
    problems = validate_against(document, SCHEMAS[name])
    return {'file': str(path.resolve()), 'schema': name, 'valid': not problems,
            'problems': problems, 'note': 'Shape and version only; digests and semantics are checked by release verify commands.'}


def verify_all(directory):
    """Check every published artifact found in a directory against its schema. Reads only."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    artifact_names = [name for name in SCHEMAS if not name.endswith('.command.json')]
    checked, missing = [], []
    for name in sorted(artifact_names):
        path = directory / name
        if not path.is_file():
            missing.append(name)
            continue
        document = json.loads(path.read_text())
        problems = validate_against(document, SCHEMAS[name])
        checked.append({'file': str(path), 'schema': name, 'valid': not problems, 'problems': problems[:5]})
    return {'directory': str(directory.resolve()), 'checked': checked, 'missing_schemas': missing,
            'valid': all(row['valid'] for row in checked),
            'note': 'Shape and version only; digests and semantics are checked by release verify commands.'}
