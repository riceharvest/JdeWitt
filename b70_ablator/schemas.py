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
    }


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
    name = Path(path).name
    if name not in SCHEMAS:
        raise ValueError(f'No published schema for {name!r}; available: {sorted(SCHEMAS)}')
    return name


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
