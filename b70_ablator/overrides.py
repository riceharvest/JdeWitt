# SPDX-License-Identifier: AGPL-3.0-or-later
"""JSON configuration overrides for --set, applied before any configuration validation."""
import json


def _declared_fields():
    """Every declared configuration field, so a defaulted field can be set even when absent from the file."""
    from dataclasses import fields
    from .search import RunConfig
    from .auto import AutoConfig
    return {f.name for f in fields(RunConfig)} | {f.name for f in fields(AutoConfig)}


def apply_overrides(config, assignments):
    """Override dotted JSON paths in a parsed configuration. Unknown top-level fields are refused."""
    for assignment in assignments:
        if '=' not in assignment:
            raise ValueError('--set expects FIELD=VALUE')
        path, text = assignment.split('=', 1)
        path = path.strip()
        if not path:
            raise ValueError('--set field name cannot be empty')
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        keys = path.split('.')
        if keys[0] not in config and keys[0] not in _declared_fields():
            raise ValueError(f'--set target {keys[0]!r} is not a field of this configuration')
        node = config
        for key in keys[:-1]:
            if key not in node:
                node[key] = {}
            elif not isinstance(node[key], dict):
                raise ValueError(f'--set path segment {key!r} is not a JSON object')
            node = node[key]
        node[keys[-1]] = value
    return config
