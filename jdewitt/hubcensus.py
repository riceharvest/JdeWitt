# SPDX-License-Identifier: AGPL-3.0-or-later
"""Live Hub census joined against the adapter table; run it to produce the worklist.

The architecture snapshot in assets/ answers "what do we support". This answers
"what shipped recently that we would have to write adapters for", ordered by
demand. Labels come from config.json model_type on the Hub, never from model
names; unsupported but downloaded model types are adapter candidates, not
guaranteed work.
"""
from __future__ import annotations

import datetime
import json
import re

from .architectures import ARCHITECTURES

# Carried forward from docs/ROADMAP.md; humans curate this list, the code never guesses it.
UNRESOLVED_GAPS = [
    'DeepSeek V4.1 loaders', 'Kimi K3 loaders',
    'MiMo V2.6 / Bolmo / Youtu wrappers', 'custom Ling VL',
    'draft-only Muse assistant', 'Qwen Drive planning models',
]


def inventory_authors():
    from importlib.resources import files
    inventory = json.loads(files('jdewitt').joinpath('assets/architecture-inventory.json').read_text())
    return [a['author'] if isinstance(a, dict) else a for a in inventory.get('authors', [])]


def _parse_instant(value):
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
    return datetime.datetime.strptime(str(value).split('.')[0].rstrip('Z'), '%Y-%m-%dT%H:%M:%S').replace(tzinfo=datetime.timezone.utc)


def hub_census(authors=None, since='2026-08-01', per_author=40):
    from huggingface_hub import HfApi
    api = HfApi()
    authors = authors or inventory_authors()
    since_dt = datetime.datetime.fromisoformat(since).replace(tzinfo=datetime.timezone.utc)
    checked = 0
    derived = 0
    by_type = {}
    errors = []
    for author in authors:
        try:
            # Created-descending; stop the moment a repository predates the window.
            for model in api.list_models(author=author, pipeline_tag='text-generation',
                                         sort='created_at', limit=per_author, fetch_config=True):
                created = _parse_instant(getattr(model, 'created_at', None))
                if created is not None and created < since_dt:
                    break
                checked += 1
                config = getattr(model, 'config', None) or {}
                model_type = config.get('model_type') if isinstance(config, dict) else None
                if model_type is None and re.search(r'gguf|ggml|mlx', model.id, re.I):
                    derived += 1
                    continue
                row = by_type.setdefault(model_type or 'no-model-type', {'models': [], 'downloads': 0})
                row['models'].append(model.id)
                row['downloads'] += int(getattr(model, 'downloads', 0) or 0)
        except Exception as e:
            errors.append({'author': author, 'error': str(e)})
    unsupported = sorted(((t, r) for t, r in by_type.items() if t not in ARCHITECTURES),
                        key=lambda tr: -tr[1]['downloads'])
    return {'scope': 'Live Hub text-generation census; adapter labels come from config.json '
                     'model_type, downloads are a demand proxy, nothing here is a support claim.',
            'window': {'since': since, 'authors': authors, 'per_author_cap': per_author},
            'repositories_checked': checked,
            'derived_artifact_repositories': derived,
            'model_types_seen': {t: r['downloads'] for t, r in sorted(by_type.items(), key=lambda tr: -tr[1]['downloads'])},
            'adapter_worklist': [{'model_type': t, 'downloads': r['downloads'],
                                  'example_models': r['models'][:3]} for t, r in unsupported],
            'supported_model_types': sorted(t for t in by_type if t in ARCHITECTURES),
            'unresolved_gaps': UNRESOLVED_GAPS,
            'author_errors': errors}
