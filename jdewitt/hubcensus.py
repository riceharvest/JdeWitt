# SPDX-License-Identifier: AGPL-3.0-or-later
"""Live Hub census joined against the adapter table; run it to produce the worklist.

The architecture snapshot in assets/ answers "what do we support". This answers
"what shipped recently that we would have to write adapters for", ordered by
demand. Labels come from config.json model_type on the Hub, never from model
names; unsupported but downloaded model types are adapter candidates, not
guaranteed work.

Two kinds of unsupported rows are kept apart on purpose. A repository tagged
`no-model-type` carries no config.json model_type, so it is not an architecture
and never an adapter target: it lands in `untyped_repositories`. Everything else
lands in `adapter_worklist`.

`out=` persists the same payload as a census artifact for the quarter. The write
is an exclusive create, so a re-run of the same date refuses to clobber it
(FILE_EXISTS): a census is a dated observation, not a regenerated cache.
"""
from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

from .architectures import ARCHITECTURES
from .clock import now_iso

UNTYPED_LABEL = 'no-model-type'

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


def quarter_of(stamp):
    """Calendar quarter of a `now_iso()` stamp; the census cadence is quarterly."""
    at = datetime.datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%SZ')
    return f'{at.year}-Q{(at.month - 1) // 3 + 1}'


def census_next_steps(worklist):
    """Per-architecture capture/tune worklist in the census's own download order.

    One capture reads the checkpoint the census actually saw (the row's first
    example model) and one tune ranks edit arms on the artifact exported from
    it. Records and artifact names come from the model_type, so two rows never
    write to the same file; the artifact and suite paths stay quoted
    placeholders because this census never built them. `--ctx-size` carries a
    literal default (capture requires one) so the line is a command the CLI
    parses, not a sketch.
    """
    steps = []
    for row in worklist:
        model = row['example_models'][0] if row.get('example_models') else '<checkpoint>'
        slug = row['model_type'].replace('/', '-')
        artifact = f'"<{slug}-artifact.gguf>"'
        steps.append(f'uv run jdewitt capture --model {model} --suite "<suite.jsonl>" '
                     f'--ctx-size 4096 --out records-{slug}.pt --gguf {artifact}')
        steps.append(f'uv run jdewitt tune --gguf {artifact} --records records-{slug}.pt')
    return steps


def hub_census(authors=None, since='2026-08-01', per_author=40, out=None):
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
                row = by_type.setdefault(model_type or UNTYPED_LABEL, {'models': [], 'downloads': 0})
                row['models'].append(model.id)
                row['downloads'] += int(getattr(model, 'downloads', 0) or 0)
        except Exception as e:
            errors.append({'author': author, 'error': str(e)})
    unsupported = sorted(((t, r) for t, r in by_type.items() if t not in ARCHITECTURES),
                        key=lambda tr: -tr[1]['downloads'])
    worklist, untyped = [], []
    for model_type, row in unsupported:
        examples = row['models'][:3]
        if model_type == UNTYPED_LABEL:
            untyped.append({'label': model_type, 'downloads': row['downloads'], 'example_models': examples})
        else:
            worklist.append({'model_type': model_type, 'downloads': row['downloads'], 'example_models': examples})
    stamp = now_iso()
    result = {'scope': 'Live Hub text-generation census; adapter labels come from config.json '
                       'model_type, downloads are a demand proxy, nothing here is a support claim.',
              'window': {'since': since, 'authors': authors, 'per_author_cap': per_author},
              'census_at': stamp,
              'quarter': quarter_of(stamp),
              'repositories_checked': checked,
              'derived_artifact_repositories': derived,
              'model_types_seen': {t: r['downloads'] for t, r in sorted(by_type.items(), key=lambda tr: -tr[1]['downloads'])},
              'adapter_worklist': worklist,
              'untyped_repositories': untyped,
              'supported_model_types': sorted(t for t in by_type if t in ARCHITECTURES),
              'unresolved_gaps': UNRESOLVED_GAPS,
              'author_errors': errors,
              'next': census_next_steps(worklist)}
    if out is not None:
        from .schemas import enforce
        # Refuse to persist a census that its own published contract rejects.
        enforce('hub-census', result)
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = path.open('x', encoding='utf-8')
        except FileExistsError:
            raise FileExistsError(f'{path} already exists; a census artifact is never overwritten') from None
        with handle:
            handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
