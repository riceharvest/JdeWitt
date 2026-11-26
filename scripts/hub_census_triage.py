# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render the hub-census triage table from a saved census artifact.

Usage: .venv/bin/python scripts/hub_census_triage.py verification/hub-census/2026-11-22.json \\
           --out docs/HUB-CENSUS.md --evidence verification/hub-census/2026-11-22-loader.json

The census artifact is the only source of demand numbers: downloads, examples and
counts are copied out of it, never retyped. What this script adds is the loader
half of the decision, measured live:

- installed support: `model_type in CONFIG_MAPPING_NAMES` for the transformers in
  the current environment;
- published loader floor: the first PyPI-released transformers tag whose
  `src/transformers/models/auto/auto_mappings.py` registers the type, searched
  from the installed version up to the latest release;
- `auto_map`: whether the example checkpoints in the artifact ship remote-code
  configs, which is what lets the installed build read a type it does not know.

Decision rule, applied per row: untyped repositories are not architectures; a row
whose config the installed build can read (registered type or auto_map) is an
adapter card; otherwise it is blocked on a loader — named by the floor when a
published release registers the type, and called out as unregistered otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

PYPI = 'https://pypi.org/pypi/transformers/json'
RELEASE = 'https://pypi.org/pypi/transformers/{v}/json'
RAW = 'https://raw.githubusercontent.com/huggingface/transformers/v{v}/src/transformers/models/auto/auto_mappings.py'
UNTYPED_LABEL = 'no-model-type'


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'jdewitt-hub-census/1'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def fetch_text(url):
    return fetch(url).decode('utf-8')


def version_tuple(text):
    return tuple(int(part) for part in text.split('.')[:3])


def published_releases():
    document = json.loads(fetch_text(PYPI))
    versions = [v for v in document['releases'] if re.fullmatch(r'\d+\.\d+\.\d+', v)]
    versions.sort(key=version_tuple)
    return document['info']['version'], versions


def mapping_source(version):
    """`auto_mappings.py` for one published release, plus where it was read from.

    The GitHub tag is the fast path; a release whose tag is absent (5.21.0 has
    none) falls back to the published wheel, so no release in the range is
    silently skipped.
    """
    import io
    import urllib.error
    import zipfile
    try:
        return fetch_text(RAW.format(v=version)), 'github-tag'
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    files = json.loads(fetch_text(RELEASE.format(v=version)))['urls']
    wheel = next((f['url'] for f in files if f['packagetype'] == 'bdist_wheel'), None)
    if wheel is None:
        raise RuntimeError(f'transformers {version} publishes no wheel and has no readable tag')
    with zipfile.ZipFile(io.BytesIO(fetch(wheel))) as archive:
        member = next(name for name in archive.namelist() if name.endswith('models/auto/auto_mappings.py'))
        return archive.read(member).decode('utf-8'), 'pypi-wheel'


def installed_support():
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    return transformers.__version__, set(CONFIG_MAPPING_NAMES)


def loader_floors(model_types, installed, releases):
    """First published tag after the installed one that registers each type."""
    tags = [v for v in releases if version_tuple(v) > version_tuple(installed)]
    floors = {name: None for name in model_types}
    sources = {}
    for version in tags:
        text, source = mapping_source(version)
        sources[version] = source
        for name in model_types:
            if floors[name] is None and f'"{name}"' in text:
                floors[name] = version
        if all(floors.values()):
            break
    checked = [installed] + list(sources)
    return floors, checked, sources


def auto_map_support(example_models):
    """How many of a row's example checkpoints carry remote-code configs."""
    from huggingface_hub import hf_hub_download
    have = 0
    for repo in example_models:
        try:
            config = json.loads(Path(hf_hub_download(repo, 'config.json')).read_text())
        except Exception:
            continue
        if config.get('auto_map'):
            have += 1
    return have, len(example_models)


def decide(label, installed, auto_map_have, floor):
    if label == UNTYPED_LABEL:
        return 'not-an-architecture'
    if label in installed or auto_map_have:
        return 'adapter card'
    if floor:
        return f'blocked on loader: transformers>={floor}'
    return 'blocked on loader: no published loader registers this type'


def cell_models(examples):
    return ', '.join(f'`{repo}`' for repo in examples) if examples else '—'


def next_section(artifact):
    """The artifact's own `next` list, rendered rather than retyped.

    A census written before the next block existed has no list; in that case the
    section is omitted instead of inventing commands.
    """
    steps = artifact.get('next') or []
    if not steps:
        return []
    return [
        f'## Next — {len(steps)} commands, a capture/tune pair per worklist row',
        '',
        "Copied from the artifact's `next` list (one pair per architecture, in the same "
        'download order). Each capture reads the checkpoint its row actually saw; angle-bracket '
        'paths are placeholders this census never built, and every line parses as a CLI command.',
        '',
        '```',
        *steps,
        '```',
        '',
    ]


def render(artifact_path, artifact, evidence):
    rows = []
    for row in artifact['adapter_worklist']:
        rows.append({**row, **evidence['rows'][row['model_type']]})
    untyped = [dict(row, **evidence['rows'][row['label']]) for row in artifact['untyped_repositories']]
    lines = [
        '# Hub census and adapter triage',
        '',
        f"Census artifact `{artifact_path}`: taken {artifact['census_at']} ({artifact['quarter']}), "
        f"sha256 `{evidence['artifact_sha256']}`.",
        '',
        f"{artifact['repositories_checked']} text-generation repositories checked across "
        f"{len(artifact['window']['authors'])} publishers (created since {artifact['window']['since']}, "
        f"cap {artifact['window']['per_author_cap']} per publisher); "
        f"{artifact['derived_artifact_repositories']} derived GGUF/MLX artifacts skipped; "
        f"{len(artifact['model_types_seen'])} model types seen; "
        f"{len(artifact['author_errors'])} author errors. "
        f"Installed transformers {evidence['installed_transformers']}; latest published on PyPI "
        f"{evidence['latest_transformers']}.",
        '',
        'Every number below is read from that artifact or measured by '
        '`scripts/hub_census_triage.py`; none is typed by hand.',
        '',
        f"## Adapter worklist — {len(rows)} rows, ordered by downloads",
        '',
        '| model_type | downloads | example models | installed transformers '
        f"{evidence['installed_transformers']} | auto_map | first published loader | decision |",
        '|---|---:|---|---|---|---|---|',
    ]
    for row in rows:
        supports = 'yes' if row['installed_support'] else 'no'
        loader = row['floor'] or 'none found'
        lines.append(f"| `{row['model_type']}` | {row['downloads']:,} | {cell_models(row['example_models'])} "
                     f"| {supports} | {row['auto_map']} | {loader} | {row['decision']} |")
    lines += ['', f"## Untyped repositories — {len(untyped)} row" + ('s' if len(untyped) != 1 else ''), '',
              '| label | downloads | example models | decision |', '|---|---:|---|---|']
    for row in untyped:
        lines.append(f"| `{row['label']}` | {row['downloads']:,} | {cell_models(row['example_models'])} "
                     f"| {row['decision']} |")
    lines += ['',
              'These repositories have no `config.json` `model_type` (some have no `config.json` at all). '
              'A missing label is a census artifact, not an architecture: they stay out of '
              '`adapter_worklist` and no adapter card is written for them.',
              '',
              '## How the decision was made',
              '',
              '- `installed transformers`: whether `model_type` is a key of `CONFIG_MAPPING_NAMES` in the '
              'transformers installed for this run' + (
                  '; none of the worklist rows is, so the installed build cannot name these '
                  'architectures itself.' if not evidence['installed_hits'] else
                  '; the rows the installed build does name are marked yes and would not need a '
                  'loader change.'),
              '- `auto_map`: how many of the row\'s example checkpoints ship remote-code configs. Those '
              'load with `trust_remote_code` today, so writing the adapter is not blocked on a release; '
              'a published floor is still recorded because installing it drops the remote-code dependency.',
              '- `first published loader`: the first PyPI-released transformers version at or above the '
              'installed one whose `src/transformers/models/auto/auto_mappings.py` registers the type, '
              f"searched over {', '.join(evidence['checked_tags'])} (read from the GitHub tag, or from "
              'the published wheel where a release has no tag).',
              '- A row is `blocked on loader` only when the installed build can read neither a registered '
              'type nor an `auto_map`. The decision names the floor that unblocks it, or says that no '
              'published loader registers the type.',
              '',
              *next_section(artifact),
              '## Human gap list',
              '',
              f"The curated `unresolved_gaps` list stays in the same artifact ({len(artifact['unresolved_gaps'])} "
              'entries) and is not re-derived here.',
              '',
              '## Reproduce',
              '',
              '```',
              f".venv/bin/python -m jdewitt hub-census --out {artifact_path}",
              f".venv/bin/python scripts/hub_census_triage.py {artifact_path} "
              f"--out docs/HUB-CENSUS.md --evidence {evidence.get('evidence_path', 'EVIDENCE.json')}",
              f".venv/bin/python -m jdewitt verify-artifact {artifact_path}",
              '```',
              'The census write is an exclusive create: re-running the same date exits 1 with '
              '`FILE_EXISTS` instead of overwriting the observation.',
              '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('artifact', type=Path)
    parser.add_argument('--out', type=Path, help='Write the markdown here instead of stdout')
    parser.add_argument('--evidence', type=Path, help='Write the loader evidence JSON here')
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text())
    raw = args.artifact.read_bytes()
    installed_version, installed = installed_support()
    latest, releases = published_releases()
    labels = [row['model_type'] for row in artifact['adapter_worklist']]
    floors, checked, sources = loader_floors(labels, installed_version, releases)
    rows = {}
    for row in artifact['adapter_worklist']:
        have, total = auto_map_support(row['example_models'])
        label = row['model_type']
        rows[label] = {'installed_support': label in installed,
                       'auto_map': f'{have}/{total}',
                       'auto_map_have': have,
                       'floor': floors[label],
                       'decision': decide(label, installed, have, floors[label])}
    for row in artifact['untyped_repositories']:
        rows[row['label']] = {'installed_support': None, 'auto_map': 'n/a', 'auto_map_have': 0,
                              'floor': None, 'decision': decide(row['label'], installed, 0, None)}
    evidence = {'artifact': str(args.artifact), 'artifact_sha256': hashlib.sha256(raw).hexdigest(),
                'census_at': artifact['census_at'], 'quarter': artifact['quarter'],
                'installed_transformers': installed_version, 'latest_transformers': latest,
                'checked_tags': checked, 'rows': rows,
                'mapping_sources': sources,
                'installed_hits': [label for label in labels if label in installed]}
    if args.evidence:
        evidence['evidence_path'] = str(args.evidence)
    markdown = render(args.artifact, artifact, evidence)
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(evidence, indent=2) + '\n')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown)
    else:
        print(markdown, end='')


if __name__ == '__main__':
    main()
