# SPDX-License-Identifier: AGPL-3.0-or-later
import hashlib
import json
import unicodedata
from pathlib import Path


def canonical_prompt(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_suite(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty suite')
    ids, prompts = set(), set()
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('id', 'prompt', 'language', 'kind')):
            raise ValueError('Suite requires nonempty id, prompt, kind and language')
        if row['kind'] not in {'refusal', 'harmless', 'math', 'code', 'direction'}:
            raise ValueError('Unknown prompt kind')
        key = canonical_prompt(row['prompt'])
        if row['id'] in ids or key in prompts:
            raise ValueError('Duplicate id or normalized prompt')
        ids.add(row['id'])
        prompts.add(key)
        if row['kind'] == 'math' and not isinstance(row.get('expected'), str):
            raise ValueError('Math task requires an expected answer string')
        if row['kind'] == 'code' and (not isinstance(row.get('tests'), str) or not row['tests'].strip()):
            raise ValueError('Code task requires actual test assertions')
        if row['kind'] == 'direction' and row.get('label') not in {'good', 'bad'}:
            raise ValueError('Direction task requires good/bad label')
    return rows


def validate_splits(splits):
    seen = {}
    families = {}
    for name, rows in splits.items():
        for row in rows:
            key = canonical_prompt(row['prompt'])
            if key in seen:
                raise ValueError(f'Prompt overlap: {seen[key]} and {name}')
            seen[key] = name
            family = row.get('family_id')
            if family is not None:
                if not isinstance(family, str) or not family.strip():
                    raise ValueError('Invalid family_id')
                if family in families and families[family] != name:
                    raise ValueError(f'Prompt family overlap: {families[family]} and {name}')
                families[family] = name


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        import os
        os.fsync(f.fileno())
