#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the per-family capture suite for the P5 SAE lane, deterministically.

Input is a family-tagged suite (``verification/harsh-suites/selection.jsonl``:
1004 refusal rows over 218 families plus 330 harmless). Output is a small jsonl
whose row count per family is fixed by ``--per-family`` and whose selection is a
sorted-id stride -- no ``random`` state, so the same source digest, families and
counts always yield the same file and the same sha256.

The harmless (compliant) rows are the contrast side: real jailbreak deliveries on
this lineage are 3-5 rows in total, so a per-family delivery side cannot be
built, and the card says the delivery side is the compliant proxy. They are
taken with the same stride rule and carry the source's ``family_id`` so a reader
can see which families they came from; every one of them stays out of the
per-family training matrices (only the contrast table reads them).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Non-protective families with at least 32 refusal rows in the source. Self-harm and
# sexual-content (CP-adjacent) families stay out permanently (PLAN-NEXT 2026-11-18).
DEFAULT_FAMILIES = (
    'hb-std-sel-illegal',
    'hb-std-sel-cybercrime-intrusion',
    'sr-sel-non-violent-crimes',
    'sr-sel-violence',
)


def read_rows(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f'{path}:{number} is not JSON: {error}')
    return rows


def stride(rows: list[dict], count: int) -> list[dict]:
    """Up to ``count`` rows from ``rows`` (already id-sorted) at an even stride."""
    if len(rows) <= count:
        return rows
    step = len(rows) / count
    picked = [rows[int(i * step)] for i in range(count)]
    # A float stride can repeat an index when the step is just under 1; dedupe by id.
    seen, unique = set(), []
    for row in picked:
        if row['id'] not in seen:
            seen.add(row['id'])
            unique.append(row)
    return unique


def build(source: Path, families: list[str], per_family: int, harmless: int) -> dict:
    rows = read_rows(source)
    for row in rows:
        for field in ('id', 'kind', 'prompt'):
            if field not in row:
                raise SystemExit(f'{source}: row {row.get("id")} needs a {field!r} field')
    pool = {family: sorted((r for r in rows if r['kind'] == 'refusal' and r.get('family_id') == family),
                           key=lambda r: r['id']) for family in families}
    missing = [family for family, got in pool.items() if len(got) < per_family]
    if missing:
        raise SystemExit(f'{source}: families short of {per_family} refusal rows: '
                         f'{[(f, len(pool[f])) for f in missing]}')
    suite = []
    for family in families:
        suite.extend(stride(pool[family], per_family))
    good_pool = sorted((r for r in rows if r['kind'] == 'harmless'), key=lambda r: r['id'])
    good = stride(good_pool, harmless)
    for row in good:  # the contrast side is not a family; keep the field honest, not invented
        row.setdefault('family_id', None)
    suite.extend(good)
    return {
        'suite': suite,
        'meta': {
            'source': str(source),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'families': list(families),
            'per_family': per_family,
            'harmless': len(suite) - per_family * len(families),
            'rows': len(suite),
            'selection': 'sorted by id, even stride (no RNG)',
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Build the deterministic P5 SAE capture suite')
    parser.add_argument('--source', type=Path, default=Path('verification/harsh-suites/selection.jsonl'))
    parser.add_argument('--family', action='append', dest='families', default=None,
                        help=f'repeatable; default: {", ".join(DEFAULT_FAMILIES)}')
    parser.add_argument('--per-family', type=int, default=32)
    parser.add_argument('--harmless', type=int, default=40)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    built = build(args.source, args.families or list(DEFAULT_FAMILIES), args.per_family, args.harmless)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in built['suite']))
    meta = dict(built['meta'], suite=args.out.name,
                suite_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest())
    meta_path = args.out.with_suffix('.meta.json')
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    per_family = {}
    for row in built['suite']:
        if row['kind'] == 'refusal':
            per_family[row['family_id']] = per_family.get(row['family_id'], 0) + 1
    print(f'{args.out}: {built["meta"]["rows"]} rows, {len(per_family)} families, '
          f'{built["meta"]["harmless"]} harmless; sha256 {meta["suite_sha256"][:16]}')
    for family, count in per_family.items():
        print(f'  {count:3d} {family}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
