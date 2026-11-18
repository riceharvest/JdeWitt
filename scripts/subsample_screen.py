# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministically subsample a family-intact harmful screen from an imported suite.

Verification power scales with families, and the banked harsh-suites holdout
carries 863 pinned, near-duplicate-dropped refusal rows across 183 categories-
anchored families — enough to take an honest 800-row screen without sourcing
anything new. Families are taken in seeded-hash order (whole families only,
except the final family may contribute a partial tail so the screen lands on
exactly N rows), so the split is reproducible from the recorded seed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from jdewitt.data import digest
from jdewitt.answers import atomic_text


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    ap.add_argument('source', type=Path, help='imported suite .jsonl')
    ap.add_argument('out', type=Path)
    ap.add_argument('--rows', type=int, default=800)
    ap.add_argument('--kind', default='refusal')
    ap.add_argument('--seed', default='harmful-800-2026-11-18')
    ap.add_argument('--provenance', type=Path)
    args = ap.parse_args(argv)

    rows = [json.loads(x) for x in open(args.source) if x.strip()]
    rows = [r for r in rows if r.get('kind') == args.kind]
    by_fam: dict[str, list] = {}
    for r in rows:
        by_fam.setdefault(r['family_id'], []).append(r)
    order = sorted(by_fam, key=lambda f: digest([args.seed, f]))
    taken, fams = [], []
    for f in order:
        fr = sorted(by_fam[f], key=lambda r: r['id'])
        if len(taken) >= args.rows:
            break
        room = args.rows - len(taken)
        fams.append(f)
        taken.extend(fr[:room])
    cats = Counter(r.get('category') or 'uncategorised' for r in taken)
    out_rows = sorted(taken, key=lambda r: r['id'])
    atomic_text(args.out, '\n'.join(json.dumps(r, sort_keys=True) for r in out_rows) + '\n')
    prov = {
        'scope': 'seeded family-intact subsample of an imported, pinned suite; no new data sourced',
        'source': str(args.source), 'seed': args.seed, 'kind': args.kind,
        'rows_source_refusals': len(rows), 'rows_taken': len(out_rows),
        'families': len(fams), 'categories': dict(cats),
        'rule': 'whole families in seeded-hash order; final family may be partial',
    }
    if args.provenance:
        atomic_text(args.provenance, json.dumps(prov, indent=1, sort_keys=True) + '\n')
    print(json.dumps(prov, indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
