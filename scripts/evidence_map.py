#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render and enforce the README's evidence map.

A README bullet and a verification file drift apart silently, and the drift is the
whole failure mode of this project: the number stays in prose after the artifact
supporting it is gone. This tool makes that drift an exit code.

Usage:
    python scripts/evidence_map.py            # print the markdown table
    python scripts/evidence_map.py --check    # exit 1 if any 'in-clone' artifact is missing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / 'verification/evidence-map.json'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true',
                    help='fail if any artifact an in-clone claim names is missing')
    args = ap.parse_args(argv)

    if not MAP.is_file():
        raise SystemExit(f'{MAP} does not exist')
    doc = json.loads(MAP.read_text())
    claims = doc['claims']

    missing, broken_status = [], []
    for claim in claims:
        for artifact in claim['artifacts']:
            if not (ROOT / artifact).is_file():
                missing.append((claim['status'], claim['claim'], artifact))
        if claim['status'] not in {'in-clone', 'pre-2026-11-17-scoring-fix', 'no-artifact-in-clone'}:
            broken_status.append((claim['claim'], claim['status']))

    print('# Evidence map\n')
    print(doc['purpose'] + '\n')
    print('| status | claim | artifacts named | command |')
    print('|' + '-' * 104 + '|')
    for claim in claims:
        arts = ', '.join(f'`{a}`' for a in claim['artifacts']) or '*(none)*'
        cmd = claim.get('commands', [''])[0]
        cmd = f'`{cmd}`' if cmd else ''
        print(f"| {claim['status']} | {claim['claim']} | {arts} | {cmd} |")
        if claim.get('notes'):
            print(f"| | ↳ {claim['notes']} | | |")
    print(f"\n{len(claims)} claims: "
          f"{sum(1 for c in claims if c['status'] == 'in-clone')} in-clone, "
          f"{sum(1 for c in claims if c['status'] == 'pre-2026-11-17-scoring-fix')} pre-fix, "
          f"{sum(1 for c in claims if c['status'] == 'no-artifact-in-clone')} with no artifact.")

    if args.check:
        if broken_status:
            for claim, status in broken_status:
                print(f'STATUS NOT IN THE ALLOWED SET [{status}]: {claim}')
        if missing:
            for status, claim, artifact in missing:
                print(f'MISSING [{status}]: {artifact}  ← {claim}')
            print(f'\n{len(missing)} artifact(s) named by a claim that do not exist in the tree.')
            return 1
        print('\nAll artifacts named by the map are present.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
