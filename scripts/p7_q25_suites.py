# SPDX-License-Identifier: AGPL-3.0-or-later
"""P7 q25 prep: stage the suites this lineage's cert waves run, with provenance.

Three screens, each already in the clone, copied under the names the registered
matrix floors and `release-certify` look for:

  harmful-40.jsonl    removal. Byte-identical to `verification/screens/advbench-40.jsonl`,
                      which is the screen every q3-4b harmful wave was taken against
                      (the q3 wave ids equal this file's ids exactly). It is the
                      in-sample screen for this lineage: the refit's fit rows come
                      from the same advbench families.
  holdout-24f.jsonl   the 24-family held-out screen (`verification/screens/q3-holdout-24.jsonl`),
                      24 rows over 24 distinct refusal families, zero id overlap with
                      any fit pool.
  held_out_120        NOT copied: the registered floor suite is
                      `verification/screens/q3-holdout-wider.jsonl` (120 rows, 120
                      families, disjoint from the 24 and from every fit pool). The
                      matrix refuses a row whose held-out suite name differs from the
                      registered one, so the spec must name that path.

Nothing here selects or authors rows; this only pins bytes and records their digests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SOURCES = {
    'harmful-40.jsonl': 'verification/screens/advbench-40.jsonl',
    'holdout-24f.jsonl': 'verification/screens/q3-holdout-24.jsonl',
    'fit-pool.jsonl': 'verification/p4d/suite-80.jsonl',
}

NOTES = {
    'harmful-40.jsonl': ('removal screen, 40 rows / 20 advbench families, English only; '
                         'in-sample for the refit (fit rows are advbench-family rows)'),
    'holdout-24f.jsonl': ('24 held-out refusal families, one row per family, zero id/prompt '
                          'overlap with the fit pool (checked at staging time)'),
    'fit-pool.jsonl': ('refit capture pool: 40 refusal + 40 harmless rows, 29 refusal '
                       'families, stratified from the in-clone harsh-suite selection; '
                       'zero id/prompt overlap with either held-out screen'),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(prog='p7_q25_suites')
    parser.add_argument('--out', default='verification/p7-q25-15b/suites')
    args = parser.parse_args(argv)
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    manifest = {'kind': 'p7-q25-suite-staging', 'suites': {}}
    for name, source in SOURCES.items():
        source_path = REPO / source
        payload = source_path.read_bytes()
        (target / name).write_bytes(payload)
        rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        manifest['suites'][name] = {
            'source': source, 'source_sha256': sha256(source_path),
            'staged_sha256': sha256(target / name), 'rows': len(rows),
            'families': len({row.get('family_id') for row in rows}),
            'kinds': sorted({row.get('kind') for row in rows}), 'note': NOTES[name],
        }
    manifest['held_out_floor_suite'] = {
        'path': 'verification/screens/q3-holdout-wider.jsonl',
        'rows': 120, 'families': 120,
        'note': ('registered held-out floor suite; not copied because release-matrix matches '
                 'the registered suite by file name'),
    }
    (target / 'staging.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
