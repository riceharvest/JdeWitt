#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Paired retention between two saved waves of the same authored suite.

The retention axis is paired: both arms decode the same 241 rows, so the only rows
that carry information are the discordant ones. This reads two wave directories
(predictions.jsonl + wave.json, as scripts/capability_wave.py writes them), refuses
a pair whose serve_structured differs in anything that can move a greedy row, and
reports per arm the pass count, invalid count and median answer length, plus the
paired lost/gained pairs and the exact McNemar p.

    python scripts/vocab_weight_arm.py retention \
        --a verification/vocab-remap/weight-arm/waves/base/retention \
        --b verification/vocab-remap/weight-arm/waves/edited/retention \
        --out verification/vocab-remap/weight-arm/retention.json

A row passes only when the shipped scorer said status=ok and passed=true; a
truncated or empty answer is never a pass, and it is counted, not dropped.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.paired_screen_test import exact_mcnemar  # noqa: E402

SCHEMA = 'jdewitt.vocab-weight-arm-retention/1'
# Fields that can change a greedy row's answer between two waves (METHODOLOGY R9).
CONFIG_FIELDS = ('ctx', 'parallel', 'max_tokens', 'temperature', 'jinja', 'cuda',
                 'no_think', 'suite')


def load(directory: Path) -> tuple[dict, dict[str, dict]]:
    wave_path = directory / 'wave.json'
    rows_path = directory / 'predictions.jsonl'
    for path in (wave_path, rows_path):
        if not path.is_file():
            raise SystemExit(f'{path} does not exist — pass a wave directory')
    wave = json.loads(wave_path.read_text())
    rows = {}
    for line in rows_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row['id']] = row
    return wave, rows


def arm_stats(rows: dict[str, dict]) -> dict:
    ok = {i: r for i, r in rows.items() if r.get('status') == 'ok'}
    passed = {i for i, r in ok.items() if r.get('passed') is True}
    lengths = [len((r.get('answer') or '')) for r in ok.values()]
    by_kind: dict[str, dict] = {}
    for row in rows.values():
        entry = by_kind.setdefault(row.get('kind', '?'), {'rows': 0, 'passed': 0,
                                                          'invalid': 0, 'chars': []})
        entry['rows'] += 1
        if row.get('status') == 'ok':
            entry['passed'] += int(row.get('passed') is True)
            entry['chars'].append(len(row.get('answer') or ''))
        else:
            entry['invalid'] += 1
    for entry in by_kind.values():
        entry['median_answer_chars'] = statistics.median(entry.pop('chars')) \
            if entry['chars'] else None
    return {'rows': len(rows), 'valid': len(ok), 'invalid': len(rows) - len(ok),
            'passed': len(passed), 'median_answer_chars': statistics.median(lengths)
            if lengths else None, 'by_kind': by_kind}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--a', type=Path, required=True, help='the control arm (pristine artifact)')
    ap.add_argument('--b', type=Path, required=True, help='the edited arm')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--allow-config-mismatch', action='store_true')
    args = ap.parse_args(argv)

    wave_a, rows_a = load(args.a)
    wave_b, rows_b = load(args.b)
    cfg_a = wave_a.get('serve_structured') or {}
    cfg_b = wave_b.get('serve_structured') or {}
    mismatch = {field: (cfg_a.get(field), cfg_b.get(field)) for field in CONFIG_FIELDS
                if cfg_a.get(field) != cfg_b.get(field)}
    if mismatch and not args.allow_config_mismatch:
        raise SystemExit('the two waves were not taken at the same config (R9): '
                         + '; '.join(f'{k} {v[0]!r} vs {v[1]!r}' for k, v in mismatch.items())
                         + ' — pass --allow-config-mismatch to compare anyway')

    stats_a, stats_b = arm_stats(rows_a), arm_stats(rows_b)
    shared = sorted(set(rows_a) & set(rows_b))
    valid = [i for i in shared
             if rows_a[i].get('status') == 'ok' and rows_b[i].get('status') == 'ok']
    pass_a = {i: rows_a[i].get('passed') is True for i in shared}
    pass_b = {i: rows_b[i].get('passed') is True for i in shared}
    lost = sorted(i for i in valid if pass_a[i] and not pass_b[i])
    gained = sorted(i for i in valid if pass_b[i] and not pass_a[i])
    mcnemar = exact_mcnemar({i: pass_a[i] for i in valid}, {i: pass_b[i] for i in valid})
    payload = {
        'schema': SCHEMA,
        'suite': cfg_a.get('suite'),
        'scope': 'paired retention of one authored suite between the pristine artifact and one '
                 'weight-side row edit; pass counts and answer lengths come from the shipped '
                 'scorer in each wave, the pairing is by row id',
        'arms': {'a': {'directory': str(args.a), 'artifact': wave_a.get('serve_config'),
                       **stats_a},
                 'b': {'directory': str(args.b), 'artifact': wave_b.get('serve_config'),
                       **stats_b}},
        'config': {field: cfg_a.get(field) for field in CONFIG_FIELDS},
        'config_mismatch': mismatch or None,
        'paired': {'shared_rows': len(shared), 'valid_both': len(valid),
                   'lost': len(lost), 'gained': len(gained),
                   'lost_ids': lost, 'gained_ids': gained,
                   'exact_mcnemar': mcnemar,
                   'median_answer_chars': {'a': stats_a['median_answer_chars'],
                                           'b': stats_b['median_answer_chars']}},
        'note': 'A truncated or empty answer is not a pass; it is counted in invalid. '
                'McNemar is exact and two-sided over the valid pairs.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: payload[k] for k in ('arms', 'paired', 'config_mismatch')},
                     indent=1)[:4000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
