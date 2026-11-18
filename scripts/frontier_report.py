# SPDX-License-Identifier: AGPL-3.0-or-later
"""One row per edit: removal and retention read together, no ad-hoc analysis.

Every qwen3-4B sweep tonight got read by hand with a throwaway script, which is
how a number ends up quoted without its p-value, its truncation count or its
median answer length. This is the standing instrument: for each artifact it pulls
removal (and invalid rows) out of a harmful screen, and maths/code deltas with
paired sign-flip p-values and the length-collapse flag out of the retention
waves, using the shipped scorers and `capability_format_audit`'s functions — never
regrading anything.

    python scripts/frontier_report.py \
      --base-dir verification/q3-4b-retention/q3base \
      --pair both10=verification/q3-4b-harmful40/q3shl10:verification/q3-4b-retention/q3shl10 \
      --pair attn20=verification/q3-4b-harmful40/attn-l935:verification/q3-4b-retention/attn-l935

A row whose screen came from a different serve config than the rest is not silently
comparable; this script prints the recorded max_tokens/parallel per row and leaves
judging that to you.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

from capability_format_audit import audit, paired  # noqa: E402
from jdewitt.stats import wilson_interval  # noqa: E402

SUITE = REPO / 'verification/capability/capability-run-en.jsonl'


def removal(screen: Path) -> dict:
    rows = [json.loads(line) for line in screen.read_text().splitlines() if line.strip()]
    invalid = sum(1 for r in rows if r.get('status') != 'ok')
    if rows and invalid == len(rows):
        # Nothing generated. An empty answer trips no refusal wording, so counting
        # refusals would report a perfect removal. Say what actually happened.
        lengths = [len(r.get('answer') or '') for r in rows]
        return {'rows': len(rows), 'removed': None, 'refused': None, 'invalid': invalid,
                'removed_rate': None, 'wilson_lower': None,
                'median_chars': int(statistics.median(lengths)) if lengths else None,
                'note': 'UNMEASURED — every row invalid, the screen never happened'}
    refused = sum(1 for r in rows
                  if (r.get('refusal_check') or {}).get('result') == 'refusal_detected')
    lengths = [len(r.get('answer') or '') for r in rows]
    interval = wilson_interval(len(rows) - refused, len(rows))
    return {'rows': len(rows), 'removed': len(rows) - refused, 'refused': refused,
            'invalid': invalid,
            'removed_rate': round((len(rows) - refused) / len(rows), 4) if rows else None,
            'wilson_lower': round(interval['lower'], 4) if rows else None,
            'median_chars': int(statistics.median(lengths)) if lengths else None}


def wave(dir_path: Path) -> dict:
    manifest = dir_path / 'wave.json'
    data = json.loads(manifest.read_text()) if manifest.exists() else {}
    serve = data.get('serve_structured') or {}
    return {'max_tokens': serve.get('max_tokens'), 'parallel': serve.get('parallel'),
            'seconds': data.get('seconds'), 'invalid': data.get('invalid')}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base-dir', type=Path, required=True,
                    help='retention wave directory for the unedited base')
    ap.add_argument('--pair', action='append', required=True, metavar='LABEL=HARMFUL:RETENTION')
    ap.add_argument('--suite', type=Path, default=SUITE)
    args = ap.parse_args(argv)

    labels, rows = [args.base_dir.name], []
    base_ret = audit('base', args.base_dir / 'predictions.jsonl', args.suite)
    for item in args.pair:
        label, paths = item.split('=', 1)
        harmful, retention = paths.split(':')
        harmful_dir, retention_dir = Path(harmful), Path(retention)
        labels.append(label)
        rem = removal(harmful_dir / 'predictions.jsonl')
        edit_ret = audit(label, retention_dir / 'predictions.jsonl', args.suite)
        kinds = {}
        for kind in ('math', 'code'):
            stats = paired(base_ret, edit_ret, kind).get('strict') or {}
            length = paired(base_ret, edit_ret, kind).get('length') or {}
            kinds[kind] = {'before': stats.get('before_pass'),
                           'after': stats.get('after_pass'),
                           'delta': stats.get('delta'),
                           'p': stats.get('paired_sign_flip_p_two_sided'),
                           'collapse': length.get('collapse_flag'),
                           'median': length.get('after_median_chars')}
        rows.append({'label': label, 'removal': rem, 'retention_wave': wave(retention_dir),
                     'kinds': kinds})

    header = (f"| {'artifact':12} | removal | refused | invalid | median | "
              f"maths | p | math med | code | p | cap/slots |")
    print(header)
    print('|' + '-' * len(header) + '|')
    for row in rows:
        r, k = row['removal'], row['kinds']
        w = row['retention_wave']
        cap = f"{w['max_tokens']}/{w['parallel']}" if w['max_tokens'] else '—'
        if r['removed'] is None:
            removal_cell = f"UNMEASURED ({r['invalid']}/{r['rows']} invalid)"
            refused_cell = '—'
        else:
            removal_cell = f"{r['removed']}/{r['rows']} (≥{r['wilson_lower']})"
            refused_cell = r['refused']
        print(f"| {row['label']:12} | {removal_cell} "
              f"| {refused_cell} | {r['invalid']} | {r['median_chars']} "
              f"| {k['math']['before']}→{k['math']['after']} ({k['math']['delta']}) "
              f"| {k['math']['p']} | {k['math']['median']}{'*' if k['math']['collapse'] else ''} "
              f"| {k['code']['before']}→{k['code']['after']} ({k['code']['delta']}) "
              f"| {k['code']['p']} | {cap} |")
    print('\n* length-collapse flag: the median answer fell below 25% of base — that '
          'is the model stopped writing, not a wrong answer.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
