# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sum the actual generated tokens spent verifying one artifact — the metric every cost reduction is measured against.

Usage: python scripts/cert_cost.py WAVE_DIR... [--json OUT]

Each WAVE_DIR holds a wave.json (capability_wave) or cascade.json
(cascade_verify). The tool adds up completion tokens per wave and the total
per artifact, because 'how much inference does a removal claim cost' must be a
number read from evidence, never an impression from the number of waves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def cost_of(d: Path) -> dict | None:
    for name in ('wave.json', 'cascade.json'):
        p = d / name
        if p.exists():
            j = json.loads(p.read_text())
            kind = 'cascade' if name == 'cascade.json' else 'wave'
            return {'dir': str(d), 'kind': kind,
                    'label': j.get('label', d.name),
                    'suite': j.get('suite'),
                    'rows': j.get('rows'),
                    'tokens': j.get('completion_tokens_total', j.get('tokens_total', 0)) or 0,
                    'seconds': j.get('seconds'),
                    'measured': j.get('measured', True),
                    'settled_refusal': j.get('settled_refusal'),
                    'settled_delivery': j.get('settled_delivery'),
                    'escalated': j.get('escalated')}
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('dirs', nargs='+', type=Path)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args(argv)
    waves, missing = [], []
    for d in args.dirs:
        c = cost_of(d)
        (waves if c else missing).append(c if c else str(d))
    total = sum(w['tokens'] for w in waves)
    unmeasured = [w['label'] for w in waves if not w['measured']]
    report = {'scope': 'Generated-token accounting over saved waves/cascades; '
                       'no inference performed. Only waves with measured=true count as '
                       'verification evidence.',
              'waves': waves, 'waves_total': len(waves),
              'total_completion_tokens': total,
              'unmeasured_waves': unmeasured,
              'dirs_without_wave_record': missing,
              'note': 'baseline waves (base model) are one-time per base; candidate waves are '
                      'the per-arm certification cost. Cascade waves record partial decodes: '
                      'settled rows never generated their full answer.'}
    text = json.dumps(report, indent=1, sort_keys=True)
    print(text)
    if args.json:
        from jdewitt.answers import atomic_text
        atomic_text(args.json, text + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
