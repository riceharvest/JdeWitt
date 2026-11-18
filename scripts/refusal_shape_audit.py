# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit refusal *shape* over saved predictions, so a style shift cannot read as removal.

Usage: python scripts/refusal_shape_audit.py LABEL=predictions.jsonl [LABEL2=...]

The shipped refusal-check looks for refusal wording. On qwen3-4B that made a
published competitor's build and our own edits look like perfect removal at a
10-row screen, because both drop "I'm sorry, but I can't assist" and replace it
with a lecture plus a pivot to legal/ethical alternatives — the refusal content
survives, the phrasing does not. This tool reports what is actually on the page:
whether a canonical opener is present, whether the answer still declines the asked
harm in substance, whether it reframes, whether it contains procedural content, and
how long it is. It never claims to know whether the asked-for harm was delivered —
that read stays human (METHODOLOGY R6) — but it makes a style shift visible as a
style shift instead of letting it pass as removal, and it counts exactly how many
rows the wording checker calls "no explicit refusal" while the shape still declines.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jdewitt.refusal_shape import classify  # noqa: E402


def audit(label: str, path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    shapes: dict[str, int] = {}
    lengths, flagged = [], []
    for row in rows:
        answer = row.get('answer') or ''
        shape = classify(answer)
        shapes[shape['shape']] = shapes.get(shape['shape'], 0) + 1
        lengths.append(shape['chars'])
        checker = (row.get('refusal_check') or {}).get('result')
        if checker == 'no_explicit_refusal' and shape['decline']:
            flagged.append(row['id'])
    return {'label': label, 'predictions': str(path), 'rows': len(rows),
            'shapes': shapes,
            'median_chars': round(statistics.median(lengths)) if lengths else 0,
            'procedural_rows': sum(1 for r in rows if classify(r.get('answer') or '')['procedural']),
            'checker_says_no_refusal_but_shape_declines': len(flagged),
            'example_flagged_rows': flagged[:6],
            'note': 'shape labels are heuristics over the answer text; the semantic '
                    'question — is the asked-for harm actually delivered — is a human '
                    'read (METHODOLOGY R6)'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('pairs', nargs='+', metavar='LABEL=PREDICTIONS.jsonl')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    audits = [audit(item.split('=', 1)[0], Path(item.split('=', 1)[1]))
              for item in args.pairs]
    text = json.dumps({'audits': audits}, indent=1)
    print(text)
    if args.out:
        from jdewitt.answers import atomic_text
        atomic_text(args.out, text + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
