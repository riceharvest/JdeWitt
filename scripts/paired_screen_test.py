#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact McNemar test between two refusal screens over the same prompts.

Two screens run against the same prompt ids are paired, not independent, so comparing
their Wilson intervals is not the comparison anyone asked for. The question is "does
this artifact refuse on any family the other one does not, and vice versa", which is
exactly McNemar: only the discordant pairs carry information. Reported exactly, because
the screens are 24-40 rows and a normal approximation is the kind of rounding that turns
n=2 into a conclusion.

Usage:
    python scripts/paired_screen_test.py SCREEN_A/predictions.jsonl SCREEN_B/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def refusals(path: Path) -> dict[str, bool]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('status') != 'ok':
            continue  # a row that never generated is not evidence either way
        out[row['id']] = (row.get('refusal_check') or {}).get('result') == 'refusal_detected'
    return out


def exact_mcnemar(a: dict[str, bool], b: dict[str, bool]) -> dict:
    ids = sorted(set(a) & set(b))
    n01 = sum(1 for i in ids if a[i] and not b[i])
    n10 = sum(1 for i in ids if b[i] and not a[i])
    m = n01 + n10
    if m == 0:
        p = 1.0
    else:
        k = min(n01, n10)
        p = min(1.0, 2 * sum(comb(m, j) for j in range(k + 1)) / 2 ** m)
    return {'paired': len(ids), 'refusals_a': sum(a.values()),
            'refusals_b': sum(b.values()),
            'refused_by_a_only': n01, 'refused_by_b_only': n10,
            'discordant': m, 'exact_mcnemar_p_two_sided': round(p, 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('screen_a', type=Path)
    ap.add_argument('screen_b', type=Path)
    args = ap.parse_args(argv)

    a = refusals(args.screen_a)
    b = refusals(args.screen_b)
    if not (set(a) & set(b)):
        raise SystemExit('no prompt id in common — these are not paired screens, '
                         'so there is nothing McNemar can say about them')
    print(json.dumps(exact_mcnemar(a, b), indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
