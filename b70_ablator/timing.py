"""Phase timing ledger for the refusal-removal search loop (lane 6).

Every phase of one candidate loop (fit, merge, serve_swap, wave, capture,
baseline, judge) appends an event to a JSONL ledger. `summarize` prints the
percentage breakdown and pivot hints from measured thresholds, so "where does
the loop time go" is a tool answer, and the decision to change strategy (e.g.
screening subsets, short decode, lora swaps) fires off numbers, not vibes.

Event shape:
  {"ts", "lineage", "phase", "seconds", "kind": "cpu|gpu|wall|proxy",
   "scope": {...}, "source": "measured|backfill-est"}
"""
from __future__ import annotations

import argparse
import json
import os
import time

VALID_KINDS = ('cpu', 'gpu', 'wall', 'proxy')
# pivot thresholds: phase fraction of total loop time that should trigger a
# strategy change; documented in docs/non-heretic-methods.md lane 6.
PIVOTS = {
    'wave': (0.60, 'candidate waves dominate: cut to screening subset (10 rows) and short decode (~150 tok)'),
    'serve_swap': (0.25, 'model reload dominates: lora-swap screening on honest-curve lineages, or parallel screening pool'),
    'baseline': (0.35, 'baseline/capture gen dominates: reuse baseline text as teacher forcing for capture (already the rule; verify no regen)'),
    'judge': (0.20, 'reading cost high: screen with subsets, full-40 only for finalists'),
}


def record(path: str, *, lineage: str, phase: str, seconds: float,
           kind: str = 'wall', scope=None, source: str = 'measured'):
    if seconds < 0 or kind not in VALID_KINDS:
        raise ValueError(f'bad event phase={phase} kind={kind} seconds={seconds}')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps({'ts': time.time(), 'lineage': lineage,
                            'phase': phase, 'seconds': float(seconds),
                            'kind': kind, 'scope': scope or {},
                            'source': source}) + '\n')


def load(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summarize(events: list[dict], *, by_lineage: bool = False) -> dict:
    """Phase percentage breakdown, wall-relevant kinds only + proxy counted
    separately (judge is read-cost proxy, not a GPU second)."""
    gpu_phases = ('wave', 'serve_swap', 'capture', 'baseline')
    groups = {}
    for e in events:
        key = (e['lineage'], e['kind']) if by_lineage else e['kind']
        groups.setdefault(key, {'total': 0.0, 'phases': {}})
        g = groups[key]
        if e['phase'] == 'judge':
            g['proxy_seconds'] = g.get('proxy_seconds', 0.0) + e['seconds']
        else:
            g['total'] += e['seconds']
            g['phases'].setdefault(e['phase'], 0.0)
            g['phases'][e['phase']] += e['seconds']
    report = {}
    for key, g in groups.items():
        tot = g['total']
        row = {'total_s': round(tot, 1)}
        hints = []
        for ph, secs in sorted(g['phases'].items(), key=lambda kv: -kv[1]):
            pct = 100.0 * secs / tot if tot else 0.0
            row[ph] = round(pct, 1)
            thr, hint = PIVOTS.get(ph, (None, None))
            if thr and pct >= 100.0 * thr:
                hints.append(f'{ph} {pct:.0f}% >= {thr*100:.0f}%: {hint}')
        if g.get('proxy_seconds'):
            row['judge_proxy_s'] = round(g['proxy_seconds'], 1)
        row['pivot_hints'] = hints
        if by_lineage:
            report[f'{key[0]}|{key[1]}'] = row
        else:
            report[key] = row
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(prog='b70_ablator.timing')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('summarize')
    s.add_argument('--ledger', required=True)
    s.add_argument('--lineages', action='store_true')
    l = sub.add_parser('log')
    l.add_argument('--ledger', required=True)
    l.add_argument('--lineage', required=True)
    l.add_argument('--phase', required=True)
    l.add_argument('--seconds', required=True, type=float)
    l.add_argument('--kind', default='wall', choices=VALID_KINDS)
    l.add_argument('--source', default='measured')
    args = ap.parse_args(argv)
    if args.cmd == 'log':
        record(args.ledger, lineage=args.lineage, phase=args.phase,
               seconds=args.seconds, kind=args.kind, source=args.source)
        print('logged')
        return
    print(json.dumps(summarize(load(args.ledger), by_lineage=args.lineages),
                      indent=1))


if __name__ == '__main__':
    main()
