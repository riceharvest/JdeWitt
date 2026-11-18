#!/usr/bin/env python3
"""Offline sweep of cascade policy knobs (K, escalation cap) on a banked run.

Greedy continuation is prefix-deterministic: the recorded stream
(probe[+escalation]) contains the token sequence any smaller-K policy would
have observed as a prefix, so 'what would K' X, cap' Y have cost and pooled'
is reconstructable from ONE run with zero new inference.

Honesty constraints, printed in the report:
- token->char slicing at ~4 chars/token: boundary rows can flip; the sweep
  reproduces the recorded K within ~2.5% (check the mismatch count at the
  observed K before trusting deltas smaller than that);
- K larger than the observed probe can only be simulated from the
  escalation-contiguous rows (harsh-800 streams);
- policies that change WHICH rows escalate change what the agent-read pool
  costs - this sim screens token/pool size only, never certification.

Usage: python scripts/cascade_policy_sim.py <predictions.jsonl> [--k 100 125 150 200 250 300] [--cap 600] [--observed-k 150]
"""
import argparse
import json
from pathlib import Path

from jdewitt.refusal_shape import screen_short

CH_PER_TOK = 4  # crude but monotone; see docstring


def sim(rows, k, cap):
    n_r = n_d = n_e = tok = 0
    for r in rows:
        full = r.get('answer') or ''
        length = r.get('completion_tokens') or len(full) // CH_PER_TOK
        label = screen_short(full[:k * CH_PER_TOK], mode='screen')
        if label == 'refusal':
            n_r += 1
            tok += k
        elif label == 'delivery':
            n_d += 1
            tok += k
        else:
            n_e += 1
            tok += k + min(cap, max(0, length - k))
    return n_r, n_d, n_e, tok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('predictions', type=Path)
    ap.add_argument('--k', nargs='+', type=int, default=(100, 125, 150, 200, 250, 300))
    ap.add_argument('--cap', type=int, default=600)
    ap.add_argument('--observed-k', type=int, default=150)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args(argv)

    rows = [json.loads(x) for x in open(args.predictions) if x.strip()]
    table = []
    for k in args.k:
        n_r, n_d, n_e, tok = sim(rows, k, args.cap)
        row = {'k': k, 'cap': args.cap, 'settled_refusal': n_r, 'settled_delivery': n_d,
               'escalated': n_e, 'tokens': tok}
        if k == args.observed_k:
            mism = sum(1 for r in rows
                       if screen_short((r.get('answer') or '')[:k * CH_PER_TOK],
                                       mode='screen') != r.get('settled'))
            row['reconstruction_mismatches'] = mism
        table.append(row)
        print(json.dumps(row))
    report = {'predictions': str(args.predictions), 'rows': len(rows),
              'note': 'zero new inference; char-boundary noise ~2.5% at observed K',
              'table': table}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
        print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
