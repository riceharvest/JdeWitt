# SPDX-License-Identifier: AGPL-3.0-or-later
"""Merge cascade chunks of one artifact into one evidence dir and sum the cost.

Chunking is a harness timeout artifact, not a measurement split: all chunks ran
with byte-identical flags and one serve config, so merging is lossless. Refuses
to merge chunks with differing k/max/serve-config, because a merged record that
silently mixes configs is exactly the R9 bug this project keeps paying for.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from jdewitt.refusal_shape import classify
from jdewitt.answers import atomic_text


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    ap.add_argument('out', type=Path)
    ap.add_argument('chunk', nargs='+', type=Path)
    ap.add_argument('--label', default='cascade-800')
    args = ap.parse_args(argv)

    records, summaries = [], []
    key = None
    for d in args.chunk:
        pred = d / 'predictions.jsonl'
        if not pred.exists():
            raise SystemExit(f'{d}: no predictions.jsonl (chunk did not finish)')
        s = json.loads((d / 'cascade.json').read_text())
        k = (s['k_tokens'], s['max_tokens'], s['settle_mode'], s['serve_config'])
        if key is None:
            key = k
        elif k != key:
            raise SystemExit(f'{d}: differs in k/max/settle/serve — refusing to merge mixed configs')
        summaries.append({'dir': str(d), 'rows': s['rows'],
                          'tokens_total': s.get('tokens_total'),
                          'tokens_stage_a': s.get('tokens_stage_a'),
                          'seconds': s.get('seconds')})
        records += [json.loads(x) for x in open(pred) if x.strip()]
    ids = [r['id'] for r in records]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise SystemExit(f'duplicate ids across chunks: {dupes[:6]}')
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_text(args.out / 'predictions.jsonl',
                ''.join(json.dumps(r, sort_keys=True) + '\n' for r in records))
    tallies = Counter(r['settled'] for r in records)
    flip = Counter()
    for r in records:
        if r['settled'] == 'escalate' and r.get('answer'):
            f = classify(r['answer'])
            flip['refusal' if f['declining'] else ('delivery' if f['procedural'] else 'other')] += 1
    fam_settle = Counter(r['family_id'] for r in records if r['settled'] != 'refusal')
    fam_all = Counter(r['family_id'] for r in records)
    summary = {
        'label': args.label, 'rows': len(records), 'families': len(fam_all),
        'settled_refusal': tallies['refusal'], 'settled_delivery': tallies['delivery'],
        'escalated': tallies['escalate'], 'escalation_full_shape': dict(flip),
        'tokens_stage_a': sum(s.get('tokens_stage_a') or 0 for s in summaries),
        'tokens_total': sum(s.get('tokens_total') or 0 for s in summaries),
        'seconds': sum(s.get('seconds') or 0 for s in summaries),
        'chunks': summaries,
        'note': 'settled_delivery and escalated rows remain subject to the agent read (R2); '
                'the 600-token escalation cap means a procedure onsetting beyond token 600 is '
                'not visible in this record — escalate-to-full is the fallback if the read asks.',
        'families_with_nonrefusal_row': dict(fam_settle),
    }
    atomic_text(args.out / 'cascade.json', json.dumps(summary, indent=1, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
