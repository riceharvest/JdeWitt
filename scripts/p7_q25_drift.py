# SPDX-License-Identifier: AGPL-3.0-or-later
"""P7 q25: control-corpus drift for the REFIT arm, prefill only.

`release-certify` reads a drift input with ``measured``, ``control_delta_ln_ppl`` and
``budget``. This measures it: one llama-perplexity pass over the banked control
corpus for the base artifact and the arm, and the delta of the log perplexities.
Prefill only -- no tokens are generated, nothing is served, and the reading answers
exactly one question: did the whole distribution move, so that a branch reading
cannot be attributed to the branch?

    python scripts/p7_q25_drift.py \
        --base verification/p7-q25-15b/artifacts/base-q8.gguf \
        --arm  verification/p7-q25-15b/artifacts/arm-refit-q8.gguf \
        --out  verification/p7-q25-15b/drift.json [--dry]

`--dry` prints the two commands and exits without running them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_CORPUS = REPO / 'verification/ppl-probe/control.txt'


def main(argv=None):
    parser = argparse.ArgumentParser(prog='p7_q25_drift')
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--arm', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budget', type=float, default=0.02,
                        help='registered drift budget in ln-PPL (matrix.manifest registered.gates)')
    parser.add_argument('--ctx', type=int, default=512)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--ngl', type=int, default=0, help='GPU layers; 0 = CPU (this card holds no GPU)')
    parser.add_argument('--binary', default=None, help='llama-perplexity; default jdewitt.ppl.DEFAULT_BINARY')
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args(argv)

    from jdewitt.ppl import DEFAULT_BINARY, ln, perplexity

    binary = args.binary or DEFAULT_BINARY
    for path in (args.base, args.arm, args.corpus):
        if not path.is_file():
            raise SystemExit(f'missing operand: {path}')
    if not Path(binary).exists():
        raise SystemExit(f'perplexity binary not found: {binary}')
    commands = {label: [binary, '-m', str(path), '-f', str(args.corpus),
                        '-ngl', str(args.ngl), '-c', str(args.ctx), '-t', str(args.threads)]
                for label, path in (('base', args.base), ('arm', args.arm))}
    if args.dry:
        print(json.dumps({'dry': True, 'commands': {k: ' '.join(v) for k, v in commands.items()},
                          'budget': args.budget, 'corpus': str(args.corpus)}, indent=1))
        return 0

    readings = {}
    for label, path in (('base', args.base), ('arm', args.arm)):
        reading = perplexity(binary, str(path), str(args.corpus), ctx=args.ctx,
                             threads=args.threads, ngl=args.ngl)
        if 'error' in reading:
            raise SystemExit(f'{label}: llama-perplexity produced no PPL line: {reading}')
        readings[label] = {**reading, 'ln_ppl': ln(reading['ppl'])}

    delta = readings['arm']['ln_ppl'] - readings['base']['ln_ppl']
    payload = {
        'schema_version': 1,
        'lineage': 'qwen25-1.5b',
        'label': 'refit-sas15',
        'measured': True,
        'instrument': f'llama-perplexity, ctx {args.ctx}, {args.threads} threads, ngl {args.ngl}',
        'control_delta_ln_ppl': round(delta, 6),
        'budget': args.budget,
        'inside_budget': abs(delta) <= args.budget,
        'corpus': str(args.corpus),
        'readings': readings,
        'source': str(args.out),
        'note': ('the control corpus drift; a branch (BR) move is not generic drift, and a drift '
                 'outside the budget is not attributable to the edit'),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + '\n')
    print(json.dumps({'control_delta_ln_ppl': payload['control_delta_ln_ppl'],
                      'budget': args.budget, 'inside_budget': payload['inside_budget']}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
