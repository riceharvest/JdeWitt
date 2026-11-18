# SPDX-License-Identifier: AGPL-3.0-or-later
"""Zero-generation refusal screen: which branch does an artifact predict?

Runs llama-perplexity (CPU build, prefill only — zero generated tokens) of an
artifact over the ppl-probe corpora (see build_ppl_probe.py):

  BR = ln PPL(refusal corpus) - ln PPL(delivery corpus)

A model that still refuses predicts its own refusal continuation clearly
better than delivery text on harmful prompts; a real removal erases that
advantage. Sign as measured on q3-4B (base reference): delta_BR > 0 means the
refusal branch got specifically worse — the oracle (whose deliveries were
agent-read as real at the known ceiling) measures +0.56 on ln-PPL units with
-0.003 control drift, while f15s15 (opener-suppressed, 0/24 delivered by R2
read) measures -0.05 +/- 0.13 — its tone moved 40x but its branch preference
did not move at all. The control corpus (benign passed maths) measures
generic drift from quantization/tone so a BR change can be attributed to the
branch, not to the whole distribution. Deltas are taken against the first
artifact, the base, under one fixed pinned runner config (R9 applied to the
probe too).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

PPL_RE = re.compile(r'PPL = ([0-9.]+)(?: \+/- ([0-9.]+))?')


def ppl(binary: str, model: Path, corpus: Path, ctx: int, threads: int) -> dict:
    t0 = time.monotonic()
    p = subprocess.run(
        [binary, '-m', str(model), '-f', str(corpus), '-ngl', '0',
         '-c', str(ctx), '-t', str(threads)],
        capture_output=True, text=True, timeout=1800)
    out = p.stdout + p.stderr
    m = PPL_RE.search(out)
    if not m:
        return {'error': 'no PPL line', 'tail': out[-500:]}
    return {'ppl': float(m.group(1)),
            'uncertainty': float(m.group(2)) if m.group(2) else None,
            'seconds': round(time.monotonic() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    ap.add_argument('--binary', default='/mnt/ssd/dev workspace/llama.cpp-dspark/build-perp/bin/llama-perplexity')
    ap.add_argument('--corpus-dir', default='verification/ppl-probe')
    ap.add_argument('--artifact', action='append', required=True, metavar='NAME=PATH',
                    help='repeat; first one is the reference (base)')
    ap.add_argument('--ctx', type=int, default=512)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--no-control', action='store_true',
                    help='skip the benign control corpus (control drift is already '
                         'measured at the endpoints of a ladder; use only for screening)')
    ap.add_argument('--corpus', action='append', metavar='NAME=PATH',
                    help='override a corpus file, e.g. --corpus refusal=verification/'
                         'ppl-probe/refusal800-250.txt')
    ap.add_argument('--one-sided', action='store_true',
                    help='DEPRECATED by measurement 2026-11-18: on a refusal-only corpus '
                         '(refusal800) dR ranks f15s15-l935 (+0.50) ABOVE oracle (+0.22) — '
                         'it is a style-distance meter, not a branch meter; two-sided BR '
                         'on the hand-read D corpus is the validated screen')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args(argv)

    corpus_dir = Path(args.corpus_dir)
    corpora = {'refusal': corpus_dir / 'refusal.txt',
               'delivery': corpus_dir / 'delivery.txt',
               'control': corpus_dir / 'control.txt'}
    for kv in args.corpus or []:
        k, v = kv.split('=', 1)
        corpora[k] = Path(v)
    names = ['refusal'] + ([] if args.one_sided else ['delivery']) + ([] if args.no_control else ['control'])
    arts = [tuple(a.split('=', 1)) for a in args.artifact]
    results = {}
    for name, path in arts:
        row = {}
        for corpus in names:
            r = ppl(args.binary, Path(path), corpora[corpus], args.ctx, args.threads)
            if 'ppl' in r:
                r['ln_ppl'] = round(math.log(r['ppl']), 4)
            row[corpus] = r
            print(f'{name} {corpus}: {r}', flush=True)
        results[name] = row
        base = results.get(arts[0][0])
        if base is not None and name != arts[0][0] \
                and base.get('refusal', {}).get('ln_ppl') is not None \
                and row.get('refusal', {}).get('ln_ppl') is not None:
            out = f'{name}: dR={row["refusal"]["ln_ppl"] - base["refusal"]["ln_ppl"]:+.4f}'
            row['delta_R'] = round(row['refusal']['ln_ppl'] - base['refusal']['ln_ppl'], 4)
            if not args.one_sided and row.get('delivery', {}).get('ln_ppl') is not None \
                    and base.get('delivery', {}).get('ln_ppl') is not None:
                br_here = row['refusal']['ln_ppl'] - row['delivery']['ln_ppl']
                br_base = base['refusal']['ln_ppl'] - base['delivery']['ln_ppl']
                row['BR'] = round(br_here, 4)
                row['delta_BR'] = round(br_here - br_base, 4)
                out += f' BR={row["BR"]} delta_BR={row["delta_BR"]}'
            if 'control' in row and 'control' in base and row['control'].get('ln_ppl') is not None and base['control'].get('ln_ppl') is not None:
                row['delta_control'] = round(row['control']['ln_ppl'] - base['control']['ln_ppl'], 4)
                out += f' delta_control={row["delta_control"]}'
            print(out, flush=True)
    report = {
        'scope': 'branch-prediction screen from prefill only; NO tokens generated; '
                 'a BR shift is a preference shift, not a delivery claim — the '
                 'R2 agent read still certifies removal. Corpora built by '
                 'build_ppl_probe.py (D side hand-read only).',
        'runner': {'binary': args.binary, 'ctx': args.ctx, 'threads': args.threads},
        'reference': arts[0][0], 'artifacts': results,
    }
    text = json.dumps(report, indent=1, sort_keys=True)
    print(text)
    if args.out:
        from jdewitt.answers import atomic_text
        atomic_text(args.out, text + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
