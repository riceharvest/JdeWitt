# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cascade verification wave: settle refusal early, pay full decode only where the shape is undecidable.

Usage: python scripts/cascade_verify.py SUITE OUTDIR --serve-config '...' [--k-tokens 600] [--max-tokens 4096]

The shape of a refusal is visible early (opener / substantive decline in the
first ~150-600 tokens), but refusal answers — especially post-edit lecture
pivots — run 2-4k+ characters. Full-budget verification therefore spends most
of its tokens proving the same negative 40 times. This wave decodes every row
to K tokens, applies `refusal_shape.screen_short`, settles rows that decline
inside the window (no further generation, prefix is the record), settles
delivery-candidates early (still read; reading is the R2 contract), and
escalates the rest by CONTINUING the same answer from the server's prefix
cache (assistant-prefix prefill), so an escalated row pays (K + remainder)
once, never a restarted full decode.

Outputs predictions.jsonl compatible with screen_agreement / refusal_shape_audit
(refusal rows carry their truncated answer), plus cascade.json: per-stage row
counts and generated tokens, and the escalation disagreement tally (how many
escalated rows would have been mislabelled by the short window) — the
instrument proves its own truncation rate, so the savings number is audited.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jdewitt.refusal_check import RefusalChecker  # noqa: E402
from jdewitt.refusal_shape import classify, screen_short  # noqa: E402


def chat(port: int, messages: list, max_tokens: int, temperature: float, timeout: int = 300) -> dict:
    body = {'messages': messages, 'temperature': temperature,
            'max_tokens': max_tokens, 'stream': False}
    last_error = ''
    for _attempt in range(4):
        try:
            request = urllib.request.Request(
                f'http://127.0.0.1:{port}/v1/chat/completions',
                json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
            choice = payload['choices'][0]
            message = choice['message']
            content = (message.get('content') or '') + (message.get('reasoning_content') or '')
            usage = payload.get('usage') or {}
            return {'content': content, 'finish_reason': choice.get('finish_reason'),
                    'completion_tokens': usage.get('completion_tokens')}
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(3)
    return {'content': '', 'finish_reason': 'client_error', 'completion_tokens': None,
            'error': last_error}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('suite', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--label', default='cascade')
    parser.add_argument('--port', type=int, default=8134)
    parser.add_argument('--k-tokens', type=int, default=150,
                        help='stage-A decode budget; refusals usually settle inside it. '
                             'K=100 saves 10-24%% but MISSES ALL oracle-form delivery settles '
                             '(0/16 reproduced at K=100 on the 800-row oracle wave) — only use '
                             'it where the corpus is already known delivery-free; default 150')
    parser.add_argument('--settle-mode', choices=('cert', 'screen'), default='cert',
                        help='cert: settle refusal only on explicit self-declination; '
                             'screen: accept decline-marker settles for 56-65%% token savings')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--serve-config', required=True,
                        help='The full llama-server argv this wave was taken against (R9)')
    args = parser.parse_args(argv)

    model_path = re.search(r'-m\s+(\S+)', args.serve_config)
    if not model_path:
        raise SystemExit('--serve-config must contain the -m path (the /props identity gate needs it)')
    model_path = model_path.group(1)
    if not Path(model_path).exists():
        raise SystemExit(f'serve config names a missing model file: {model_path}')
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/props', timeout=5) as response:
            served = json.loads(response.read()).get('model_path', '')
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f'port {args.port} is not answering /props ({exc}); refusing to run a cascade '
                         'against an unknown server') from exc
    if served and Path(served).resolve() != Path(model_path).resolve():
        raise SystemExit(f'port {args.port} is serving {served}, not {model_path}. Refusing to '
                         'record a cascade against an artifact the serve config does not name.')

    rows = [json.loads(line) for line in args.suite.read_text().splitlines() if line.strip()]
    checker = RefusalChecker()
    started = time.monotonic()

    def stage(row):
        messages = [{'role': 'user', 'content': row['prompt']}]
        first = chat(args.port, messages, args.k_tokens, args.temperature)
        prefix = first['content']
        st = screen_short(prefix, mode=args.settle_mode)
        record = {'id': row['id'], 'prompt': row['prompt'], 'kind': row.get('kind'),
                  'language': row.get('language') or 'en', 'family_id': row.get('family_id'),
                  'answer': prefix, 'completion_tokens': first['completion_tokens'],
                  'finish_reason': first['finish_reason'], 'settled': st,
                  'status': 'ok' if st != 'escalate' or prefix else 'truncated_or_error'}
        if st == 'escalate' and prefix:
            cont = chat(args.port, messages + [{'role': 'assistant', 'content': prefix}],
                        max(args.max_tokens - args.k_tokens, 1), args.temperature)
            record['answer'] = prefix + cont['content']
            record['finish_reason'] = cont['finish_reason']
            total = (first['completion_tokens'] or 0) + (cont['completion_tokens'] or 0)
            record['completion_tokens'] = total
            record['escalation_error'] = cont.get('error')
            if not cont['content'] and cont.get('error'):
                record['status'] = 'truncated_or_error'
        return record

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        records = list(pool.map(stage, rows))

    for rec in records:
        if rec.get('answer'):
            rec['refusal_check'] = checker.check(rec['answer'], rec['language'])
            rec['refusal_check_wide'] = checker.check(rec['answer'], rec['language'], wide=True)

    out_path = args.output / 'predictions.jsonl'
    args.output.mkdir(parents=True, exist_ok=True)
    from jdewitt.answers import atomic_text
    atomic_text(out_path, ''.join(json.dumps(r, sort_keys=True) + '\n' for r in records))

    tallies = {'refusal': 0, 'delivery': 0, 'escalate': 0}
    tok_stage_a = 0
    tok_total = 0
    escalation_flip = {'refusal': 0, 'delivery': 0, 'other': 0}
    for rec in records:
        tallies[rec['settled']] += 1
        stage_a = min(rec['completion_tokens'] or 0, args.k_tokens)
        tok_stage_a += stage_a
        tok_total += rec['completion_tokens'] or 0
        if rec['settled'] == 'escalate':
            full = classify(rec['answer'])
            gold = ('refusal' if full['declining'] else
                    'delivery' if full['procedural'] else 'other')
            escalation_flip[gold] += 1
    measured = any(r['completion_tokens'] for r in records)
    summary = {
        'scope': 'Cascade verification wave against a live llama-server; refusal rows are '
                 'the answers as far as they were decoded — full text was never generated for them.',
        'label': args.label, 'suite': str(args.suite), 'rows': len(records),
        'k_tokens': args.k_tokens, 'max_tokens': args.max_tokens,
        'settle_mode': args.settle_mode,
        'temperature': args.temperature,
        'settled_refusal': tallies['refusal'], 'settled_delivery': tallies['delivery'],
        'escalated': tallies['escalate'],
        'escalation_full_shape': escalation_flip,
        'tokens_stage_a': tok_stage_a, 'tokens_total': tok_total,
        'invalid': sum(1 for r in records if r['status'] != 'ok'),
        'measured': measured,
        'serve_config': args.serve_config, 'port': args.port,
        'seconds': round(time.monotonic() - started, 1), 'predictions': str(out_path),
        'note': 'settled_delivery rows are STILL subject to the agent read (METHODOLOGY R2); '
                'this wave minimizes generation, it does not replace certification. escalation_full_shape '
                'counts what the escalated rows actually turned out to be: a high refusal count there '
                'means K is too low for this artifact, a high other count means shape is not decidable '
                'in any window and the read must cover them.'}
    atomic_text(args.output / 'cascade.json', json.dumps(summary, indent=1, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0 if measured else 1


if __name__ == '__main__':
    raise SystemExit(main())
