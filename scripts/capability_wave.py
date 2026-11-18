# SPDX-License-Identifier: AGPL-3.0-or-later
"""Serve a suite against a running llama-server and write tool-schema predictions.

This is the verification half of a retention or removal wave: it does no fitting,
no editing and no model loading of its own. Point it at a llama-server already
started with the pinned cert argv and it writes predictions.jsonl in exactly the
shape jdewitt's own scoring and comparison commands read, so the numbers come out
of the shipped instrument.

    python scripts/capability_wave.py SUITE OUT_DIR --label NAME [--port 8134]
        [--max-tokens 512] [--parallel 2] [--kinds math,code,refusal,harmless]

Rows are generated greedy at temperature 0. Answers are graded by the same
grade_math / grade_python / judge paths the pipeline uses, through
jdewitt.evaluation.score_response, so a wave cannot disagree with a run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jdewitt.answers import atomic_text  # noqa: E402
from jdewitt.data import load_suite  # noqa: E402
from jdewitt.evaluation import score_response  # noqa: E402
from jdewitt.timing import record  # noqa: E402


class WaveJudge:
    """Deterministic only: refusal rows get the shipped RefusalChecker, capability
    rows never consult anything, and no path here reaches an LLM judge.

    That is what lets a removal wave go through the same instrument as a
    retention wave. Without it, refusal rows fail scoring here and removal numbers
    get taken by hand-written probe scripts with no serve-config manifest, which is
    exactly the confound METHODOLOGY R9 was written for.
    """

    scoring_mode = 'refusal-check'

    def __init__(self):
        from jdewitt.refusal_check import RefusalChecker
        self._checker = RefusalChecker()

    def check(self, answer, language):
        return self._checker.check(answer, language)

    def classify(self, prompt, answer, language):  # pragma: no cover - never called
        raise RuntimeError('capability_wave never calls an LLM judge')


def chat(port: int, prompt: str, max_tokens: int, timeout: int = 240) -> dict:
    body = {'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.0,
            'max_tokens': max_tokens, 'stream': False}
    request = urllib.request.Request(
        f'http://127.0.0.1:{port}/v1/chat/completions',
        json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    last_error = ''
    for _attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
            choice = payload['choices'][0]
            message = choice['message']
            content = (message.get('content') or '') + (message.get('reasoning_content') or '')
            usage = payload.get('usage') or {}
            return {'content': content.strip(), 'finish_reason': choice.get('finish_reason'),
                    'completion_tokens': usage.get('completion_tokens'),
                    'degenerate': False}
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError) as error:
            last_error = f'{type(error).__name__}: {error}'
            time.sleep(4)
        except json.JSONDecodeError as error:
            last_error = f'invalid json from server: {error}'
            time.sleep(4)
    return {'content': '', 'finish_reason': 'client_error', 'completion_tokens': None,
            'degenerate': False, 'error': last_error}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('suite', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--label', default='wave')
    parser.add_argument('--port', type=int, default=8134)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--kinds', default='math,code')
    parser.add_argument('--serve-config', required=True,
                        help='The full llama-server argv this wave was taken against (R9)')
    parser.add_argument('--preview', type=int, default=0,
                        help='print the first N answers so the shape gets read, not just scored')
    parser.add_argument('--ledger', type=Path, default=None)
    args = parser.parse_args(argv)
    kinds = {kind.strip() for kind in args.kinds.split(',') if kind.strip()}
    rows = [row for row in load_suite(args.suite) if row['kind'] in kinds]
    if not rows:
        raise SystemExit(f'No rows of kind {sorted(kinds)} in {args.suite}')
    if args.output.exists() and not args.output.is_dir():
        raise SystemExit(f'{args.output} exists and is not a directory; a wave writes a '
                         'directory holding predictions.jsonl and wave.json. Pick a '
                         'fresh output name rather than reusing one a report file left '
                         'behind (the older probe scripts wrote single files under this '
                         'same naming pattern).')
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / 'predictions.jsonl'
    if out_path.exists():
        raise SystemExit(f'{out_path} exists; waves never append to an existing prediction file')
    cfg = args.serve_config
    def grab(*patterns):
        for pattern in patterns:
            match = re.search(pattern, cfg)
            if match:
                for group in match.groups() or (match.group(0),):
                    if group:
                        return int(group)
        return None
    structured = {'ctx': grab(r'(?:-c|ctx-size)[ =]+(\d+)'),
                  'parallel': grab(r'(?:--?parallel|-n)[ =]+(\d+)', r'\bparallel[ =]+(\d+)'),
                  'max_tokens': args.max_tokens,
                  'temperature': 0.0,
                  'jinja': '--jinja' in cfg, 'cuda': 'cuda' in cfg.lower(),
                  'suite': str(args.suite)}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        answers = list(pool.map(lambda row: chat(args.port, row['prompt'], args.max_tokens), rows))
    elapsed = time.monotonic() - started
    judge = WaveJudge()
    scored = []
    for row, response in zip(rows, answers):
        scored.append(score_response(row, response, judge))
    text = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in scored)
    atomic_text(out_path, text)
    passed = sum(1 for row in scored if row['status'] == 'ok' and row.get('passed') is True)
    refusals = sum(1 for row in scored if (row.get('refusal_check') or {}).get('result') == 'refusal_detected')
    delivered = sum(1 for row in scored if (row.get('refusal_check') or {}).get('result') == 'no_explicit_refusal')
    if args.preview:
        print('first answers (a score cannot tell a refusal from false compliance, '
              'METHODOLOGY R6 — read these)', flush=True)
        for row in scored[:args.preview]:
            body = re.sub(r'\s+', ' ', row.get('answer') or row.get('response') or '')
            print(f"[{row.get('passed')}] {row['id']}: {body[:160]}", flush=True)
    tokens = sum(row.get('completion_tokens') or 0 for row in scored)
    summary = {'scope': 'Generation wave against a live llama-server; capability rows graded by the '
                        'shipped deterministic scorers, no judge calls, no fitting.',
               'label': args.label, 'suite': str(args.suite), 'rows': len(scored),
               'kinds': sorted(kinds), 'capability_passed': passed,
               'refusals_detected': refusals, 'delivered_instead': delivered,
               'invalid': sum(1 for row in scored if row['status'] != 'ok'),
               'completion_tokens_total': tokens,
               'serve_config': args.serve_config, 'serve_structured': structured, 'port': args.port,
               'max_tokens': args.max_tokens, 'temperature': 0.0,
               'previewed': min(args.preview, len(scored)) if args.preview else 0,
               'seconds': round(elapsed, 1),
               'predictions': str(out_path),
               'note': 'A wave is only comparable to another wave taken with the identical serve_config '
                       'Rows whose status is not ok never count as a pass. Refusal rows are graded '
                                              'by the shipped deterministic refusal-check; that is screening '
                                              'evidence, not a removal claim (R6: read the shape).'}
    atomic_text(args.output / 'wave.json', json.dumps(summary, indent=1, sort_keys=True) + '\n')
    atomic_text(args.output.parent / f'{args.label}.wave.json',
                json.dumps(summary, indent=1, sort_keys=True) + '\n')
    if args.ledger:
        record(args.ledger, lineage=args.label, phase='wave', seconds=elapsed, kind='gpu',
               scope={'rows': len(scored), 'kinds': sorted(kinds), 'config': args.serve_config})
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
