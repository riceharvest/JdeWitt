# SPDX-License-Identifier: AGPL-3.0-or-later
"""Split 'capability pass rate' into the three things it actually conflates.

A pass rate over the authored ladder is contaminated by output style: a model
that answers '92' and a model that writes 200 tokens of LaTeX ending in the
right number are not being measured by the same yardstick, and a wave taken at a
serve config whose per-slot context is smaller than the model's new verbosity
scores truncation as capability loss. This script regrades saved prediction
files with no GPU and no new inference, reporting per label:

  strict    jdewitt's own last-number / sandbox verdict (the shipped instrument)
  computed  the expected value appears as a standalone number anywhere in the
            answer (did the computation survive, ignoring how it was dressed)
  format    the answer matches the suite's 'only the number' contract
  truncated rows the serve config cut off before the model finished

    python scripts/capability_format_audit.py LABEL=PREDICTIONS.jsonl [...]

Strict is never recomputed here; it is read from the row, so the numbers always
match what the comparison commands reported.

Answer length is reported per kind alongside every rate, because a pass-rate drop
has two opposite causes that look identical in a count: the model lost the
computation, or the model stopped writing. On qwen3-4B an edit that halved the maths
pass rate had a median maths answer of 3 characters — it was emitting a bare number
and stopping, which no rate can distinguish from a wrong one. A paired run therefore
flags a length collapse when the after-median falls below --collapse-of of the
before-median.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jdewitt.data import load_suite  # noqa: E402
from jdewitt.stats import cohens_h, paired_permutation_test, wilson_interval  # noqa: E402

NUMBER = re.compile(r'(?<![\w./])-?\d+(?:\.\d+)?(?:/\d+)?(?!\w|[\w/]\d|/)')
ONLY_NUMBER = re.compile(r'^\s*[-+]?\d+(?:[.,]\d+)?(?:/\d+)?\s*[.:\n]*$')



def _values(text: str) -> list[str]:
    found = re.findall(r'\\\\boxed\\{([^{}]+)\\}', text)
    found += NUMBER.findall(text)
    return [value.strip() for value in found]


def _matches(expected: str, values: list[str]) -> bool:
    for value in values:
        try:
            if Fraction(value.replace(',', '')) == Fraction(expected):
                return True
        except (ValueError, ZeroDivisionError):
            if value == expected.strip():
                return True
    return False


def audit(label: str, predictions: Path, suite: Path) -> dict:
    tasks = {row['id']: row for row in load_suite(suite)}
    counters = Counter()
    kinds = Counter()
    flags = {}
    lens = {'math': [], 'code': []}
    tokens = {'math': 0, 'code': 0}
    rows = 0
    for line in predictions.read_text().splitlines():
        row = json.loads(line)
        task = tasks.get(row['id'])
        if task is None:
            counters['unknown_row'] += 1
            continue
        rows += 1
        kind = task['kind']
        kinds[kind] += 1
        counters[f'{kind}_rows'] += 1
        text = row.get('answer') or ''
        raw = row.get('raw') or text
        body = row.get('answer') or ''
        flags[row['id']] = {'strict': row.get('passed') is True,
                        'computed': False, 'format': False, 'ok': row['status'] == 'ok',
                        'kind': kind, 'chars': len(body)}
        if row['status'] != 'ok':
            counters[f'{kind}_invalid'] += 1
            if row['status'] in ('truncated_or_error', 'timeout'):
                counters[f'{kind}_truncated'] += 1
        else:
            if kind in lens:
                lens[kind].append(len(body))
                tokens[kind] += row.get('completion_tokens') or 0
            if row.get('passed') is True:
                counters[f'{kind}_strict'] += 1
            if kind == 'math':
                expected = task.get('expected')
                if expected is not None and _matches(expected, _values(raw)):
                    counters[f'{kind}_computed'] += 1
                    flags[row['id']]['computed'] = True
                if ONLY_NUMBER.match(body):
                    counters[f'{kind}_format'] += 1
                    flags[row['id']]['format'] = True
            else:
                counters[f'{kind}_code_unchecked'] += 1
    def rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None
    def measure(num: int, den: int) -> dict:
        interval = wilson_interval(num, den) if den else None
        return {'passed': num, 'rows': den, 'rate': rate(num, den),
                'wilson': interval['lower'] if interval else None,
                'wilson_upper': interval['upper'] if interval else None}
    math_rows = counters['math_rows']
    def length(values: list[int], total_tokens: int) -> dict:
        if not values:
            return {'rows': 0}
        ordered = sorted(values)
        count = len(ordered)
        return {'rows': count, 'median_chars': ordered[count // 2],
                'p10_chars': ordered[int(0.1 * count)],
                'p90_chars': ordered[min(count - 1, int(0.9 * count))],
                'max_chars': ordered[-1], 'completion_tokens_total': total_tokens}
    return {'label': label, 'rows': rows,
            'math': {'strict': measure(counters['math_strict'], math_rows),
                     'computed_anywhere': measure(counters['math_computed'], math_rows),
                     'format_only_number': measure(counters['math_format'], math_rows),
                     'invalid': counters['math_invalid'],
                     'truncated': counters['math_truncated'],
                     'length': length(lens['math'], tokens['math'])},
            'code': {'strict': measure(counters['code_strict'], counters['code_rows']),
                     'invalid': counters['code_invalid'],
                     'truncated': counters['code_truncated'],
                     'length': length(lens['code'], tokens['code'])},
            'flags': flags,
            'note': 'computed_anywhere is a recall measure, not a grade: it asks whether the '
                    'right value appears anywhere in the output, so it separates lost computation '
                    'from lost format-following. code grading is execution, not regex, so it has '
                    'no computed_anywhere row.'}


def paired(before: dict, after: dict, kind: str, collapse_of: float = 0.25) -> dict:
    """Paired per-row comparison of every audited rate, on rows both sides generated.

    A sign-flip permutation on the paired 0/1 labels is the honest test here: the
    rows are the same prompts, so the pairing carries the difficulty, and a metric
    that only moved because a different row got sampled cannot survive it.
    """
    shared = sorted(set(before['flags']) & set(after['flags']))
    ids = [pid for pid in shared
           if before['flags'][pid]['ok'] and after['flags'][pid]['ok']
           and before['flags'][pid]['kind'] == kind and after['flags'][pid]['kind'] == kind]
    out = {}
    for metric in ('strict', 'computed', 'format'):
        if metric == 'computed' and kind != 'math':
            continue
        a = [1 if before['flags'][pid][metric] else 0 for pid in ids]
        b = [1 if after['flags'][pid][metric] else 0 for pid in ids]
        if not ids:
            continue
        permutation = paired_permutation_test([bb - aa for aa, bb in zip(a, b)])
        out[metric] = {
            'rows': len(ids), 'before_pass': sum(a), 'after_pass': sum(b),
            'before_rate': round(sum(a) / len(ids), 4), 'after_rate': round(sum(b) / len(ids), 4),
            'delta': round((sum(b) - sum(a)) / len(ids), 4),
            'cohens_h': round(cohens_h(sum(a), len(ids), sum(b), len(ids)), 4),
            'paired_sign_flip_p_two_sided': permutation.get('p_two_sided'),
            'fixed': sum(1 for aa, bb in zip(a, b) if bb and not aa),
            'broke': sum(1 for aa, bb in zip(a, b) if aa and not bb)}
    before_lens = sorted(before['flags'][pid]['chars'] for pid in ids)
    after_lens = sorted(after['flags'][pid]['chars'] for pid in ids)
    if before_lens:
        before_median = before_lens[len(before_lens) // 2]
        after_median = after_lens[len(after_lens) // 2]
        out['length'] = {
            'before_median_chars': before_median, 'after_median_chars': after_median,
            'ratio': round(after_median / before_median, 4) if before_median else None,
            'collapse_flag': bool(before_median and after_median < collapse_of * before_median),
            'note': 'a pass-rate drop can mean lost computation or lost writing; length is '
                    'the only thing in this report that tells them apart, and a genuine '
                    'removal edit lengthens answers rather than shrinking them'}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('pairs', nargs='+', metavar='LABEL=PREDICTIONS.jsonl')
    parser.add_argument('--suite', type=Path,
                        default=REPO / 'verification/capability/capability-run-en.jsonl')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--collapse-of', type=float, default=0.25,
                        help='flag a length collapse when the after-median answer falls '
                             'below this fraction of the before-median')
    args = parser.parse_args(argv)
    audits = [audit(item.split('=', 1)[0], Path(item.split('=', 1)[1]), args.suite)
              for item in args.pairs]
    report = {'labels': [item['label'] for item in audits],
              'metrics': [{key: value for key, value in item.items() if key != 'flags'}
                          for item in audits]}
    if len(audits) == 2:
        report['paired'] = {'rows': 'only rows with status ok on both sides, split by kind',
                            'pairs': {'before': audits[0]['label'], 'after': audits[1]['label']},
                            'per_kind': {kind: paired(audits[0], audits[1], kind,
                                                      args.collapse_of)
                                         for kind in ('math', 'code')}}
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        from jdewitt.answers import atomic_text
        atomic_text(args.out, text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
