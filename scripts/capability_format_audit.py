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
        if row['status'] != 'ok':
            counters[f'{kind}_invalid'] += 1
            if row['status'] in ('truncated_or_error', 'timeout'):
                counters[f'{kind}_truncated'] += 1
        else:
            if row.get('passed') is True:
                counters[f'{kind}_strict'] += 1
            if kind == 'math':
                expected = task.get('expected')
                if expected is not None and _matches(expected, _values(raw)):
                    counters[f'{kind}_computed'] += 1
                if ONLY_NUMBER.match(body):
                    counters[f'{kind}_format'] += 1
            else:
                counters[f'{kind}_code_unchecked'] += 1
    def rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None
    math_rows = counters['math_rows']
    return {'label': label, 'rows': rows,
            'math': {'rows': math_rows,
                     'strict': rate(counters['math_strict'], math_rows),
                     'computed_anywhere': rate(counters['math_computed'], math_rows),
                     'format_only_number': rate(counters['math_format'], math_rows),
                     'invalid': counters['math_invalid'],
                     'truncated': counters['math_truncated']},
            'code': {'rows': counters['code_rows'],
                     'strict': rate(counters['code_strict'], counters['code_rows']),
                     'invalid': counters['code_invalid'],
                     'truncated': counters['code_truncated']},
            'note': 'computed_anywhere is a recall measure, not a grade: it asks whether the '
                    'right value appears anywhere in the output, so it separates lost computation '
                    'from lost format-following. code grading is execution, not regex, so it has '
                    'no computed_anywhere row.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('pairs', nargs='+', metavar='LABEL=PREDICTIONS.jsonl')
    parser.add_argument('--suite', type=Path,
                        default=REPO / 'verification/capability/capability-run-en.jsonl')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    report = [audit(item.split('=', 1)[0], Path(item.split('=', 1)[1]), args.suite)
              for item in args.pairs]
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        from jdewitt.answers import atomic_text
        atomic_text(args.out, text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
