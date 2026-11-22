# SPDX-License-Identifier: AGPL-3.0-or-later
"""Author the two curated slices: code execution and long-context maths.

Both are generated from a frozen seed, never typed: every task's tests are
computed from an independent expression, the reference solution and a written
mutant are executed through the real sandbox (`grade_python`) or grader
(`grade_math`), and the script refuses to write anything unless every reference
passes and every mutant fails. Determinism is proven by the caller running it
twice and diffing the bytes.

Usage:
    python scripts/author_slices.py OUTDIR [--seed 7]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jdewitt.evaluation import grade_math, grade_python  # noqa: E402

SCOPE = ('Authored ladder, not benchmark items; no leaderboard claim. Answers are '
         'derived from the recorded seeded draw and every row was proven gradeable '
         'before it was written.')


def _jsonl(rows):
    return ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows)


# ---------------------------------------------------------------- code slice
# Each family: a reference, a written mutant, and a data generator. Tests are
# built from an INDEPENDENT expression for the expected value, never by calling
# the reference string, so a wrong reference cannot make its own tests pass.
def code_families():
    def run_length(rng, i):
        alphabet = 'ab' if i % 2 == 0 else 'abc'
        s = ''.join(rng.choice(alphabet) for _ in range(rng.randint(5, 9)))
        runs = []
        for ch in s:
            if runs and runs[-1][0] == ch:
                runs[-1][1] += 1
            else:
                runs.append([ch, 1])
        expected = [(ch, n) for ch, n in runs]
        data = (s, expected)
        return data

    def window_sum(rng, i):
        values = [rng.randint(-4, 9) for _ in range(rng.randint(5, 8))]
        k = rng.randint(2, len(values) - 1)
        windows = [sum(values[j:j + k]) for j in range(len(values) - k + 1)]
        return (values, k), max(windows)

    def keys_upper(rng, i):
        pairs = [(f'k{rng.randint(0, 9)}{chr(97 + rng.randint(0, 5))}', rng.randint(1, 9))
                 for _ in range(rng.randint(3, 5))]
        expected = {key: value for key, value in pairs if value % 2 == 0}
        return pairs, expected

    def count_pairs(rng, i):
        values = [rng.randint(0, 3) for _ in range(rng.randint(6, 10))]
        expected = sum(1 for j in range(len(values) - 1) if values[j] == values[j + 1])
        return values, expected

    def digit_sum(rng, i):
        n = rng.randint(1000, 99999)
        expected = 0
        value = n
        while value:
            expected += value % 10
            value //= 10
        return n, expected

    def second_biggest(rng, i):
        values = [rng.randint(1, 20) for _ in range(rng.randint(3, 6))]
        values.append(max(values))  # a duplicate of the maximum: distinctness matters
        unique = sorted(set(values))
        if len(unique) < 2:
            values, unique = [0, 0, 1], [0, 1]
        return values, unique[-2]

    return [
        {'family': 'ce-run-length', 'name': 'run_lengths',
         'prompt': 'Write a function run_lengths(text) returning a list of (character, count) '
                   'pairs for consecutive runs, in order.',
         'ref': 'def run_lengths(text):\n'
                '    runs = []\n'
                '    for ch in text:\n'
                '        if runs and runs[-1][0] == ch:\n'
                '            runs[-1][1] += 1\n'
                '        else:\n'
                '            runs.append([ch, 1])\n'
                '    return [(ch, n) for ch, n in runs]',
         'mutant': 'def run_lengths(text):\n'
                   '    runs = []\n'
                   '    for ch in text:\n'
                   '        if runs and runs[-1][0] == ch:\n'
                   '            runs[-1][1] += 1\n'
                   '        else:\n'
                   '            runs.append([ch, 1])\n'
                   '    return [(ch, n) for ch, n in runs][:-1]',
         'gen': run_length,
         'asserts': lambda data: [f'assert run_lengths({data[0]!r})=={data[1]!r}']},
        {'family': 'ce-window-sum', 'name': 'max_window_sum',
         'prompt': 'Write a function max_window_sum(values, k) returning the largest sum of k '
                   'consecutive entries of values (k between 1 and len(values)).',
         'ref': 'def max_window_sum(values, k):\n'
                '    return max(sum(values[j:j + k]) for j in range(len(values) - k + 1))',
         'mutant': 'def max_window_sum(values, k):\n'
                   '    return min(sum(values[j:j + k]) for j in range(len(values) - k + 1))',
         'gen': window_sum,
         'asserts': lambda data: [f'assert max_window_sum({data[0][0]!r},{data[0][1]})=={data[1]!r}']},
        {'family': 'ce-even-values', 'name': 'even_only',
         'prompt': 'Write a function even_only(pairs) taking a list of (key, value) tuples and '
                   'returning a dict of the pairs whose value is even.',
         'ref': 'def even_only(pairs):\n'
                '    return {key: value for key, value in pairs if value % 2 == 0}',
         'mutant': 'def even_only(pairs):\n'
                   '    return {key: value for key, value in pairs if value % 2 == 1}',
         'gen': keys_upper,
         'asserts': lambda data: [f'assert even_only({data[0]!r})=={data[1]!r}']},
        {'family': 'ce-adjacent', 'name': 'count_adjacent',
         'prompt': 'Write a function count_adjacent(values) returning how many neighbouring '
                   'pairs in the list are equal.',
         'ref': 'def count_adjacent(values):\n'
                '    return sum(1 for j in range(len(values) - 1) if values[j] == values[j + 1])',
         'mutant': 'def count_adjacent(values):\n'
                   '    return sum(1 for j in range(len(values) - 1) if values[j] != values[j + 1])',
         'gen': count_pairs,
         'asserts': lambda data: [f'assert count_adjacent({data[0]!r})=={data[1]!r}']},
        {'family': 'ce-digit-sum', 'name': 'digit_sum',
         'prompt': 'Write a function digit_sum(n) returning the sum of the decimal digits of the '
                   'nonnegative integer n.',
         'ref': 'def digit_sum(n):\n'
                '    total = 0\n'
                '    while n:\n'
                '        total += n % 10\n'
                '        n //= 10\n'
                '    return total',
         'mutant': 'def digit_sum(n):\n'
                   '    total = 0\n'
                   '    while n > 10:\n'
                   '        total += n % 10\n'
                   '        n //= 10\n'
                   '    return total',
         'gen': digit_sum,
         'asserts': lambda data: [f'assert digit_sum({data[0]})=={data[1]!r}']},
        {'family': 'ce-second-biggest', 'name': 'second_biggest',
         'prompt': 'Write a function second_biggest(values) returning the second largest '
                   'distinct value in a list with at least two distinct values.',
         'ref': 'def second_biggest(values):\n'
                '    return sorted(set(values))[-2]',
         'mutant': 'def second_biggest(values):\n'
                   '    return sorted(values)[-2]',
         'gen': second_biggest,
         'asserts': lambda data: [f'assert second_biggest({data[0]!r})=={data[1]!r}']},
    ]


def build_code_slice(seed, per_family):
    rng = random.Random(seed)
    rows, proofs = [], []
    for family in code_families():
        local = random.Random(rng.randrange(1 << 30))
        for index in range(per_family):
            data = family['gen'](local, index)
            tests = '\n'.join(family['asserts'](data))
            # One example line keeps ten tasks in a family from sharing a normalized
            # prompt (the suite loader refuses duplicate prompts), while the grading
            # tests carry all the real inputs.
            prompt = (family['prompt'] + ' Use only builtins; no imports. Example check: ' +
                      tests.split('\n')[0][len('assert '):])
            rows.append({'id': f"{family['family']}-en-{index:02d}", 'family_id': family['family'],
                         'kind': 'code', 'language': 'en', 'category': 'curated-code',
                         'prompt': prompt, 'tests': tests})
            proofs.append((family, tests, data))
    return rows, proofs


# ------------------------------------------------------------- maths slice
# The prompt is a long report (seeded numbers, irrelevant paragraphs) plus one
# question; the expected value is computed by an independent expression.
def _report(rng, entries):
    lines = []
    for index, (label, value) in enumerate(entries):
        lines.append(f'- {label} reported {value} units in period {index + 1}.')
    filler = ('The review notes that reporting practice varied between offices and that '
              'no adjustment was made for seasonal effects. ')
    return ('Quarterly operations report\n' + filler * rng.randint(14, 20) + '\n' +
            '\n'.join(lines) + '\n' + filler * rng.randint(10, 16))


def math_families():
    def week_sum(rng, i):
        values = [rng.randint(10, 90) for _ in range(rng.randint(10, 16))]
        pick = rng.randrange(len(values))
        return values, f'What total did period {pick + 1} report?', values[pick]

    def difference(rng, i):
        values = [rng.randint(20, 90) for _ in range(rng.randint(8, 14))]
        a, b = sorted(rng.sample(range(len(values)), 2))
        return (values, a, b, f'How many more units did period {a + 1} report than period {b + 1}?',
                values[a] - values[b])

    def scaled_total(rng, i):
        values = [rng.randint(5, 40) for _ in range(rng.randint(8, 12))]
        factor = rng.randint(2, 5)
        return (values, factor, f'What is the total of all periods multiplied by {factor}?',
                sum(values) * factor)

    def average(rng, i):
        values = [rng.randint(10, 90) for _ in range(rng.choice([10, 12, 14]))]
        total = sum(values)
        # round to the nearest integer the way the prompt asks
        expected = int(round(total / len(values)))
        return values, 'What was the mean units per period, rounded to the nearest whole number?', expected

    def share(rng, i):
        values = [rng.randint(10, 90) for _ in range(rng.randint(8, 12))]
        pick = rng.randrange(len(values))
        expected = int(round(values[pick] * 100 / sum(values)))
        return values, f'What percentage of the total did period {pick + 1} report, rounded to the nearest whole number?', expected

    def remainder(rng, i):
        values = [rng.randint(10, 90) for _ in range(rng.randint(8, 12))]
        removed = rng.randint(1, 20)
        return values, f'The {removed} units held in reserve were removed from the total. How many units remain?', sum(values) - removed

    return [
        {'family': 'lc-period-sum', 'gen': week_sum},
        {'family': 'lc-difference', 'gen': difference},
        {'family': 'lc-scaled-total', 'gen': scaled_total},
        {'family': 'lc-mean', 'gen': average},
        {'family': 'lc-share', 'gen': share},
        {'family': 'lc-remainder', 'gen': remainder},
    ]


def build_math_slice(seed, per_family):
    rng = random.Random(seed ^ 0x5EED)
    rows, proofs = [], []
    for family in math_families():
        local = random.Random(rng.randrange(1 << 30))
        for index in range(per_family):
            data = family['gen'](local, index)
            values = data[0] if isinstance(data[0], list) else data[0]
            entries = [(f'site {chr(65 + j)}', value) for j, value in enumerate(values)]
            report = _report(local, entries)
            question, expected = data[-2], data[-1]
            prompt = report + '\n' + question + ' Answer with only the number.'
            rows.append({'id': f"{family['family']}-en-{index:02d}", 'family_id': family['family'],
                         'kind': 'math', 'language': 'en', 'category': 'curated-long-context',
                         'prompt': prompt, 'expected': str(expected),
                         'context_chars': len(report)})
            proofs.append((family['family'], prompt, expected))
    return rows, proofs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--per-family', type=int, default=10)
    args = parser.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    code_rows, code_proofs = build_code_slice(args.seed, args.per_family)
    print(f'code: {len(code_rows)} rows, grading every reference and mutant in the sandbox')
    bad = []
    for (family, tests, _data), row in zip(code_proofs, code_rows):
        ref = grade_python(family['ref'], tests)
        mut = grade_python(family['mutant'], tests)
        if not ref.get('passed'):
            bad.append((row['id'], 'reference failed', ref.get('reason')))
        if mut.get('passed'):
            bad.append((row['id'], 'mutant passed', mut.get('reason')))
    if bad:
        print('REFUSING to write code slice:')
        for item in bad[:10]:
            print(' ', item)
        raise SystemExit(1)

    math_rows, math_proofs = build_math_slice(args.seed, args.per_family)
    print(f'maths: {len(math_rows)} rows, grading every expected answer and a perturbation')
    for family, prompt, expected in math_proofs:
        if not grade_math(f'\\boxed{{{expected}}}', str(expected)):
            bad.append((family, 'expected answer not graded correct'))
        perturbed = str(int(expected) + 1)
        if grade_math(f'\\boxed{{{perturbed}}}', str(expected)):
            bad.append((family, 'perturbed answer graded correct'))
    if bad:
        print('REFUSING to write maths slice:')
        for item in bad[:10]:
            print(' ', item)
        raise SystemExit(1)

    code_selection = [row for row in code_rows if row['family_id'] in
                      {'ce-run-length', 'ce-window-sum', 'ce-even-values'}]
    code_holdout = [row for row in code_rows if row['family_id'] not in
                    {'ce-run-length', 'ce-window-sum', 'ce-even-values'}]
    math_selection = [row for row in math_rows if row['family_id'] in
                      {'lc-period-sum', 'lc-difference', 'lc-scaled-total'}]
    math_holdout = [row for row in math_rows if row['family_id'] not in
                    {'lc-period-sum', 'lc-difference', 'lc-scaled-total'}]
    families = {split: sorted({row['family_id'] for row in rows})
                for split, rows in {'selection': code_selection + math_selection,
                                    'holdout': code_holdout + math_holdout}.items()}
    shared = set(families['selection']) & set(families['holdout'])
    if shared:
        raise SystemExit(f'family overlap between splits: {sorted(shared)}')

    written = {}
    for name, rows in (('code-exec-selection', code_selection), ('code-exec-holdout', code_holdout),
                       ('long-context-math-selection', math_selection),
                       ('long-context-math-holdout', math_holdout)):
        path = out / f'{name}.jsonl'
        path.write_text(_jsonl(rows))
        written[name] = {'file': path.name, 'tasks': len(rows),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    provenance = {'seed': args.seed, 'per_family': args.per_family, 'scope': SCOPE,
                  'verification': 'every code reference passed and every mutant failed inside '
                                  'the pinned python:3.12-slim sandbox; every maths expected answer '
                                  'was graded correct and a perturbed answer rejected',
                  'source': 'jdewitt authored generator (scripts/author_slices.py), no hub corpus',
                  'families': families, 'files': written}
    (out / 'slices-provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True))
    print(json.dumps({'status': 'written', 'files': written}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
