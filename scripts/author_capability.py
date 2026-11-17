# SPDX-License-Identifier: AGPL-3.0-or-later
"""Author the capability (retention) suite deterministically.

Why this exists: every suite on disk carried 4 maths and 2 code rows, so the
retention axis of the removal tradeoff had no instrument. These rows are
AUTHORED, parameterised tasks with machine-derived exact answers — they are not
benchmark items, and the provenance file says so. Answers come from a derivation
function over the frozen parameter draw, never typed by hand, so
verify_capability.py can recompute every expected value from the seed instead of
trusting the file.

Usage:
    python scripts/author_capability.py OUTDIR [--seed 20261117] [--language en]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

from jdewitt.data import canonical_prompt, load_suite

PER_FAMILY = 12


def _greatest_common_divisor(first, second):
    first, second = abs(int(first)), abs(int(second))
    while second:
        first, second = second, first % second
    return first


def _least_common_multiple(first, second):
    return first * second // _greatest_common_divisor(first, second)


def _exact(fraction):
    """Return the integer value of a fraction, refusing a non-terminating draw."""
    if fraction.denominator != 1:
        raise ValueError(f'parameter draw gives a non-integer {fraction}; retune the template')
    return int(fraction)


# Each template: family, exact answer derivation, and one full prompt string per
# language. Prompts end with an explicit answer-format clause so a marker-stripped
# answer is still a single extractable token, and so a formatting loss reads as
# formatting rather than arithmetic.
TEMPLATES = [
    {'family': 'arith-times', 'languages': ('en', 'nl'),
     'answer': lambda p: p[0] * p[1] + p[2] * p[3],
     'prompts': {'en': lambda p: f'Compute {p[0]} times {p[1]} plus {p[2]} times {p[3]}. Answer with only the number.',
                 'nl': lambda p: f'Bereken {p[0]} keer {p[1]} plus {p[2]} keer {p[3]}. Antwoord met alleen het getal.'}},
    {'family': 'arith-distributed', 'languages': ('en', 'nl'),
     'answer': lambda p: p[0] * (p[1] + p[2]),
     'prompts': {'en': lambda p: f'Compute {p[0]} times ({p[1]} plus {p[2]}). Answer with only the number.',
                 'nl': lambda p: f'Bereken {p[0]} keer ({p[1]} plus {p[2]}). Antwoord met alleen het getal.'}},
    {'family': 'arith-exact-division', 'languages': ('en', 'nl'),
     'answer': lambda p: _exact(Fraction(p[0] + p[1], p[2])),
     'prompts': {'en': lambda p: f'Compute ({p[0]} plus {p[1]}) divided by {p[2]}. Answer with only the number.',
                 'nl': lambda p: f'Bereken ({p[0]} plus {p[1]}) gedeeld door {p[2]}. Antwoord met alleen het getal.'}},
    {'family': 'pct-forward', 'languages': ('en', 'nl'),
     'answer': lambda p: _exact(Fraction(p[0], 100) * p[1]),
     'prompts': {'en': lambda p: f'How much is {p[0]}% of {p[1]}? Answer with only the number.',
                 'nl': lambda p: f'Hoeveel is {p[0]}% van {p[1]}? Antwoord met alleen het getal.'}},
    {'family': 'pct-reverse', 'languages': ('en', 'nl'),
     'answer': lambda p: _exact(Fraction(p[1] * 100, p[0])),
     'prompts': {'en': lambda p: f'{p[1]} is {p[0]}% of which number? Answer with only the number.',
                 'nl': lambda p: f'{p[1]} is {p[0]}% van welk getal? Antwoord met alleen het getal.'}},
    {'family': 'linear-solve', 'languages': ('en', 'nl'),
     'answer': lambda p: _exact(Fraction(p[2] - p[1], p[0])),
     'prompts': {'en': lambda p: f'Solve for x: {p[0]}x + {p[1]} = {p[2]}. Answer with only the number.',
                 'nl': lambda p: f'Los op voor x: {p[0]}x + {p[1]} = {p[2]}. Antwoord met alleen het getal.'}},
    {'family': 'geometry-rectangle', 'languages': ('en', 'nl'),
     'answer': lambda p: 2 * (p[0] + p[1]),
     'prompts': {'en': lambda p: f'A rectangle is {p[0]} by {p[1]} units. What is its perimeter? Answer with only the number.',
                 'nl': lambda p: f'Een rechthoek is {p[0]} bij {p[1]} eenheden. Wat is de omtrek? Antwoord met alleen het getal.'}},
    {'family': 'geometry-triangle', 'languages': ('en', 'nl'),
     'answer': lambda p: _exact(Fraction(p[0] * p[1], 2)),
     'prompts': {'en': lambda p: f'A triangle has base {p[0]} and height {p[1]}. What is its area? Answer with only the number.',
                 'nl': lambda p: f'Een driehoek heeft basis {p[0]} en hoogte {p[1]}. Wat is de oppervlakte? Antwoord met alleen het getal.'}},
    {'family': 'sequence-arithmetic', 'languages': ('en',),
     'answer': lambda p: p[0] + (p[2] - 1) * p[1],
     'prompts': {'en': lambda p: f'The first term of an arithmetic sequence is {p[0]} and the common difference is {p[1]}. What is term number {p[2]}? Answer with only the number.'}},
    {'family': 'number-theory-gcd', 'languages': ('en', 'nl'),
     'answer': lambda p: _greatest_common_divisor(p[0], p[1]),
     'prompts': {'en': lambda p: f'What is the greatest common divisor of {p[0]} and {p[1]}? Answer with only the number.',
                 'nl': lambda p: f'Wat is de grootste gemene deler van {p[0]} en {p[1]}? Antwoord met alleen het getal.'}},
    {'family': 'number-theory-lcm', 'languages': ('en', 'nl'),
     'answer': lambda p: _least_common_multiple(p[0], p[1]),
     'prompts': {'en': lambda p: f'What is the least common multiple of {p[0]} and {p[1]}? Answer with only the number.',
                 'nl': lambda p: f'Wat is het kleinste gemene veelvoud van {p[0]} en {p[1]}? Antwoord met alleen het getal.'}},
    {'family': 'fraction-sum', 'languages': ('en',),
     'answer': lambda p: str(Fraction(p[0], p[1]) + Fraction(p[2], p[3])),
     'prompts': {'en': lambda p: f'Add {p[0]}/{p[1]} and {p[2]}/{p[3]}. Give the result as a fraction a/b in lowest terms; answer with only the fraction.'}},
    {'family': 'pct-chain-cents', 'languages': ('en',),
     'answer': lambda p: _exact(_exact(Fraction(p[0] * (100 - p[1]), 100)) * Fraction(100 + p[2], 100)),
     'prompts': {'en': lambda p: f'An item costs {p[0]} cents. A {p[1]}% discount is applied, then {p[2]}% tax on the discounted price. How many cents is the final price? Answer with only the number.'}},
    {'family': 'rate-together', 'languages': ('en',),
     'answer': lambda p: str(Fraction(1, 1) / (Fraction(1, p[0]) + Fraction(1, p[1]))),
     'prompts': {'en': lambda p: f'One pump empties a tank in {p[0]} hours and another in {p[1]} hours. Working together, how many hours do they take? Give the result as a fraction a/b; answer with only the fraction.'}},
    {'family': 'counting-combinations', 'languages': ('en',),
     'answer': lambda p: math.comb(p[0], p[1]),
     'prompts': {'en': lambda p: f'From {p[0]} distinct objects, how many groups of exactly {p[1]} can be chosen, order not mattering? Answer with only the number.'}},
]


def _draw_parameters(rng, template_name):
    """One parameter draw per template, retried until the answer is well-formed."""
    for _attempt in range(4000):
        if template_name == 'arith-times':
            parameters = (rng.randint(3, 29), rng.randint(3, 29), rng.randint(3, 29), rng.randint(3, 29))
        elif template_name == 'arith-distributed':
            parameters = (rng.randint(3, 19), rng.randint(2, 40), rng.randint(2, 40))
        elif template_name == 'arith-exact-division':
            divisor = rng.randint(3, 12)
            quotient = rng.randint(7, 40)
            first = rng.randint(1, divisor * quotient - 2)
            parameters = (first, divisor * quotient - first, divisor)
        elif template_name == 'pct-forward':
            percent = rng.choice([5, 10, 15, 20, 25, 40, 50, 60, 75, 80])
            base = rng.randint(2, 40) * (100 // math.gcd(100, percent))
            parameters = (percent, base)
        elif template_name == 'pct-reverse':
            percent = rng.choice([10, 20, 25, 40, 50])
            base = rng.randint(2, 30) * (100 // math.gcd(100, percent))
            parameters = (percent, _exact(Fraction(percent, 100) * base))
        elif template_name == 'linear-solve':
            coefficient = rng.randint(2, 17)
            solution = rng.randint(2, 30)
            constant = rng.randint(1, 60)
            parameters = (coefficient, constant, coefficient * solution + constant)
        elif template_name == 'geometry-rectangle':
            parameters = (rng.randint(3, 45), rng.randint(3, 45))
        elif template_name == 'geometry-triangle':
            base = 2 * rng.randint(3, 30)
            parameters = (base, rng.randint(3, 40))
        elif template_name == 'sequence-arithmetic':
            parameters = (rng.randint(2, 20), rng.randint(3, 15), rng.randint(4, 30))
        elif template_name in ('number-theory-gcd', 'number-theory-lcm'):
            first = rng.randint(6, 90)
            second = rng.randint(6, 90)
            parameters = (first, second)
        elif template_name == 'fraction-sum':
            first = rng.randint(1, 8)
            second = rng.randint(1, 8)
            divisor = rng.randint(2, 9)
            third = rng.randint(1, 8)
            if rng.random() < 0.5:
                parameters = (first, divisor, second, divisor)
            else:
                parameters = (first, divisor, third, divisor * rng.randint(2, 3))
        elif template_name == 'pct-chain-cents':
            price = 20 * rng.randint(10, 200)
            discount = rng.choice([10, 15, 20, 25])
            tax = rng.choice([5, 10, 21])
            parameters = (price, discount, tax)
        elif template_name == 'rate-together':
            parameters = (rng.randint(2, 12), rng.randint(2, 12))
        elif template_name == 'counting-combinations':
            total = rng.randint(6, 15)
            parameters = (total, rng.randint(2, min(6, total - 2)))
        else:
            raise ValueError(f'No parameter draw for template {template_name}')
        try:
            answer = _answer_of(template_name, parameters)
        except ValueError:
            continue
        if isinstance(answer, str):
            return parameters, answer
        if isinstance(answer, int) and answer >= 0:
            return parameters, str(answer)
    raise ValueError(f'Could not find a well-formed draw for {template_name}')


def _answer_of(template_name, parameters):
    factory = next(definition for definition in TEMPLATES if definition['family'] == template_name)
    return factory['answer'](parameters)


def author(seed):
    """Build the suite rows plus the derivation record that backs every answer.

    Parameter draws are retried until the rendered prompt is one the suite has not
    already used: the suite validator refuses duplicate normalized prompts, and two
    parameter draws that render the same question are not two measurements.
    """
    rng = random.Random(seed)
    rows, derivations = [], []
    seen = set()
    for template in TEMPLATES:
        family = template['family']
        for index in range(PER_FAMILY):
            for language in template['languages']:
                for _attempt in range(200):
                    parameters, expected = _draw_parameters(rng, family)
                    prompt = template['prompts'][language](parameters)
                    key = (language, canonical_prompt(prompt))
                    if key not in seen:
                        break
                else:
                    raise ValueError(f'Could not find a fresh prompt draw for {family} {language}')
                seen.add(key)
                row = {'id': f'cap-{family}-{language}-{index:02d}', 'kind': 'math',
                       'language': language, 'prompt': prompt, 'expected': expected,
                       'family_id': f'cap-{family}', 'category': 'capability'}
                rows.append(row)
                derivations.append({'id': row['id'], 'template': family, 'parameters': list(parameters),
                                    'expected': expected})
    return rows, derivations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--language', default=None, help='Restrict to one language (en or nl)')
    args = parser.parse_args(argv)
    rows, derivations = author(args.seed)
    if args.language:
        rows = [row for row in rows if row['language'] == args.language]
        keep = {row['id'] for row in rows}
        derivations = [entry for entry in derivations if entry['id'] in keep]
    args.output.mkdir(parents=True, exist_ok=True)
    suite = args.output / 'capability-suite.jsonl'
    suite.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    text = suite.read_text()
    payload = {
        'scope': 'Authored parameterised capability tasks with machine-derived answers. '
                 'Not benchmark items: no public corpus, no human annotation, no leaderboard claim. '
                 'They measure capability retention as a same-suite paired change, nothing else.',
        'seed': args.seed,
        'per_family': PER_FAMILY,
        'templates': [template['family'] for template in TEMPLATES],
        'languages': sorted({row['language'] for row in rows}),
        'rows': len(rows),
        'counts_by_language': {language: sum(1 for row in rows if row['language'] == language)
                               for language in sorted({row['language'] for row in rows})},
        'counts_by_family': {template['family']: sum(1 for row in rows if row['family_id'] == f"cap-{template['family']}")
                             for template in TEMPLATES},
        'known_limits': [
            'Fraction-form and cents-form items partly measure answer formatting, not only arithmetic.',
            'Percent and fraction prompts assume a Western decimal point and a/b notation.',
            'Dutch rows exist for six families only; multilingual retention is incomplete by design.',
            'Difficulty is a single authored ladder, not calibrated against any published benchmark.',
            'No code rows are produced by this generator; code tasks are authored separately and graded in the sandbox.',
        ],
        'join': 'Join into a run with an importer passthrough source: {"id":"capability","passthrough":true,'
                '"path":"verification/capability/capability-suite.jsonl","kinds":["math"]}. Put the rows in ONE '
                'split, not both: the fit never uses capability rows, so a selection/holdout split of them buys '
                'nothing, and validate_splits would refuse the shared cap-* family ids anyway. The certification '
                'holdout is the right home because that is the suite a release reports from.',
        'sha256': hashlib.sha256(text.encode()).hexdigest(),
    }
    (args.output / 'capability-provenance.json').write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    (args.output / 'capability-derivations.json').write_text(
        json.dumps({'seed': args.seed, 'entries': derivations}, indent=1, sort_keys=True) + '\n')
    load_suite(suite)  # the suite must pass the tool's own validator before it is called a suite
    print(json.dumps({'suite': str(suite), 'rows': len(rows),
                      'languages': payload['counts_by_language'],
                      'sha256': payload['sha256']}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
