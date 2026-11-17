# SPDX-License-Identifier: AGPL-3.0-or-later
"""Re-derive every expected answer in the capability suite and prove the scorer separates right from wrong.

Run after authoring, and again any time a suite file is touched:

    python scripts/verify_capability.py verification/capability

What it proves, and what it does not:
  - every stored expected value equals the value recomputed from the frozen
    parameter draw (so an edited file cannot quietly keep its own answers);
  - jdewitt.evaluation.grade_math accepts a correct rendering and rejects a
    perturbed one on every row (an instrument that cannot fail is not an instrument);
  - the suite passes the tool's own validator, including id and duplicate-prompt rules.
It does not prove the maths is correct in the philosophical sense — the answer
factories are the same code the generator used, so arithmetic bugs common to
both pass. The derivation record exists so a human can audit the draw itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from author_capability import _answer_of  # noqa: E402
from jdewitt.data import load_suite  # noqa: E402
from jdewitt.evaluation import grade_math  # noqa: E402


def perturb(expected):
    """A wrong-but-well-formed rendering for the negative check."""
    if '/' in expected:
        fraction = Fraction(expected)
        return str(fraction + 1)
    try:
        return str(int(expected) + 1)
    except ValueError:
        return 'not-a-number'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('directory', type=Path, help='Directory holding capability-suite.jsonl and its records')
    args = parser.parse_args(argv)
    suite_path = args.directory / 'capability-suite.jsonl'
    derivations_path = args.directory / 'capability-derivations.json'
    if not suite_path.is_file() or not derivations_path.is_file():
        raise SystemExit(f'Missing {suite_path.name} or {derivations_path.name} in {args.directory}')
    rows = load_suite(suite_path)
    by_id = {row['id']: row for row in rows}
    entries = json.loads(derivations_path.read_text())['entries']
    mismatches, accepted, rejected = [], 0, 0
    for entry in entries:
        row = by_id.get(entry['id'])
        if row is None:
            mismatches.append({'id': entry['id'], 'problem': 'derivation has no suite row'})
            continue
        recomputed = _answer_of(entry['template'], entry['parameters'])
        recomputed = str(recomputed) if not isinstance(recomputed, str) else recomputed
        if recomputed != row['expected'] or entry['expected'] != row['expected']:
            mismatches.append({'id': entry['id'], 'problem': 'stored answer differs from recomputed',
                               'stored': row['expected'], 'recomputed': recomputed})
            continue
        wrong = perturb(row['expected'])
        right_prompt_shaped = f'... so the answer is {row["expected"]}.'
        if not grade_math(right_prompt_shaped, row['expected']):
            mismatches.append({'id': row['id'], 'problem': 'scorer rejected the correct answer'})
            continue
        if grade_math(wrong, row['expected']):
            mismatches.append({'id': row['id'], 'problem': 'scorer accepted a perturbed answer', 'wrong': wrong})
            continue
        accepted += 1
        rejected += 1
    unused = sorted(set(by_id) - {entry['id'] for entry in entries})
    text = suite_path.read_bytes()
    report = {'scope': 'Re-derivation plus scorer falsification. Not a proof of arithmetic truth: the answer '
                       'factories are shared with the generator.',
              'suite': str(suite_path), 'sha256': hashlib.sha256(text).hexdigest(),
              'rows': len(rows), 'derivations': len(entries),
              'accepted_and_rejected_checked': accepted,
              'rows_without_a_derivation': len(unused),
              'mismatch_count': len(mismatches),
              'problems': mismatches[:25],
              'counts_by_language': {language: sum(1 for row in rows if row['language'] == language)
                                     for language in sorted({row['language'] for row in rows})},
              'counts_by_family': {row['family_id']: sum(1 for other in rows if other['family_id'] == row['family_id'])
                                   for row in rows if row.get('family_id')}}
    print(json.dumps(report, indent=1))
    if mismatches or unused:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
