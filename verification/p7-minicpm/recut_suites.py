# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recut the two P7 MiniCPM-V 4.7 suites, byte for byte, from committed sources.

`suites.provenance.json` records a rule and a sha256 for each suite; this script
is the executable form of that rule, so a stranger can recut both files and check
the digest instead of trusting the prose. It reads only committed files and
writes only with ``--write``; by default it prints the digests it would produce.

    python verification/p7-minicpm/recut_suites.py            # check
    python verification/p7-minicpm/recut_suites.py --write    # rewrite the two files

Rules (identical to suites.provenance.json):

fit-suite.jsonl
  refusal rows: from verification/harsh-suites/selection.jsonl, kind=refusal,
    the first 5 rows of each of the 8 sorted family ids (document order);
  harmless rows: the first 20 kind=harmless rows of
    verification/p5/sae/inputs/suite.jsonl in document order;
  refusal rows first, then harmless rows. No prompt appears in advbench-40 and no
    prompt is in the held-out suite (both checked below).

heldout-24f.jsonl
  from verification/harsh-suites/holdout.jsonl, kind=refusal, prompt not verbatim
  in advbench-40, family_id != hb-cp-hold-copyright; only families that carry at
  least 5 eligible rows (so every family contributes 5); the first 24 sorted
  family ids, the first 5 rows per family by id.

The "at least 5 eligible rows" clause and the "first 20 harmless rows" clause are
in this script because the prose rule in suites.provenance.json left them implicit
and the literal reading did not reproduce the committed digests: plain sorted
family order starts with the two-row adv-hold-* families, which is not what the
committed file contains. The committed files are the ones pinned; this script is
the executable rule that reproduces them.

A mismatch is a hard failure: the two files are the P7 operands and their digests
are pinned in suites.provenance.json and operands.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

EXPECTED = {
    'fit-suite.jsonl': 'e2e83db8e56c55df0bc930154a0c8159b0b58333820c68ee1bc695e4e6771df4',
    'heldout-24f.jsonl': '9eb470f9d179b16a7777c6fa39ebb34c47231a46900c464c30b1eceab373100a',
}
FIT_FAMILIES = 8
FIT_ROWS_PER_FAMILY = 5
HARMLESS_ROWS = 20
HELDOUT_FAMILIES = 24
HELDOUT_ROWS_PER_FAMILY = 5


def read_rows(relative: str) -> list[dict]:
    path = REPO / relative
    if not path.is_file():
        raise SystemExit(f'{relative} is missing; the suite cannot be recut from this checkout')
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def render(rows: list[dict]) -> str:
    return ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def fit_suite() -> list[dict]:
    selection = read_rows('verification/harsh-suites/selection.jsonl')
    harmless = read_rows('verification/p5/sae/inputs/suite.jsonl')
    refusals = [row for row in selection if row.get('kind') == 'refusal']
    families = sorted({row['family_id'] for row in refusals})
    if len(families) < FIT_FAMILIES:
        raise SystemExit(f'selection.jsonl carries {len(families)} refusal families, need '
                         f'{FIT_FAMILIES}')
    chosen = []
    for family in families[:FIT_FAMILIES]:
        members = [row for row in refusals if row['family_id'] == family]
        if len(members) < FIT_ROWS_PER_FAMILY:
            raise SystemExit(f'{family} has {len(members)} refusal rows, need '
                             f'{FIT_ROWS_PER_FAMILY}')
        chosen += members[:FIT_ROWS_PER_FAMILY]
    good = [row for row in harmless if row.get('kind') == 'harmless']
    if len(good) < HARMLESS_ROWS:
        raise SystemExit(f'the harmless source carries {len(good)} kind=harmless rows, need '
                         f'{HARMLESS_ROWS}')
    return chosen + good[:HARMLESS_ROWS]


def heldout_suite() -> list[dict]:
    holdout = read_rows('verification/harsh-suites/holdout.jsonl')
    forbidden = {row['prompt'] for row in read_rows('verification/screens/advbench-40.jsonl')}
    eligible = [row for row in holdout
                if row.get('kind') == 'refusal'
                and row['prompt'] not in forbidden
                and row.get('family_id') != 'hb-cp-hold-copyright']
    counts: dict[str, int] = {}
    for row in eligible:
        counts[row['family_id']] = counts.get(row['family_id'], 0) + 1
    families = sorted(family for family, count in counts.items()
                      if count >= HELDOUT_ROWS_PER_FAMILY)
    if len(families) < HELDOUT_FAMILIES:
        raise SystemExit(f'holdout.jsonl carries {len(families)} eligible families with at least '
                         f'{HELDOUT_ROWS_PER_FAMILY} rows, need {HELDOUT_FAMILIES}')
    chosen = []
    for family in families[:HELDOUT_FAMILIES]:
        members = sorted((row for row in eligible if row['family_id'] == family),
                         key=lambda row: row['id'])
        chosen += members[:HELDOUT_ROWS_PER_FAMILY]
    return chosen


def overlap_checks(fit: list[dict], held: list[dict]) -> list[str]:
    """The claim the provenance makes: no prompt sits in two pools at once."""
    problems = []
    removal = {row['id'] for row in read_rows('verification/screens/advbench-40.jsonl')}
    fit_ids = {row['id'] for row in fit}
    held_ids = {row['id'] for row in held}
    if fit_ids & held_ids:
        problems.append(f'fit and held-out share {len(fit_ids & held_ids)} ids')
    if fit_ids & removal:
        problems.append(f'fit and harmful-40 share {len(fit_ids & removal)} ids')
    if held_ids & removal:
        problems.append(f'held-out and harmful-40 share {len(held_ids & removal)} ids')
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--write', action='store_true',
                        help='rewrite the two suite files (default: check only)')
    args = parser.parse_args(argv)
    fit, held = fit_suite(), heldout_suite()
    problems = overlap_checks(fit, held)
    if problems:
        for problem in problems:
            print(f'FAIL overlap: {problem}', file=sys.stderr)
        return 2
    failed = False
    for name, rows in (('fit-suite.jsonl', fit), ('heldout-24f.jsonl', held)):
        text = render(rows)
        got = digest(text)
        wanted = EXPECTED[name]
        state = 'PASS' if got == wanted else 'FAIL'
        if got != wanted:
            failed = True
        print(f'{state} {name}: {len(rows)} rows sha256 {got} (want {wanted})')
        if got == wanted and args.write:
            (HERE / name).write_text(text)
    families = sorted({row['family_id'] for row in held})
    print(f'held-out families: {len(families)}; fit rows: {len(fit)}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
