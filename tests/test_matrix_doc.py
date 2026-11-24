# SPDX-License-Identifier: AGPL-3.0-or-later
"""The evidence table is a render of the matrix JSON, and a loss cannot outlive its source."""
import json
from pathlib import Path

from jdewitt.matrix import _evidence_table, _ledger_problems

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / 'verification/releases/matrix.json'
LEDGER = ROOT / 'verification/releases/loss-ledger.json'


def matrix():
    return json.loads(MATRIX.read_text())


def test_render_is_deterministic_and_names_every_required_lineage():
    rendered = _evidence_table(matrix())
    assert _evidence_table(matrix()) == rendered

    order = [rendered.index(f'### {lineage} —')
             for lineage in ('inkling-2-mini', 'minicpm-v47', 'qwen25-1.5b',
                             'qwen3-4b', 'spark-1.7b')]
    assert order == sorted(order)


def test_the_losses_are_in_the_table_beside_the_reference():
    rendered = _evidence_table(matrix())
    # The qwen3-4b row is the pinned 4096/4 certification of the REFIT arm: it is a
    # loss, and the table says so beside the published reference in the same line.
    assert 'maths 158 -> 153    reference 159 -> 159' in rendered
    assert 'code 22 -> 18    reference 22 -> 19' in rendered
    assert 'median answer 156 chars vs base 155 (reference 170)' in rendered
    for text in ('6/40', '11/24', '34/120', '36/40', '~21/40'):
        assert text in rendered
    assert ('the fitted arm hands over the asked-for content on 5/24, the published build on '
            '4/24 and the unedited base on 1/24') in rendered


def test_unmeasured_cells_are_listed_and_not_counted():
    data = matrix()
    rendered = _evidence_table(data)
    assert f"certified (claims): {data['certified']} of {len(data['rows'])}" in rendered
    assert f"`complete`: {str(data['complete']).lower()}" in rendered
    unmeasured = [row['lineage'] for row in data['rows'] + data['spares']
                  if row['status'] == 'UNMEASURED']
    assert unmeasured
    section = rendered.split('## UNMEASURED', 1)[1]
    for lineage in unmeasured:
        assert f'| {lineage} |' in section


def test_ledger_refuses_a_number_whose_source_is_gone():
    ledger = json.loads(LEDGER.read_text())
    ledger['entries'][0]['numbers'][0]['source'] = 'verification/not-in-this-clone.jsonl'
    problems = _ledger_problems(ledger)
    assert any('source does not exist' in problem for problem in problems)


def test_ledger_refuses_a_value_that_no_longer_recomputes():
    ledger = json.loads(LEDGER.read_text())
    number = ledger['entries'][0]['numbers'][0]
    number['value'] = number['text'] = '7/40'
    ledger['entries'][0]['statement'] = ledger['entries'][0]['statement'].replace('6/40', '7/40')
    problems = _ledger_problems(ledger)
    assert any('no longer recomputes' in problem for problem in problems)
