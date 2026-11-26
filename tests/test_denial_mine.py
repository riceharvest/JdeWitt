# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mining the denial table from saved answers: rules, adoption, guarded spellings."""
import json
from pathlib import Path

import pytest

from jdewitt.denial_grammar import (DenialChecker, SURFACES_SCHEMA, surface_elisions)
from jdewitt.denial_mine import (adopt, candidate_surfaces, fold,
                                 is_compliant, is_refusal, measure_candidates, mine_table,
                                 normalize, opening_window)


def make_row(row_id, answer, narrow='refusal_detected', wide=None, file='arm/predictions.jsonl',
             line=1):
    return {'row_id': row_id, 'file': file, 'line': line, 'answer': answer,
            'folded_answer': fold(normalize(answer)),
            'opening': opening_window(answer),
            'folded_opening': fold(opening_window(answer)),
            'refusal_check': narrow, 'check_method': 'explicit-wording-v1',
            'wide_check': wide, 'wide_evidence': None, 'language': 'en'}


def seed_document(entries):
    return {'schema': 'jdewitt.vocab-remap-denial/1', 'entries': entries}


def entry(row_id, form, kind='refusal_opener', scope='opener_window'):
    return {'id': row_id, 'language': 'en', 'form': form, 'kind': kind, 'scope': scope,
            'replacement': None, 'note': 'test row'}


# --------------------------------------------------------------------------
# Rules are literal: a match is the surface, and the surface is the text
# --------------------------------------------------------------------------

def test_a_rule_extends_the_stem_with_the_words_the_answer_uses():
    found = candidate_surfaces('I refuse to create or promote content that incites violence.')
    forms = {item['form'] for item in found}
    assert 'I refuse to create' in forms
    assert all(item['rule']['id'] == 'refusal-plain' for item in found)


def test_an_apology_rule_keeps_the_intensifier_that_the_answer_uses():
    forms = {item['form'] for item in candidate_surfaces("I'm really sorry, but it is important.")}
    assert "I'm really sorry" in forms


def test_the_extension_stops_at_the_object_so_the_surface_generalises():
    forms = {item['form'] for item in candidate_surfaces('I cannot provide instructions for that.')}
    assert 'I cannot provide instructions' in forms or 'I cannot provide' in forms
    assert not any(form.endswith('for') for form in forms)


def test_compliant_prose_yields_no_candidate_from_the_shape_rules():
    assert candidate_surfaces('Here is a step-by-step guide to baking sourdough bread at home.') == []


def test_the_elided_spellings_are_generated_for_every_apostrophe_surface():
    assert surface_elisions("can't") == ['cant']
    assert surface_elisions("I'm sorry") == ['Im sorry']


# --------------------------------------------------------------------------
# Adoption: gain is refusal coverage, cost is compliant regression
# --------------------------------------------------------------------------

def test_adoption_takes_a_shape_that_closes_an_opener_and_costs_nothing():
    rows = [make_row('r1', 'I refuse to create that. ' + 'x' * 60),
            make_row('c1', 'Here is how to bake bread at home. ' + 'x' * 60,
                     narrow='no_explicit_refusal')]
    candidates = measure_candidates(rows)
    adopted, excluded, covered, blocked = adopt(candidates, [], rows)
    forms = [record['form'] for record in adopted]
    assert 'I refuse to create' in forms
    assert covered == 1 and blocked == 0


def test_adoption_rejects_a_frame_even_when_it_closes_openers():
    rows = [make_row('r1', "It's important to clarify that stealing is illegal. " + 'x' * 60)]
    candidates = measure_candidates(rows)
    adopted, excluded, _covered, _blocked = adopt(candidates, [], rows)
    assert adopted == []
    frame = [record for record in excluded if record['kind'] == 'refusal_frame']
    assert frame and 'frame, not a denial' in frame[0]['reason']


def test_adoption_measures_the_compliant_rows_a_shape_would_block():
    rows = [make_row('r1', 'I do not endorse that at all. ' + 'x' * 60),
            make_row('c1', 'I do not endorse a specific library here. ' + 'x' * 60,
                     narrow='no_explicit_refusal')]
    candidates = measure_candidates(rows)
    adopted, excluded, _covered, _blocked = adopt(candidates, [], rows)
    record = (adopted + excluded)[0]
    assert record['marginal_cost'] >= 1
    assert record['evidence'] and record['evidence'][0]['row_id'] == 'r1'


def test_the_taxonomy_of_a_row_is_the_union_of_both_checkers():
    narrow_compliant = make_row('a', 'text', narrow='no_explicit_refusal')
    wide_refusal = make_row('b', 'text', narrow='no_explicit_refusal', wide='refusal_detected')
    clean = make_row('c', 'text', narrow='no_explicit_refusal', wide='no_explicit_refusal')
    assert is_refusal(wide_refusal) and not is_refusal(narrow_compliant)
    assert is_compliant(clean) and not is_compliant(wide_refusal)


def test_the_emitted_table_carries_one_row_per_surface_with_its_evidence_ids(tmp_path):
    path = tmp_path / 'predictions.jsonl'
    rows = [{'id': 'advbench-1', 'answer': 'I refuse to create that. ' + 'x' * 60,
             'language': 'en', 'refusal_check': {'result': 'refusal_detected',
                                                 'method': 'explicit-wording-v1'}},
            {'id': 'other-1', 'answer': 'Here is how to bake bread. ' + 'x' * 60,
             'language': 'en', 'refusal_check': {'result': 'no_explicit_refusal'}}]
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    document, report = mine_table([path], seed_document=seed_document([entry('en-cannot', 'cannot')]))
    assert document['schema'] == SURFACES_SCHEMA
    mined = [row for row in document['entries'] if row['source'] == 'mined']
    assert mined and mined[0]['evidence'][0]['row_id'] == 'advbench-1'
    assert mined[0]['form'] == 'I refuse to create'
    assert report['candidates']['adopted'] == len(mined)


# --------------------------------------------------------------------------
# Guarded spellings: a word-start guard, not a substring
# --------------------------------------------------------------------------

def test_an_elided_spelling_is_rejected_at_a_word_start_and_ignored_inside_a_word():
    document = {'schema': SURFACES_SCHEMA, 'entries': [
        {'id': 'en-cant', 'language': 'en', 'form': "can't", 'kind': 'denial_word',
         'scope': 'any', 'replacement': None, 'note': '', 'guarded_forms': ['cant']}]}
    checker = DenialChecker(table=document)
    assert checker.scan('I cant help with that.')['ok'] is False
    assert checker.scan('cant help with that.')['ok'] is False
    for text in ('significant results', 'a scant chance', 'the applicants were many',
                 'he decanted the wine', 'scanty evidence'):
        assert checker.scan(text)['ok'] is True, text


def test_without_the_guard_the_same_spelling_blocks_ordinary_words():
    document = {'schema': SURFACES_SCHEMA, 'entries': [
        {'id': 'en-cant', 'language': 'en', 'form': 'cant', 'kind': 'denial_word',
         'scope': 'any', 'replacement': None, 'note': '', 'guarded_forms': []}]}
    checker = DenialChecker(table=document)
    assert checker.scan('significant results')['ok'] is False


def test_the_guard_does_not_weaken_the_surface_it_belongs_to():
    document = {'schema': SURFACES_SCHEMA, 'entries': [
        {'id': 'en-cant', 'language': 'en', 'form': "can't", 'kind': 'denial_word',
         'scope': 'any', 'replacement': None, 'note': '', 'guarded_forms': ['cant']},
        {'id': 'en-cannot', 'language': 'en', 'form': 'cannot', 'kind': 'denial_word',
         'scope': 'any', 'replacement': None, 'note': '', 'guarded_forms': []}]}
    checker = DenialChecker(table=document)
    for text in ("I can't do that.", 'I can\u2019t do that.', 'I cannot do that.'):
        assert checker.scan(text)['ok'] is False, text


# --------------------------------------------------------------------------
# The seed compilation is unchanged by the machinery
# --------------------------------------------------------------------------

def test_the_seed_table_still_compiles_to_the_previous_artifact():
    checker = DenialChecker()
    grammar = checker.grammar()
    assert len(checker.minimal['transitions']) == 103
    assert len(checker.symbols) == 54
    assert len(grammar.splitlines()) and grammar.count('::=') == 5048
    assert len(grammar.encode()) == 862128
    assert all(not row['guarded_forms'] for row in checker.catalogue['rows'])


def test_the_mined_asset_in_the_repository_compiles_and_is_fully_rejected():
    path = Path('verification/denial-grammar/denial-surfaces.json')
    if not path.exists():
        pytest.skip('the mined table asset has not been generated in this tree')
    document = json.loads(path.read_text())
    assert document['schema'] == SURFACES_SCHEMA
    assert all(entry.get('evidence') is not None for entry in document['entries'])
    checker = DenialChecker(table=path)
    assert len(checker.catalogue['rows']) == len(document['entries'])
    for row in checker.catalogue['rows']:
        for spelling in list(row['variants']) + list(row['guarded_forms']):
            assert checker.scan(spelling)['ok'] is False, (row['id'], spelling)
