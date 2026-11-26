# SPDX-License-Identifier: AGPL-3.0-or-later
"""The compiled denial grammar: surfaces, length floor, and a usable fallback."""
import random

import pytest

from jdewitt.denial_grammar import (MIN_ANSWER_CHARS, DenialChecker, emit_gbnf, surface_catalogue,
                                    token_report)
from jdewitt.vocabremap import _byte_level

COMPLIANT = [
    'Here is a step-by-step guide to baking sourdough bread at home.',
    'The quickest way to debug that script is to print the value before the loop.',
    'Hier is een stapsgewijze uitleg om zelf zuurdesembrood te bakken.',
    'Eerst installeer je het pakket en daarna voer je de migratie uit.',
    'Hier ist eine Schritt-fuer-Schritt-Anleitung, um zu Hause Sauerteigbrot zu backen.',
    'Oeffne die Einstellungen und lege ein neues Profil an, um das VPN einzurichten.',
]


@pytest.fixture(scope='module')
def checker():
    return DenialChecker()


def test_table_has_twenty_surfaces_in_three_languages(checker):
    rows = checker.catalogue['rows']
    assert len(rows) == 20
    assert sorted({row['language'] for row in rows}) == ['de', 'en', 'nl']
    assert len({row['id'] for row in rows}) == 20


def test_every_surface_is_rejected_in_every_spelling(checker):
    for row in checker.catalogue['rows']:
        for variant in row['variants']:
            result = checker.scan(variant)
            assert result['ok'] is False, (row['id'], variant)
            assert row['id'] in {hit['id'] for hit in result['surfaces']}


def test_case_apostrophe_and_whitespace_do_not_route_around_the_automaton(checker):
    for text, surface in [('I CanNot help with that.', 'en-cannot'),
                          ('I CANNOT help with that.', 'en-cannot'),
                          ('I can\u2019t help with that.', 'en-cant'),
                          ('I can\u2018t help with that.', 'en-cant'),
                          ('I can   not help with that.', 'en-can-not'),
                          ('I can\tnot help with that.', 'en-can-not'),
                          ('I can\nnot help with that.', 'en-can-not'),
                          ('ich\u00a0kann\u00a0nicht helfen.', 'de-ich-kann-nicht'),
                          ('Es tut mir Leid.', 'de-es-tut-mir-leid'),
                          ('NICHT M\u00d6GLICH.', 'de-nicht-moeglich'),
                          ('Ik kan  ik  niet helpen.', 'nl-kan-ik-niet')]:
        result = checker.scan(text)
        assert result['ok'] is False, text
        assert surface in {hit['id'] for hit in result['surfaces']}, text


def test_a_surface_inside_a_longer_sentence_is_still_rejected(checker):
    text = 'The guide is clear, but I cannot recommend running it without a backup.'
    assert checker.scan(text)['ok'] is False


def test_compliant_openings_are_accepted(checker):
    for text in COMPLIANT:
        assert checker.scan(text)['ok'] is True, text
        assert checker.opening(text) is True


def test_the_length_floor_is_an_end_of_sequence_rule_not_an_opening_rule(checker):
    short = 'A gear changes the ratio of effort to speed.'
    assert len(short) < MIN_ANSWER_CHARS
    assert checker.opening(short) is True
    result = checker.complete(short)
    assert result['ok'] is False and result['reason'] == 'below_min_chars'
    padded = short + ' ' + 'x' * (MIN_ANSWER_CHARS - len(short))
    assert checker.complete(padded)['ok'] is True
    assert checker.complete('')['reason'] == 'below_min_chars'


def test_exactly_min_chars_is_accepted_and_one_less_is_not(checker):
    assert checker.complete('x' * MIN_ANSWER_CHARS)['ok'] is True
    assert checker.complete('x' * (MIN_ANSWER_CHARS - 1))['reason'] == 'below_min_chars'


def test_a_denial_is_rejected_even_when_it_is_long_enough(checker):
    text = ('I cannot assist with that request. ' + 'ordinary words follow this sentence ' * 4)
    assert len(text) >= MIN_ANSWER_CHARS
    result = checker.complete(text)
    assert result['ok'] is False and result['reason'] == 'denial_surface'


def test_the_prefix_before_the_completing_character_is_allowed(checker):
    # The compilation must not ban "can" itself: it bans the character that
    # would complete "cannot".  This is the fallback the report names.
    assert checker.opening('I can') is True
    assert checker.opening('I can help you') is True
    assert checker.opening('I cann') is True
    assert checker.opening('I canno') is True
    assert checker.opening('I cannot') is False
    forced = checker.forced_continuations('canno')
    assert {item['symbol'] for item in forced['forbidden']} == {'t', 'T'}
    assert checker.forced_continuations('can no')['forbidden'][0]['surfaces'] == ['en-can-not']


def test_the_literal_no_pending_prefix_reading_is_an_alphabet_dead_end():
    live = DenialChecker(mode='live_prefix')
    alphabet = live.live_prefix_alphabet()
    assert set(alphabet['banned']) == {'C', 'E', 'H', 'I', 'K', 'N', 'U', 'W',
                                       'c', 'e', 'h', 'i', 'k', 'n', 'u', 'w'}
    # No authored opening survives: every one contains one of those letters.
    assert all(live.opening(text) is False for text in COMPLIANT)


def test_the_minimal_automaton_and_the_raw_one_agree_on_random_text(checker):
    rng = random.Random(20261126)
    alphabet = list('canotwilbeusrpgmk hd\'.,\u2019\u00f6\u00fc\n')
    for _ in range(400):
        text = ''.join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
        assert checker.minimal_agrees(text), repr(text)


def test_an_independent_substring_oracle_agrees_with_the_automaton(checker):
    forms = [(' '.join(row['form'].split()).casefold(), row['id']) for row in checker.catalogue['rows']]
    rng = random.Random(7)
    for _ in range(300):
        text = ''.join(rng.choice(list('abcnokwilteusrpgmh \'\u2019\u00f6')) for _ in range(rng.randint(0, 30)))
        normalized = ' '.join(text.casefold().replace('\u2019', "'").split())
        oracle = [surface_id for form, surface_id in forms if form in normalized]
        assert (not oracle) == checker.scan(text)['ok'], repr(text)


def test_grammar_defines_every_rule_it_references(checker):
    grammar = checker.grammar()
    rules = {}
    for line in grammar.splitlines():
        if not line or line.startswith('#') or '::=' not in line:
            continue
        name, body = line.split('::=', 1)
        rules[name.strip()] = body
    assert 'root' in rules
    referenced = set()
    for body in rules.values():
        for token in body.replace('|', ' ').split():
            if token.startswith('""'):
                continue
            if token.isalnum() and token in rules:
                referenced.add(token)
    assert referenced <= set(rules), sorted(referenced - set(rules))
    assert rules['root'].strip() == 's0t0'


def test_emit_gbnf_rejects_a_zero_floor(checker):
    with pytest.raises(ValueError):
        emit_gbnf(checker.minimal, checker.symbols, 0)


def test_token_view_counts_pieces_that_carry_a_whole_surface_and_ones_that_straddle(checker):
    pieces = [_byte_level(text) for text in [' cannot', ' can', 'not', ' canno', ' the', 'Cannot', ' unable']]
    report = token_report(checker.automaton, pieces)
    assert report['pieces'] == 7
    # " cannot", "Cannot" and " unable" carry a whole surface inside one piece.
    assert report['pieces_forbidden_at_start'] == 3
    assert set(report['examples_forbidden_at_start']) == {' cannot', 'Cannot', ' unable'}
    # " can" and " canno" end inside a pending English denial and "not" inside
    # "not able"; " the" ends inside "het spijt me".  A whole-token blocklist
    # cannot see any of that without tracking characters across token bounds.
    assert report['pieces_ending_inside_a_partial_match'] == 4
    assert report['states_with_no_legal_piece'] == []
    assert report['pieces_forbidden_from_every_state'] == 3
    assert report['allowed_pieces_per_state']['min'] < 7


def test_the_table_is_the_only_surface_source(checker):
    catalogue = surface_catalogue()
    assert len(catalogue['rows']) == len(checker.catalogue['rows'])
    assert all(row['variants'] for row in checker.catalogue['rows'])
