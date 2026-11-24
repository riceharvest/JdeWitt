# SPDX-License-Identifier: AGPL-3.0-or-later
"""The input-side remap planner: token-exact, longest-first, and honest about exposure."""
from __future__ import annotations

import pytest

from jdewitt.vocabremap import _byte_level, load_table
from jdewitt.vocabremap_embed import load_affirmative, plan_remap


def _items(*specs):
    """Build planner items without a server: (entry_id, denial_text, denial_ids, affirmative_text, affirmative_ids)."""
    items = []
    for entry_id, denial_text, denial_ids, affirmative_text, affirmative_ids in specs:
        items.append({'entry': {'id': entry_id, 'language': 'en', 'form': denial_text.strip(), 'scope': 'any'},
                      'affirmative': affirmative_text.strip(),
                      'contexts': [{'context': 'leads_with_space', 'denial_text': denial_text,
                                    'denial_ids': denial_ids, 'affirmative_text': affirmative_text,
                                    'affirmative_ids': affirmative_ids}]})
    return items


def _tokens(pieces):
    return [_byte_level(piece) for piece in pieces]


def test_fused_denial_is_one_id_for_one_id():
    tokens = _tokens(['I', ' cannot', ' get', ' in'])
    items = _items(('en-cannot', ' cannot', [1], ' can', [2]))
    out, edits, unmatched = plan_remap([0, 1, 2, 3], tokens, items)
    assert out == [0, 2, 2, 3]
    assert unmatched == []
    assert edits == [{'entry_id': 'en-cannot', 'language': 'en', 'form': 'cannot', 'context': 'leads_with_space',
                      'start': 1, 'after_start': 1, 'matched_ids': [1], 'matched_text': ' cannot',
                      'replacement_ids': [2], 'replacement_text': ' can', 'affirmative': 'can'}]


def test_residual_negative_is_dropped_not_kept_as_its_own_token():
    tokens = _tokens(['I', ' can', ' not', ' go'])
    items = _items(('en-can-not', ' can not', [1, 2], ' can', [1]))
    out, edits, unmatched = plan_remap([0, 1, 2, 3], tokens, items)
    assert out == [0, 1, 3]
    assert unmatched == []
    assert edits[0]['matched_ids'] == [1, 2]
    assert edits[0]['replacement_ids'] == [1]


def test_longest_match_wins():
    tokens = _tokens(['ik', ' kan', ' ik', ' niet', ' doen'])
    items = _items(('nl-kan-niet', ' kan niet', [1, 3], ' kan', [1]),
                   ('nl-kan-ik-niet', ' kan ik niet', [1, 2, 3], ' kan ik', [1, 2]))
    out, edits, unmatched = plan_remap([0, 1, 2, 3, 4], tokens, items)
    assert out == [0, 1, 2, 4]
    assert [edit['entry_id'] for edit in edits] == ['nl-kan-ik-niet']
    assert unmatched == ['nl-kan-niet']


def test_unanchored_text_guard_rejects_a_sequence_that_spells_another_word():
    tokens = _tokens(['x', ' cannot', ' y'])
    items = _items(('en-cannot', ' cannot', [1], ' can', [2]))
    items[0]['contexts'][0]['denial_text'] = ' something else'
    out, edits, unmatched = plan_remap([0, 1, 2], tokens, items)
    assert out == [0, 1, 2]
    assert edits == []
    assert unmatched == ['en-cannot']


def test_every_authored_entry_maps_an_id_in_the_checked_in_table():
    table, _raw = load_table()
    affirmative, _raw_aff = load_affirmative()
    ids = {entry['id'] for entry in table['entries']}
    mapped = {entry['id'] for entry in affirmative['entries']}
    unmapped = {entry['id'] for entry in affirmative['unmapped']}
    assert mapped | unmapped == ids, 'every table entry is either mapped or named as unmapped'
    assert not mapped & unmapped
    assert all(entry['affirmative'].strip() for entry in affirmative['entries'])


def test_affirmative_loader_rejects_an_unknown_id(tmp_path):
    import json

    from jdewitt.vocabremap_embed import AFFIRMATIVE_SCHEMA, affirmative_items
    path = tmp_path / 'affirmative.json'
    path.write_text(json.dumps({'schema': AFFIRMATIVE_SCHEMA,
                                'entries': [{'id': 'not-a-table-entry', 'affirmative': 'can'}]}))
    affirmative_document, _raw = load_affirmative(path)
    table, _raw_table = load_table()
    with pytest.raises(ValueError):
        affirmative_items(table['entries'], affirmative_document, port=None)
