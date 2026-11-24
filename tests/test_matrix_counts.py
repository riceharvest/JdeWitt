# SPDX-License-Identifier: AGPL-3.0-or-later
"""A wave asks N rows and grades the ones that answered; the floor check must tell
those apart from a suite that was quietly trimmed.

At the pinned -c 4096 --parallel 4 config one slot is 1024 tokens, so long answers
truncate and the graded denominator is smaller than the deck. release-certify now
carries that difference as `ungraded`, and the matrix reconciles `asked = graded +
ungraded` instead of calling the count ambiguous -- without letting a packet that
simply reports fewer rows slip under the registered floor.
"""
from jdewitt.matrix import _count_problems, _packet_counts

REGISTERED = {'suite_rows': {'removal': 40, 'held_out': 120, 'retention': 241}}


def packet(graded=38, ungraded=None, asked=40, wide=None):
    removal = {
        'inventories': {'shipped': {'total': graded},
                        'wide': {'total': graded if wide is None else wide}},
        'families': 20,
    }
    if ungraded is not None:
        removal['ungraded'] = {'rows': ungraded, 'of': asked}
    return {
        'removal': {'harmful40': removal,
                    'held_out_120': {'inventories': {'shipped': {'total': 120},
                                                     'wide': {'total': 120}}}},
        'retention': {'math': {'strict': {'all_rows': {'before': '159/180'}}},
                      'code': {'strict': {'all_rows': {'before': '22/61'}}}},
        'measured_axes': {'removal': {'rows': asked},
                          'held_out': {'rows': 120},
                          'retention': {'rows': 241}},
    }


def codes(problems):
    return [problem['code'] for problem in problems]


def test_named_ungraded_rows_reconcile_the_denominator_without_narrowing_the_suite():
    counts = _packet_counts(packet(graded=38, ungraded=2, asked=40))
    assert counts['removal']['rows'] == 38
    assert counts['removal']['ungraded'] == 2
    assert counts['removal']['disagree'] is False
    assert _count_problems('removal', packet(graded=38, ungraded=2, asked=40),
                           REGISTERED) == []


def test_a_trimmed_inventory_is_still_below_the_floor():
    # No `ungraded` block: the packet simply reports 38 rows where the deck has 40.
    problems = _count_problems('removal', packet(graded=38, asked=38), REGISTERED)
    assert 'suite_narrowed' in codes(problems)


def test_asked_rows_that_do_not_add_up_are_an_ambiguity_not_a_pass():
    counts = _packet_counts(packet(graded=38, ungraded=0, asked=40))
    assert counts['removal']['disagree'] is True
    assert 'row_count_disagreement' in codes(
        _count_problems('removal', packet(graded=38, ungraded=0, asked=40), REGISTERED))


def test_the_two_inventories_must_agree_even_when_ungraded_rows_are_named():
    counts = _packet_counts(packet(graded=38, ungraded=2, asked=40, wide=36))
    assert counts['removal']['disagree'] is True
