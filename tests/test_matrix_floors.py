# SPDX-License-Identifier: AGPL-3.0-or-later
"""A row's floors come from the manifest, but a lineage may register its own.

The manifest-wide floor is what every row is checked against by default. A
lineage whose pinned config serves 1024 tokens per slot, or whose removal screen
is its own file, registers that floor beside its artifact: the override replaces
only the axes it names, and the default note survives so the departure is visible.
"""
from jdewitt.matrix import _count_problems, _floors

REGISTERED = {
    'suite_rows': {'removal': 40, 'held_out': 120, 'retention': 241, 'note': 'default'},
    'cap': {'removal': 4096, 'held_out': 4096, 'retention': 1024},
    'suites': {'removal': 'verification/p7-q3-4b/harmful-40.jsonl',
               'held_out': 'verification/screens/q3-holdout-wider.jsonl',
               'retention': 'verification/capability/capability-run-en.jsonl'},
}

ENTRY = {
    'suite_rows': {'held_out': 120},
    'cap': {'removal': 1024, 'held_out': 1024},
    'suites': {'removal': 'verification/screens/advbench-40.jsonl',
               'held_out': 'verification/p7-minicpm/heldout-24f.jsonl'},
}


def packet(cap=1024, removal_suite='verification/screens/advbench-40.jsonl'):
    return {
        'removal': {'harmful40': {'inventories': {'shipped': {'total': 40},
                                                  'wide': {'total': 40}}},
                    'held_out_120': {'inventories': {'shipped': {'total': 120},
                                                     'wide': {'total': 120}}}},
        'retention': {'math': {'strict': {'all_rows': {'before': '159/180'}}},
                      'code': {'strict': {'all_rows': {'before': '22/61'}}}},
        'measured_axes': {'removal': {'rows': 40, 'cap': cap, 'suite': removal_suite},
                          'held_out': {'rows': 120, 'cap': cap,
                                       'suite': 'verification/p7-minicpm/heldout-24f.jsonl'},
                          'retention': {'rows': 241, 'cap': 1024}},
    }


def codes(problems):
    return [problem['code'] for problem in problems]


def test_the_override_replaces_only_the_axes_it_names():
    floors = _floors(ENTRY, REGISTERED)
    assert floors['cap'] == {'removal': 1024, 'held_out': 1024, 'retention': 1024}
    assert floors['suites']['removal'] == 'verification/screens/advbench-40.jsonl'
    assert floors['suites']['retention'] == REGISTERED['suites']['retention']
    assert floors['suite_rows'] == REGISTERED['suite_rows']


def test_a_row_with_its_own_floors_is_graded_at_its_pinned_config():
    # 1024 is one slot of -c 4096 --parallel 4; advbench-40 is this lineage's screen.
    assert _count_problems(ENTRY, packet(), _floors(ENTRY, REGISTERED)) == []


def test_without_the_override_the_same_packet_is_refused_as_lowered_and_substituted():
    problems = _count_problems(ENTRY, packet(), REGISTERED)
    assert 'cap_lowered' in codes(problems)
    assert 'suite_substituted' in codes(problems)


def test_an_override_cannot_narrow_the_deck():
    narrowed = packet() | {
        'removal': {'harmful40': {'inventories': {'shipped': {'total': 38},
                                                  'wide': {'total': 38}}},
                    'held_out_120': {'inventories': {'shipped': {'total': 120},
                                                     'wide': {'total': 120}}}},
        'measured_axes': {'removal': {'rows': 38, 'cap': 1024,
                                      'suite': 'verification/screens/advbench-40.jsonl'},
                          'held_out': {'rows': 120, 'cap': 1024,
                                       'suite': 'verification/p7-minicpm/heldout-24f.jsonl'},
                          'retention': {'rows': 241, 'cap': 1024}}}
    assert 'suite_narrowed' in codes(_count_problems(ENTRY, narrowed, _floors(ENTRY, REGISTERED)))
