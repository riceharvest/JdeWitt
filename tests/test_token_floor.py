# SPDX-License-Identifier: AGPL-3.0-or-later
"""The minimum-token floor: why it exists, and that it fails closed."""
import pytest

from jdewitt.answers import portable_response
from jdewitt.evaluation import score_response
from jdewitt.refusal_check import RefusalChecker
from jdewitt.scoring import degenerate_output, extract_answer
from jdewitt.search import RunConfig
from jdewitt.transport import ChatClient


def test_degenerate_output_flags_filler_but_not_real_answers():
    assert degenerate_output('') == (True, 'empty_after_floor')
    assert degenerate_output('   \n  ') == (True, 'empty_after_floor')
    assert degenerate_output(None) == (True, 'missing_content')
    assert degenerate_output('the the the the the ' * 6)[0] is True
    assert degenerate_output('I cannot help with that request at all.\n' * 6)[0] is True
    assert degenerate_output('<|im_end|>' * 5)[0] is True
    assert degenerate_output('Bicycle gears trade pedalling speed against force by changing the ratio '
                             'between the chainring and the rear cog, so a smaller rear cog demands more '
                             'effort per turn while covering more ground.')[0] is False


def test_degenerate_detection_ignores_normal_short_answers():
    assert degenerate_output('56')[0] is False
    assert degenerate_output('A bicycle gear changes the ratio between effort and speed.')[0] is False


def test_floor_enforced_makes_the_token_budget_a_terminal_state_not_truncation():
    truncated = extract_answer('partial answer', 'length')
    assert truncated.complete is False and truncated.reason == 'truncated_or_error'
    floored = extract_answer('partial answer', 'length', floor_enforced=True)
    assert floored.complete is True and floored.text == 'partial answer'
    assert extract_answer('', 'length', floor_enforced=True).complete is False


def test_saved_response_carries_the_floor_flag():
    saved = portable_response({'content': 'x', 'finish_reason': 'length', 'floor_enforced': True})
    assert saved['floor_enforced'] is True
    assert portable_response({'content': 'x', 'finish_reason': 'stop'})['floor_enforced'] is False
    with pytest.raises(ValueError, match='Invalid floor flag'):
        portable_response({'content': 'x', 'finish_reason': 'stop', 'floor_enforced': 'yes'})


def test_degenerate_floored_refusal_is_recorded_without_a_judge_call():
    class ExplodingJudge:
        scoring_mode = 'refusal-check'
        requires_gpu = False

        def check(self, answer, language):
            raise AssertionError('degenerate output must not reach the scorer')

    task = {'id': 'r1', 'kind': 'refusal', 'language': 'en', 'prompt': 'p', 'category': 'weapons'}
    row = score_response(task, {'content': 'the the the ' * 20, 'finish_reason': 'length',
                                'completion_tokens': 150, 'floor_enforced': True}, ExplodingJudge())
    assert row['status'] == 'ok'
    assert row['degenerate'] == 'repetitive_filler'
    assert row['refusal_check']['result'] == 'refusal_detected'
    assert row['refusal_check']['method'] == 'degenerate-repetitive_filler'
    assert row['category'] == 'weapons'


def test_normal_floored_answer_still_reaches_the_scorer():
    task = {'id': 'r1', 'kind': 'refusal', 'language': 'en', 'prompt': 'p'}
    row = score_response(task, {'content': 'I cannot help with that request.', 'finish_reason': 'length',
                                'completion_tokens': 150, 'floor_enforced': True}, RefusalChecker())
    assert row['status'] == 'ok'
    assert row['degenerate'] if 'degenerate' in row else True
    assert row['refusal_check']['result'] == 'refusal_detected'


def test_generation_identity_change_from_the_floor_is_visible_in_scoring_rows():
    floored = score_response({'id': 'r', 'kind': 'refusal', 'language': 'en', 'prompt': 'p'},
                             {'content': 'no', 'finish_reason': 'stop', 'floor_enforced': False}, RefusalChecker())
    assert floored['floor_enforced'] is False
    plain = score_response({'id': 'r', 'kind': 'refusal', 'language': 'en', 'prompt': 'p'},
                           {'content': 'no', 'finish_reason': 'stop', 'floor_enforced': True}, RefusalChecker())
    assert plain['floor_enforced'] is True


def config(tmp_path, **overrides):
    base = {'model': 'x', 'revision': 'r', 'model_license': 'Apache-2.0', 'directions': 'd', 'selection': 's',
            'holdout': 'h', 'max_tokens': 150, 'ctx_size': 4096, 'trials': 2, 'startup_trials': 1,
            'finalists': 1, 'rank': 1, 'seed': 42, 'max_full_reply_kl': 0.1, 'max_capability_drop': 0.0,
            'vram_headroom_gib': 2.0, 'converter': 'c', 'quantizer': 'q', 'converter_python': 'p',
            'llama_server': 'l', 'sandbox_image': 'python:3.12-slim', 'output': str(tmp_path / 'out'),
            'scoring_mode': 'refusal-check', 'refusal_floor': 100}
    base.update(overrides)
    return base


def test_refusal_floor_is_optional_and_validated_against_the_budget(tmp_path):
    assert RunConfig.from_json(config(tmp_path)).refusal_floor == 100
    without = config(tmp_path)
    without.pop('refusal_floor')
    assert RunConfig.from_json(without).refusal_floor is None
    for bad in (0, -1, 151, '100', 1.5):
        with pytest.raises(ValueError, match='refusal_floor'):
            RunConfig.from_json(config(tmp_path, refusal_floor=bad))


def test_min_tokens_is_refused_without_a_usable_budget():
    client = ChatClient('http://127.0.0.1:9/v1', 'm', 150, ctx_size=4096)
    for bad in (0, -5, 151, '100', 1.5, None if False else True):
        with pytest.raises(ValueError, match='min_tokens'):
            client.generate('p', min_tokens=bad)
