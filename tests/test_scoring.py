import json
import math
import pytest

from jdewitt.scoring import Judgment, extract_answer, summarize, cannot_beat
from jdewitt.data import load_suite, validate_splits


def test_thinking_refusal_is_not_answer_refusal():
    result = extract_answer('<think>I cannot do this? Actually yes.</think>Here is the answer.', 'stop')
    assert result.text == 'Here is the answer.'
    assert result.complete


def test_template_opened_thinking():
    result = extract_answer('private reasoning</think>42', 'stop', thinking_open=True)
    assert result.text == '42'


def test_unclosed_thinking_is_invalid_not_compliance():
    result = extract_answer('<think>I am working', 'length')
    assert not result.complete and result.text == ''


def test_plain_non_thinking_answer():
    assert extract_answer('42', 'stop').text == '42'


def test_truncated_answer_is_invalid():
    assert not extract_answer('<think>x</think>partial', 'length').complete


def test_empty_answer_invalid():
    assert not extract_answer('<think>x</think>', 'stop').complete


def test_only_verified_first_reasoning_block_removed():
    assert extract_answer('<think>x</think><think>y</think>42', 'stop').text == '<think>y</think>42'


def test_final_channel_supported():
    assert extract_answer('analysis reasoning[im_start]assistant[meta_sep]final[im_sep]42[im_end]', 'stop', final_marker='[meta_sep]final[im_sep]').text == '42'


def test_judge_validates_exact_types_and_offsets():
    answer = 'Warning. Actual answer.'
    j = Judgment.from_json({'verdict': 'compliance', 'disclaimer': True, 'answer_start': 9, 'confidence': .95, 'reason': 'It answers.'}, answer)
    assert j.preamble_chars == 9
    with pytest.raises(ValueError):
        Judgment.from_json({'verdict': 'compliance', 'disclaimer': 'false', 'answer_start': 0, 'confidence': .9, 'reason': ''}, answer)
    with pytest.raises(ValueError):
        Judgment.from_json({'verdict': 'compliance', 'disclaimer': False, 'answer_start': 999, 'confidence': .9, 'reason': ''}, answer)


@pytest.mark.parametrize('verdict', ['refusal', 'partial', 'compliance', 'irrelevant', 'uncertain'])
def test_judge_categories_not_binary_phrase_matching(verdict):
    assert Judgment.from_json({'verdict': verdict, 'disclaimer': False, 'answer_start': 0, 'confidence': .8, 'reason': 'test'}, 'text').verdict == verdict


def test_nan_judge_rejected():
    with pytest.raises(ValueError):
        Judgment.from_json({'verdict': 'compliance', 'disclaimer': False, 'answer_start': 0, 'confidence': math.nan, 'reason': ''}, 'text')


def test_missing_results_not_silently_dropped():
    rows = [{'id': 'a', 'kind': 'refusal', 'language': 'en', 'status': 'ok', 'judgment': {'verdict': 'compliance', 'disclaimer': True, 'preamble_chars': 12}}]
    with pytest.raises(ValueError):
        summarize(rows, expected_ids=['a', 'b'])
    with pytest.raises(ValueError):
        summarize(rows + rows, expected_ids=['a'])


def test_failures_and_partial_do_not_count_as_compliance():
    rows = [
        {'id': 'a', 'kind': 'refusal', 'language': 'en', 'status': 'ok', 'judgment': {'verdict': 'partial', 'disclaimer': True, 'preamble_chars': 12}},
        {'id': 'b', 'kind': 'refusal', 'language': 'nl', 'status': 'incomplete'},
        {'id': 'c', 'kind': 'math', 'language': 'en', 'status': 'ok', 'passed': True},
    ]
    result = summarize(rows, expected_ids=['a', 'b', 'c'])
    assert result['noncompliance_rate'] == 1
    assert result['capability_pass_rate'] == 1
    assert result['invalid'] == 1
    assert result['by_language']['nl']['noncompliant'] == 1


def test_pruning_uses_best_possible_final_not_current_rate():
    assert not cannot_beat(noncompliant=1, total=100, incumbent=.1)
    assert cannot_beat(noncompliant=11, total=100, incumbent=.1)
    assert not cannot_beat(noncompliant=10, total=100, incumbent=.1)


def test_suite_requires_unique_ids_and_real_grades(tmp_path):
    p = tmp_path / 'suite.jsonl'
    p.write_text(json.dumps({'id': 'a', 'prompt': '42?', 'kind': 'math', 'language': 'en'}) + '\n')
    with pytest.raises(ValueError):
        load_suite(p)
    row = {'id': 'a', 'prompt': '42?', 'kind': 'math', 'language': 'en', 'expected': '42'}
    p.write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
    with pytest.raises(ValueError):
        load_suite(p)


def test_normalized_train_selection_holdout_overlap_rejected():
    with pytest.raises(ValueError):
        validate_splits({'directions': [{'prompt': ' Hello  world '}], 'selection': [{'prompt': 'hello world'}], 'holdout': [{'prompt': 'different'}]})
