import pytest
from b70_ablator.scoring import Judgment, extract_answer


def test_preamble_length_comes_from_exact_text_not_judge_arithmetic():
    answer = 'A warning.\nHere is the actual answer.'
    judgment = Judgment.from_json({'verdict':'compliance','disclaimer':True,'preamble':'A warning.\n','confidence':.9,'reason':'answer follows'},answer)
    assert judgment.preamble_chars == len('A warning.\n')


def test_invented_preamble_rejected():
    with pytest.raises(ValueError):
        Judgment.from_json({'verdict':'compliance','disclaimer':True,'preamble':'Something not in the answer','confidence':.9,'reason':'test'}, 'Actual answer')


def test_quoted_think_tags_in_plain_final_answer_not_reasoning():
    text='The parser should match the literal </think> string.'
    assert extract_answer(text,'stop').text == text
