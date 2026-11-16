import pytest
from b70_ablator import hf
from b70_ablator.scoring import extract_answer


class Tokens:
    def decode(self, ids, **kwargs):
        return ''.join(ids)


def test_explicit_adapter_boundary_ignores_quoted_closing_tag_in_final_answer():
    text = 'analysis</think>Explain the literal </think> tag.'
    assert hf.reply_boundary(Tokens(), list(text), True) == len('analysis</think>')


def test_adapter_contract_rejects_unknown_reasoning_format_before_search():
    with pytest.raises(ValueError, match='boundary adapter'):
        hf.BoundaryContract.for_architecture('future_model')


def test_plain_adapter_rejects_reasoning_output_instead_of_scoring_monologue():
    contract = hf.BoundaryContract.for_architecture('llama')
    with pytest.raises(ValueError, match='plain'):
        contract.boundary(Tokens(), list('<think>reason</think>answer'), False)


def test_thinking_adapter_proves_boundary_and_rejects_unfinished_thinking():
    contract = hf.BoundaryContract.for_architecture('qwen3')
    ids = list('<think>reason</think>answer')
    assert contract.boundary(Tokens(), ids, False) == len('<think>reason</think>')
    with pytest.raises(ValueError, match='completed'):
        contract.boundary(Tokens(), list('<think>reason'), False)
    assert contract.boundary(Tokens(), list('ordinary answer'), False) == 0


def test_nested_reasoning_blocks_fail_instead_of_moving_the_answer_boundary():
    contract = hf.BoundaryContract.for_architecture('qwen3')
    with pytest.raises(ValueError, match='nested'):
        contract.boundary(Tokens(), list('<think>one<think>two</think>answer'), False)
    answer = extract_answer('<think>one<think>two</think>answer', 'stop')
    assert not answer.complete


def test_scoring_preserves_literal_think_block_at_start_of_final_answer():
    answer = extract_answer('reason</think><think>literal example</think> explanation', 'stop', True)
    assert answer.complete and answer.text == '<think>literal example</think> explanation'
