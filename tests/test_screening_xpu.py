"""Actual B70 prefill-margin operator on a small fixture, NOT refusal-quality evidence."""
import math
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from jdewitt.editor import Editor
from jdewitt.hf import HFModel, require_b70
from jdewitt.pipeline import configure_compiler_paths

pytestmark = pytest.mark.xpu


class StubTokenizer:
    """Deterministic tiny tokenizer: only the render/encode surface screening needs."""

    def apply_chat_template(self, messages, **kwargs):
        return ' '.join(message['content'] for message in messages)

    def __call__(self, text, **kwargs):
        ids = [3 + min(len(word), 55) for word in text.split()]
        return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones(1, len(ids))}


def backend_on_b70(device):
    configure_compiler_paths()
    torch.manual_seed(7)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2)).to(device=device, dtype=torch.float16).eval()
    return HFModel.from_objects(model, StubTokenizer(), Editor(model, rank=1), max_tokens=8, ctx_size=64)


def directions(device):
    vectors = torch.zeros(2, 1, 32, device=device)
    vectors[:, :, 0] = 1
    return vectors


def params(strength):
    return {group: {'strength': strength, 'center': .5, 'width': 1., 'floor': 1.}
            for group in ('attn', 'mlp')}


PROMPTS = ['why did the chicken cross the road', 'list three colors', 'how does rain form']


def test_answer_start_margins_move_only_under_active_edits():
    device = require_b70()
    backend = backend_on_b70(device)
    try:
        baseline = backend.answer_start_margins(PROMPTS, [5, 9], [11, 13])
        assert len(baseline) == 3 and all(math.isfinite(value) for value in baseline)
        backend.editor.apply(directions(device), params(1.))
        edited = backend.answer_start_margins(PROMPTS, [5, 9], [11, 13])
        assert all(math.isfinite(value) for value in edited)
        assert max(abs(e - b) for e, b in zip(edited, baseline)) > 1e-6
        # A zero-strength apply leaves the operator disabled: margins must return exactly.
        backend.editor.apply(directions(device), params(0.))
        zeroed = backend.answer_start_margins(PROMPTS, [5, 9], [11, 13])
        assert all(abs(e - b) < 1e-6 for e, b in zip(zeroed, baseline))
        backend.editor.reset()
        restored = backend.answer_start_margins(PROMPTS, [5, 9], [11, 13])
        assert all(abs(e - b) < 1e-6 for e, b in zip(restored, baseline))
    finally:
        backend.release()


def test_answer_start_margins_reject_out_of_vocabulary_tokens():
    device = require_b70()
    backend = backend_on_b70(device)
    try:
        with pytest.raises(ValueError, match='outside the target vocabulary'):
            backend.answer_start_margins(PROMPTS, [200], [11])
    finally:
        backend.release()
