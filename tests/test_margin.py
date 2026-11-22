"""Tests for jdewitt.margin — token resolution, margin math, projection hooks, capture wiring."""
from types import SimpleNamespace

import pytest
import torch

from jdewitt import margin
from jdewitt.capture import capture_one


class _Tokenizer:
    """Whitespace tokenizer with a fixed vocab; no transformers needed."""

    pad_token_id, eos_token_id, chat_template = 0, 2, None
    vocab = {"hello": 3, "world": 4, "here": 5, " Sure": 6, " Sure.": 7, " cannot": 8}

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([self._ids(text)], dtype=torch.long)}
        return {"input_ids": self._ids(text)}

    def decode(self, ids, skip_special_tokens=True):
        reverse = {value: key for key, value in self.vocab.items()}
        return " ".join(reverse.get(int(token), "hello") for token in ids)

    def _ids(self, text):
        return [self.vocab.get(word, 3) for word in text.split()] or [3]


class _Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, ids):
        # [1, seq, width] residual stream the capture hook expects
        return torch.zeros(1, ids.shape[1], self.width)


class _ToyModel(torch.nn.Module):
    def __init__(self, width=4, vocab=16, logits=None, emit=(6, 2), with_logits=True):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(width) for _ in range(2)])
        self.logits = torch.zeros(vocab) if logits is None else logits
        self.emit = list(emit)
        self.with_logits = with_logits
        self.forward_calls, self.generate_calls = 0, 0
        self.generation_config = SimpleNamespace(eos_token_id=2)

    def forward(self, ids, **kwargs):
        self.forward_calls += 1
        for block in self.blocks:
            block(ids)
        if not self.with_logits:
            return SimpleNamespace()
        return SimpleNamespace(logits=self.logits.reshape(1, 1, -1).expand(1, ids.shape[1], -1))

    def generate(self, ids, **kwargs):
        self.generate_calls += 1
        return torch.cat([ids, torch.tensor([self.emit], dtype=ids.dtype)], dim=1)


def _tokenizer_with(encode):
    class Tok(_Tokenizer):
        def encode(self, text, add_special_tokens=False):
            return encode(text)
    return Tok()


def test_token_table_lists_every_spelling_and_refuses_multitoken():
    def encode(text):
        return [1, 2] if text in (" Sure", "Sure") else [3]
    tok = _tokenizer_with(encode)
    table = margin.token_table(tok)
    assert [row["spelling"] for row in table["Sure"]] == list(margin.SURE_SPELLINGS)
    assert [row["resolved"] for row in table["Sure"]] == [False, False]
    assert table["cannot"][0]["resolved"] is True
    with pytest.raises(margin.MarginRefused):
        margin.resolved_pairs(table)


def test_pairs_are_cross_product_of_resolved_spellings():
    def encode(text):
        return [11] if text == " Sure" else [12] if text == "Sure" else [13]
    table = margin.token_table(_tokenizer_with(encode))
    assert margin.resolved_pairs(table) == [[11, 13], [12, 13]]


def test_margin_arithmetic_and_sign_flip():
    logits = [0.0, 2.0, 0.5, -1.0, 4.0, -2.0]
    assert margin.margin_from_logits(logits, [[1, 4]]) == -2.0
    assert margin.margin_from_logits(logits, [[1, 4], [2, 4]]) == -2.75
    assert margin.margin_from_logits(logits, [[4, 1]]) == 2.0


def test_margin_reads_one_position_of_every_shape():
    row = torch.arange(6, dtype=torch.float64)
    sequence = torch.stack([row, row * 2.0])
    assert margin.margin_at_position(sequence, 0, [[1, 4]]) == -3.0
    assert margin.margin_at_position(sequence, 1, [[1, 4]]) == -6.0
    assert margin.margin_at_position(sequence.unsqueeze(0), 0, [[1, 4]]) == -3.0
    assert margin.margin_at_position([[0.0, 2.0, 0.0, 0.0, -1.0]], 0, [[1, 4]]) == 3.0


def test_missing_logits_raise_margin_unavailable():
    with pytest.raises(margin.MarginUnavailable):
        margin.margin_at_position(None, 0, [[1, 4]])


def test_project_removes_the_axis_component():
    axis = [0.6, 0.8, 0.0, 0.0]
    vector = [1.0, 2.0, 3.0, 4.0]
    projected = margin.project(vector, axis)
    assert abs(sum(value * component for value, component in zip(projected, axis))) < 1e-12
    tensor = torch.tensor([vector])
    out = margin.project(tensor, torch.tensor([axis]))
    assert torch.allclose(out, torch.tensor([projected]))


def test_hooks_move_the_margin_and_are_removed():
    axis = [0.0, 1.0, 0.0, 0.0]
    layer = _Block(4)
    hooked = margin.install_projection_hooks([layer], [axis])
    try:
        assert len(layer._forward_hooks) == 1
    finally:
        margin.remove_hooks(hooked)
    assert not layer._forward_hooks
    assert hooked == []


def test_install_removes_earlier_handles_when_a_later_register_fails():
    class Sour(_Block):
        def register_forward_hook(self, hook):
            raise RuntimeError("register failed")
    first = _Block(4)
    with pytest.raises(RuntimeError):
        margin.install_projection_hooks([first, Sour(4)], [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    assert not first._forward_hooks


def _capture(model, tokenizer, pairs, row=None):
    row = row or {"id": "r", "kind": "harmless", "prompt": "hello world"}
    return capture_one(model, tokenizer, row, list(model.blocks), 4, locus="answer",
                       max_tokens=4, ctx_size=64, system="", raw=True, device="cpu",
                       settle=False, answer_text=" Sure cannot", margin_pairs=pairs)


def test_capture_reads_margin_from_the_existing_forward_only():
    logits = torch.zeros(16, dtype=torch.float64)
    logits[6], logits[8] = 3.0, -1.0
    model = _ToyModel(logits=logits, emit=(6, 2))
    (meta, residual), reason = _capture(model, _Tokenizer(), [[6, 8]])
    assert reason is None
    # the banked-answer path never generates, and the margin came out of the one
    # forward pass that also captured the residual stream: zero extra passes.
    assert model.forward_calls == 1 and model.generate_calls == 0
    assert meta["margin_position"] == meta["prompt_tokens"] - 1 == 1
    assert meta["margin_pairs"] == [[6, 8]]
    assert meta["margin"] == 4.0 and meta["margin_available"] is True
    assert residual.shape == (2, 4)


def test_capture_without_logits_records_null_margin_instead_of_failing():
    model = _ToyModel(with_logits=False, emit=(6, 2))
    (meta, _residual), reason = _capture(model, _Tokenizer(), [[6, 8]])
    assert reason is None
    assert meta["margin"] is None and meta["margin_available"] is False
    assert meta["margin_position"] is None


def test_capture_without_pairs_costs_nothing_and_stores_no_margin():
    model = _ToyModel(emit=(6, 2))
    (meta, _residual), reason = _capture(model, _Tokenizer(), [])
    assert reason is None
    assert meta["margin"] is None and meta["margin_pairs"] == []
