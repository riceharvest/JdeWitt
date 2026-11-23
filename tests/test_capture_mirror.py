"""Tests for the delivery-mirror capture path (--mirror-answers)."""
from types import SimpleNamespace

import pytest
import torch

from jdewitt.capture import capture_one, mirror_frame, read_mirror_answers
from jdewitt.frames import render_frame


class _Tokenizer:
    pad_token_id, eos_token_id, chat_template = 0, 2, None
    vocab = {"hello": 3, "world": 4, "Sure": 5, "here": 6, "is": 7, "how": 8}

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
        return torch.zeros(1, ids.shape[1], self.width)


class _ToyModel(torch.nn.Module):
    def __init__(self, width=4, vocab=16):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(width) for _ in range(2)])
        self.logits = torch.zeros(vocab)
        self.generate_calls = 0
        self.generation_config = SimpleNamespace(eos_token_id=2)

    def forward(self, ids, **kwargs):
        for block in self.blocks:
            block(ids)
        return SimpleNamespace(logits=self.logits.reshape(1, 1, -1).expand(1, ids.shape[1], -1))

    def generate(self, ids, **kwargs):
        self.generate_calls += 1
        return ids


def _write(tmp_path, rows):
    path = tmp_path / "mirror.jsonl"
    path.write_text("\n".join(rows) + "\n")
    return path


def test_mirror_answers_round_trip_and_keep_order(tmp_path):
    path = _write(tmp_path, [
        '{"id": "a", "frame": "oracle", "answer": "Sure here is how"}',
        '{"id": "b", "frame": "seed-x", "answer": "hello world"}',
    ])
    assert read_mirror_answers(path) == [
        ("a", "oracle", "Sure here is how"), ("b", "seed-x", "hello world")]


@pytest.mark.parametrize("line, message", [
    ('{"id": "a", "frame": "none", "answer": "x"}', "other than 'none'"),
    ('{"id": "a", "frame": "", "answer": "x"}', "other than 'none'"),
    ('{"id": "a", "frame": "oracle", "answer": "  "}', "nonempty text"),
    ('{"frame": "oracle", "answer": "x"}', "id must be"),
    ('not json', "is not JSON"),
])
def test_mirror_answers_refuse_malformed_rows(tmp_path, line, message):
    with pytest.raises(ValueError, match=message):
        read_mirror_answers(_write(tmp_path, [line]))


def test_mirror_frame_renders_exactly_the_plain_prompt():
    # raw path: the template is {prompt} with no seed, so the text is unchanged
    assert render_frame(None, mirror_frame("oracle"), "hello world", "", raw=True) == \
        ("hello world", "raw")
    # a tokenizer with no chat template takes the same fallback the plain path
    # takes, so the mirror record and its plain sibling are the same prompt text
    assert render_frame(None, mirror_frame("oracle"), "hello world", "") == ("hello world", "raw")


def test_capture_forces_the_mirror_answer_without_generating():
    model = _ToyModel()
    row = {"id": "a", "kind": "refusal", "prompt": "hello world"}
    captured, reason = capture_one(
        model, _Tokenizer(), row, list(model.blocks), 4, locus="answer",
        max_tokens=8, ctx_size=64, system="", raw=True, device="cpu",
        settle=False, answer_text="Sure here is how", frame=mirror_frame("oracle"))
    assert reason is None
    meta, residual = captured
    assert meta["frame"] == "oracle"
    assert meta["answer_source"] == "banked"
    assert model.generate_calls == 0
    assert tuple(residual.shape) == (2, 4)
