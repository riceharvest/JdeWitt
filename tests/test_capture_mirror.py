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


# --- model factory selection -------------------------------------------------

def _factory(name, model_class):
    class _F:
        calls = 0

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            cls.calls += 1
            return model_class()
    _F.__name__ = name
    return _F


class _Headless(torch.nn.Module):
    pass


class _Generating(torch.nn.Module):
    def generate(self):
        return None


def test_unmapped_vision_config_selects_image_text_factory_first():
    import transformers
    from jdewitt.capture import capture_factories
    # SimpleNamespace is not a mapped config class, like minicpmv4_7 for the causal-LM factory.
    vision = SimpleNamespace(vision_config=object(), architectures=["NoSuchClassForCapture"])
    order = capture_factories(vision)
    assert order[0] is transformers.AutoModelForImageTextToText
    assert order[-1] is transformers.AutoModel
    text = capture_factories(SimpleNamespace(architectures=[]))
    assert text[0] is transformers.AutoModelForCausalLM
    assert transformers.AutoModelForImageTextToText not in text


def test_real_minicpm_and_qwen35_configs_pick_their_factories():
    import transformers
    from transformers import CONFIG_MAPPING
    from jdewitt.capture import capture_factories
    mini = CONFIG_MAPPING["minicpmv4_7"]()
    assert capture_factories(mini)[0] is transformers.AutoModelForImageTextToText
    qwen = CONFIG_MAPPING["qwen3_5"]()
    assert capture_factories(qwen)[0] is transformers.AutoModelForCausalLM


def test_architecture_class_is_a_named_attempt():
    import transformers
    from jdewitt.capture import capture_factories
    cfg = SimpleNamespace(architectures=["Qwen3ForCausalLM"])
    assert transformers.Qwen3ForCausalLM in capture_factories(cfg)


def test_headless_class_is_never_chosen():
    from jdewitt.capture import load_generating_model
    headless, good = _factory("Headless", _Headless), _factory("Good", _Generating)
    model = load_generating_model([headless, good], "src", revision=None, dtype=None,
                                  device="cpu", trust_remote_code=False)
    assert isinstance(model, _Generating) and headless.calls == 1
    with pytest.raises(ValueError, match="has no generate"):
        load_generating_model([headless], "src", revision=None, dtype=None,
                              device="cpu", trust_remote_code=False)


def test_load_failure_falls_through_to_next_factory():
    from jdewitt.capture import load_generating_model

    class _Broken:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise KeyError("no mapping")
    model = load_generating_model([_Broken, _factory("Good", _Generating)], "src", revision=None,
                                  dtype=None, device="cpu", trust_remote_code=False)
    assert isinstance(model, _Generating)
