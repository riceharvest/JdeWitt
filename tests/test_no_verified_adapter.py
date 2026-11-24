"""P8: a layout with no verifiable adapter refuses with its own typed code.

The four sites that used to raise a bare ValueError (which the CLI maps to
INPUT_ERROR, the same code as a typo'd flag value) now raise NoVerifiedAdapter
and carry what discovery saw: layer ids with no target, the linear module names
in the first such block, and the width-matching tensor names that would not be
edited. The honest boundary stays in the class docstring: discovery proves a
tensor NAME, not that the architecture serves the edit (that is the export
smoke's job).
"""
import json
import sys

import numpy as np
import pytest
import torch
from torch import nn

from jdewitt import gguf_adapters, spanfit
from jdewitt.errors import NoVerifiedAdapter, error_code, explain_error

WIDTH = 8


def _write_fixture(path):
    """A GGUF whose hidden-wide 2D weights are read-side names, plus one real target.

    blk.0.attn_q.weight is hidden-wide but not a residual-write name, so layer 0
    has no adapter this tool can name; blk.1.attn_output.weight is one.
    """
    from gguf import GGUFWriter
    g = np.random.default_rng(11)
    tensors = {'blk.0.attn_q.weight': g.standard_normal((WIDTH, WIDTH)).astype(np.float32),
               'blk.1.attn_output.weight': g.standard_normal((WIDTH, WIDTH)).astype(np.float32)}
    w = GGUFWriter(str(path), 'fixturearch')
    for name, arr in tensors.items():
        w.add_tensor(name, arr)
    w.open_output_file()
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    axes = torch.nn.functional.normalize(
        torch.from_numpy(g.standard_normal((2, WIDTH)).astype(np.float32)), dim=1)
    axes_path = path.with_suffix('.axes.pt')
    torch.save({'axes': axes}, axes_path)
    return axes_path


def test_error_code_is_its_own_code_not_input_error():
    error = NoVerifiedAdapter('unnameable layout', layer=3)
    assert error_code(error) == 'NO_VERIFIED_ADAPTER'
    assert error.facts == {'layer': 3}
    # A plain ValueError is still INPUT_ERROR: the new class is the exception.
    assert error_code(ValueError('bad flag value')) == 'INPUT_ERROR'
    # The subclass relationship is deliberate: existing except-ValueError callers
    # keep working while the CLI can tell the two apart.
    assert isinstance(error, ValueError)


def test_explainer_row_names_the_next_move():
    row = explain_error('NO_VERIFIED_ADAPTER')
    assert row['code'] == 'NO_VERIFIED_ADAPTER'
    assert 'adapter' in row['meaning']
    assert 'spanfit targets' in row['agent_action']
    assert 'envelope' in row


def test_resolve_gguf_names_refuses_a_key_it_cannot_map():
    shapes = {'blk.0.attn_q.weight': (WIDTH, WIDTH),
              'blk.1.attn_output.weight': (WIDTH, WIDTH)}
    with pytest.raises(NoVerifiedAdapter) as caught:
        gguf_adapters.resolve_gguf_names(['model.layers.0.self_attn.q_proj.weight'], shapes)
    message = str(caught.value)
    assert 'no llama.cpp LoRA name mapping' in message
    assert 'blk.0.attn_q.weight' in message            # the unmapped width-matching tensor
    assert '[1]' in message                            # the layer that does have a target
    assert caught.value.facts['source_key'] == 'model.layers.0.self_attn.q_proj.weight'


def test_resolve_gguf_names_refuses_a_missing_tensor():
    shapes = {'blk.0.attn_q.weight': (WIDTH, WIDTH)}
    with pytest.raises(NoVerifiedAdapter) as caught:
        gguf_adapters.resolve_gguf_names(['model.layers.0.self_attn.o_proj.weight'], shapes)
    assert 'artifact has no tensor' in str(caught.value)
    assert caught.value.facts['looked_for'] == 'blk.0.attn_output.weight'


def test_spanfit_auto_refuses_and_names_what_it_saw(tmp_path, capsys):
    src = tmp_path / 'in.gguf'
    axes_path = _write_fixture(src)
    with pytest.raises(SystemExit) as exit_info:
        spanfit.main(['build', '--gguf-in', str(src), '--axes', str(axes_path),
                      '--out', str(tmp_path / 'out.gguf'), '--gguf-targets', 'auto'])
    assert exit_info.value.code == 1
    envelope = json.loads(capsys.readouterr().err)
    assert envelope['error']['code'] == 'NO_VERIFIED_ADAPTER'
    assert envelope['error']['type'] == 'NoVerifiedAdapter'
    assert envelope['error']['command'] == 'build'
    message = envelope['error']['message']
    assert 'layers [0]' in message          # discovered layer ids with no target
    assert 'blk.0.attn_q.weight' in message  # width-matching name it would not edit
    assert 'found residual-write layers [1]' in message
    assert not (tmp_path / 'out.gguf').exists()


def test_spanfit_auto_still_merges_when_targets_are_found(tmp_path, capsys):
    """Positive control: the refusal must not swallow a layout it can name."""
    src = tmp_path / 'in.gguf'
    axes_path = _write_fixture(src)
    out = tmp_path / 'out.gguf'
    spanfit.main(['build', '--gguf-in', str(src), '--axes', str(axes_path),
                  '--out', str(out), '--gguf-targets', 'auto', '--only-layers', '1'])
    report = json.loads(capsys.readouterr().out)
    assert report['tensors_changed'] == 1
    assert report['tensors'][0]['tensor'] == 'blk.1.attn_output.weight'
    assert out.is_file()


def test_capture_refuses_blocks_without_writeback_names(tmp_path, monkeypatch):
    """A block whose only hidden-wide linear is q_proj has no nameable adapter."""
    import transformers

    class ToyConfig:
        hidden_size = WIDTH
        num_hidden_layers = 2

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(WIDTH, WIDTH, bias=False)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(2)])

        def eval(self):
            return self

        def generate(self, *args, **kwargs):
            raise AssertionError('the adapter check runs before any decoding')

    class Factory:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return Toy()

    monkeypatch.setattr(transformers, 'AutoConfig',
                        type('C', (), {'from_pretrained': staticmethod(lambda *a, **k: ToyConfig())}))
    monkeypatch.setattr(transformers, 'AutoTokenizer',
                        type('T', (), {'from_pretrained': staticmethod(lambda *a, **k: object())}))
    monkeypatch.setattr(transformers, 'AutoModelForCausalLM', Factory)
    monkeypatch.setattr(transformers, 'AutoModel', Factory)

    suite = tmp_path / 'suite.jsonl'
    suite.write_text(json.dumps({'id': 'r1', 'kind': 'refusal', 'prompt': 'x'}) + '\n')

    from types import SimpleNamespace
    from jdewitt.capture import capture_command
    args = SimpleNamespace(suite=suite, kinds='refusal', limit=None, model='toy-model',
                           revision=None, trust_remote_code=False, hidden=None,
                           dtype='float16', allow_cuda=False, out=tmp_path / 'records.pt',
                           gguf=None, answers=None, locus='answer', max_tokens=8,
                           ctx_size=128, system=None, raw_prompt=False, settle=None,
                           settle_min_tokens=None)
    with pytest.raises(NoVerifiedAdapter) as caught:
        capture_command(args)
    message = str(caught.value)
    assert 'blocks [0, 1]' in message
    assert 'q_proj' in message             # the linear modules seen in that block
    assert caught.value.facts['layers_without_target'] == [0, 1]
    assert error_code(caught.value) == 'NO_VERIFIED_ADAPTER'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
