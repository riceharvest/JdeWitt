"""Tests for the precomputed-axis contract of `tune --axes` (P4e)."""
import argparse

import pytest
import torch

from jdewitt import tune

LAYERS, WIDTH = 4, 16


def _records(layers=LAYERS, width=WIDTH, rows=2):
    return [({'label': 'bad' if index % 2 else 'good'},
             torch.randn(layers, width)) for index in range(rows)]


def _args(**overrides):
    base = {'axes': None, 'axes_key': 'axes', 'records': 'records.pt', 'rank': 1}
    base.update(overrides)
    return argparse.Namespace(**base)


def test_load_axis_file_reads_dict_and_bare_tensor(tmp_path):
    tensor = torch.randn(LAYERS, WIDTH)
    wrapped = tmp_path / 'wrapped.pt'
    torch.save({'axes': tensor, 'meta': {'method': 'rdo'}}, wrapped)
    axes, meta, vector = tune.load_axis_file(wrapped)
    assert axes.shape == (LAYERS, WIDTH)
    assert torch.equal(axes, tensor.float())
    assert meta == {'method': 'rdo'} and vector is False

    bare = tmp_path / 'bare.pt'
    torch.save(tensor, bare)
    axes, meta, vector = tune.load_axis_file(bare)
    assert axes.shape == (LAYERS, WIDTH) and meta == {} and vector is False


def test_load_axis_file_accepts_a_vector_and_a_non_default_key(tmp_path):
    path = tmp_path / 'axes.pt'
    torch.save({'direction': torch.randn(WIDTH)}, path)
    axes, _meta, vector = tune.load_axis_file(path, key='direction')
    assert axes.shape == (1, WIDTH)
    assert vector is True


def test_load_axis_file_refuses_rank_k_and_missing_key(tmp_path):
    rank_k = tmp_path / 'rank-k.pt'
    torch.save({'axes': torch.randn(LAYERS, 3, WIDTH)}, rank_k)
    with pytest.raises(ValueError, match='different contract'):
        tune.load_axis_file(rank_k)

    wrong = tmp_path / 'wrong.pt'
    torch.save({'layers': torch.randn(LAYERS, WIDTH)}, wrong)
    with pytest.raises(ValueError, match='--axes-key'):
        tune.load_axis_file(wrong)


def test_resolve_axes_broadcasts_a_vector_over_the_capture(tmp_path):
    path = tmp_path / 'vector.pt'
    torch.save({'axes': torch.randn(WIDTH)}, path)
    axes, info = tune.resolve_axes(_args(axes=path), _records())
    assert axes.shape == (LAYERS, WIDTH)
    assert info['vector'] is True and info['layers'] == LAYERS


def test_resolve_axes_refuses_a_width_mismatch(tmp_path):
    path = tmp_path / 'wrong-width.pt'
    torch.save({'axes': torch.randn(LAYERS, WIDTH + 1)}, path)
    with pytest.raises(ValueError, match='width'):
        tune.resolve_axes(_args(axes=path), _records())


def test_resolve_axes_refuses_more_layer_rows_than_the_capture(tmp_path):
    path = tmp_path / 'too-tall.pt'
    torch.save({'axes': torch.randn(LAYERS + 1, WIDTH)}, path)
    with pytest.raises(ValueError, match='captured only'):
        tune.resolve_axes(_args(axes=path), _records())


def test_axes_digest_separates_sources_and_is_stable():
    torch.manual_seed(0)
    first = torch.randn(LAYERS, WIDTH)
    other = torch.randn(LAYERS, WIDTH)
    digest = tune.axes_digest({'perlayer': first})
    assert digest == tune.axes_digest({'perlayer': first.clone()})
    assert digest != tune.axes_digest({'perlayer': other})
    assert digest != tune.axes_digest({'perlayer': first, 'shared': first})


def test_axes_source_names_the_file_or_the_fit(tmp_path):
    assert tune.axes_source(_args()) == 'contrast'
    path = tmp_path / 'axes.pt'
    assert tune.axes_source(_args(axes=path)) == f'file:{path}'


def test_tune_refuses_axes_with_rank(tmp_path):
    from jdewitt.cli import build_parser

    args = build_parser().parse_args(['tune', '--gguf', str(tmp_path / 'model.gguf'),
                                      '--records', str(tmp_path / 'records.pt'),
                                      '--axes', str(tmp_path / 'axes.pt'), '--rank', '2',
                                      '--out', str(tmp_path / 'out')])
    with pytest.raises(ValueError, match='--rank'):
        tune.tune_command(args)


def test_cli_parses_the_axes_flags():
    from jdewitt.cli import build_parser

    args = build_parser().parse_args(['tune', '--gguf', 'x.gguf', '--records', 'r.pt',
                                      '--axes', 'axes.pt'])
    assert str(args.axes) == 'axes.pt'
    assert args.axes_key == 'axes'
