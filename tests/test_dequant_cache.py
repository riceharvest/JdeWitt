"""DequantBases: content-addressed bases, one pass per arm, identical bytes.

The cache is keyed on a tensor's RAW block bytes + name + quant type, never on
the file digest: tune merges and reverts an artifact in place, so the file sha
changes exactly when a cached base is still the right base.
"""
import numpy as np
import torch
from gguf import GGUFReader, GGUFWriter

from jdewitt.gguf_adapters import served_deltas
from jdewitt.ggufio import (DequantBases, _tensor_info, dequantize_tensor, edit_tensors,
                            project_tensors_inplace, revert_inplace, write_gguf)


def _gguf(path, tensors):
    """tensors: [(name, dims, dtype_id, flat_bytes)]"""
    write_gguf(path, {'general.architecture': 'test', 'general.name': 'pytest'}, tensors)
    return path


def _f32(dims, seed):
    arr = np.random.RandomState(seed).randn(*dims).astype(np.float32)
    return arr, np.ascontiguousarray(arr).reshape(-1).tobytes()


def _writer(path, tensors, arch='ggmf'):
    writer = GGUFWriter(str(path), arch)
    for name, arr in tensors.items():
        writer.add_tensor(name, arr)
    writer.open_output_file()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_cache_miss_then_hit_and_bytes_match_no_cache(tmp_path):
    arr, raw = _f32((4, 3), 1)  # logical [out, in]; gguf dims are (in, out)
    src = _gguf(tmp_path / 'm.gguf', [('t.weight', (3, 4), 0, raw)])
    delta = np.full((4, 3), 0.01, dtype=np.float32)

    plain = tmp_path / 'plain.gguf'
    edit_tensors(src, plain, {'t.weight': delta})

    cache = DequantBases(tmp_path / 'cache')
    first = edit_tensors(src, tmp_path / 'a.gguf', {'t.weight': delta}, dequant_cache=cache)
    second = edit_tensors(src, tmp_path / 'b.gguf', {'t.weight': delta}, dequant_cache=cache)

    assert (first['cache_misses'], first['cache_hits']) == (1, 0)
    assert (second['cache_misses'], second['cache_hits']) == (0, 1)
    assert second['dequant_seconds'] == 0.0
    assert plain.read_bytes() == (tmp_path / 'a.gguf').read_bytes() == (tmp_path / 'b.gguf').read_bytes()
    # one memmap .npy per tensor, and a hit is served from it
    assert len(list((tmp_path / 'cache').glob('*.npy'))) == 1


def test_cache_key_follows_raw_bytes_not_file_identity(tmp_path):
    arr, raw = _f32((4, 3), 2)
    src = _gguf(tmp_path / 'm.gguf', [('t.weight', (3, 4), 0, raw)])
    delta = np.zeros((4, 3), dtype=np.float32)
    delta[0, 0] = 0.5
    cache = DequantBases(tmp_path / 'cache')
    edit_tensors(src, tmp_path / 'a.gguf', {'t.weight': delta}, dequant_cache=cache)

    # Rewrite the tensor's bytes (a Q4_K merge does exactly this, then a revert
    # restores them): the key must change, so this is a miss even though the
    # file is at the same path.
    _tensor, _dtype, _bs, _els, _n, _q, _dq, offset, n_bytes, _shape = _tensor_info(
        GGUFReader(str(src)), 't.weight')
    with open(src, 'r+b') as handle:
        handle.seek(offset)
        mutated = bytearray(handle.read(n_bytes))
        mutated[0] ^= 0xFF
        handle.seek(offset)
        handle.write(mutated)
    report = edit_tensors(src, tmp_path / 'b.gguf', {'t.weight': delta}, dequant_cache=cache)
    assert (report['cache_misses'], report['cache_hits']) == (1, 0)
    assert len(list((tmp_path / 'cache').glob('*.npy'))) == 2


def test_inplace_patch_reads_cached_base_and_reverts(tmp_path):
    arr, raw = _f32((4, 3), 7)
    src = _gguf(tmp_path / 'm.gguf', [('t.weight', (3, 4), 0, raw)])
    before = src.read_bytes()
    axis = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)
    cache = DequantBases(tmp_path / 'cache')
    patch = project_tensors_inplace(str(src), {'t.weight': (axis, 'output')},
                                    revert_path=str(tmp_path / 'r.npz'), dequant_cache=cache)
    assert (patch['cache_misses'], patch['cache_hits']) == (1, 0)
    assert patch['changed_blocks_total'] > 0
    revert_inplace(str(src), str(tmp_path / 'r.npz'))
    assert src.read_bytes() == before  # revert restores the bytes -> the key hits again
    warm = project_tensors_inplace(str(src), {'t.weight': (axis, 'output')}, dequant_cache=cache)
    assert (warm['cache_misses'], warm['cache_hits']) == (0, 1)
    assert warm['dequant_seconds'] == 0.0


def test_projection_writer_matches_delta_writer(tmp_path):
    arr, raw = _f32((4, 3), 3)
    src = _gguf(tmp_path / 'm.gguf', [('t.weight', (3, 4), 0, raw)])
    axis = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)  # output width 4
    delta = -(np.outer(axis, axis @ arr)).astype(np.float32)

    delta_out = tmp_path / 'delta.gguf'
    proj_out = tmp_path / 'proj.gguf'
    edit_tensors(src, delta_out, {'t.weight': delta})
    edit_tensors(src, proj_out, projections={'t.weight': (axis, 'output')})
    assert delta_out.read_bytes() == proj_out.read_bytes()


def test_renormalize_projection_matches_precomputed_delta(tmp_path):
    arr, raw = _f32((4, 3), 4)
    src = _gguf(tmp_path / 'm.gguf', [('t.weight', (3, 4), 0, raw)])
    axis = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)
    edited = arr - np.outer(axis, axis @ arr)
    rows = np.linalg.norm(arr, axis=1, keepdims=True)
    edited = edited * (rows / np.maximum(np.linalg.norm(edited, axis=1, keepdims=True), 1e-12))
    delta = (edited - arr).astype(np.float32)

    a = tmp_path / 'delta.gguf'
    b = tmp_path / 'proj.gguf'
    edit_tensors(src, a, {'t.weight': delta})
    edit_tensors(src, b, projections={'t.weight': (axis, 'output')}, renormalize=True)
    assert a.read_bytes() == b.read_bytes()


def test_served_deltas_as_projections_matches_delta_path(tmp_path):
    W = np.random.RandomState(5).randn(4, 3).astype(np.float32)  # logical [out, in]
    src = _writer(tmp_path / 'real.gguf', {'t.weight': W}, arch='pytestarch')
    direction = torch.tensor([[0.6, 0.8, 0.0, 0.0]])
    projections = served_deltas(src, direction, ['t.weight'], as_projections=True)
    assert projections['t.weight'][1] == 'output'
    deltas = served_deltas(src, direction, ['t.weight'], strength=1.0)
    a = tmp_path / 'a.gguf'
    b = tmp_path / 'b.gguf'
    edit_tensors(src, a, deltas)
    edit_tensors(src, b, projections=projections)
    assert a.read_bytes() == b.read_bytes()


def test_out_of_place_merge_dequantizes_each_tensor_once(tmp_path, monkeypatch):
    """The old path dequantized once for the delta and once in the writer."""
    from jdewitt import quants, spanfit
    raw = _f32((8, 8), 8)[1]
    src = _gguf(tmp_path / 'in.gguf',
                [('blk.0.attn_output.weight', (8, 8), 0, raw),
                 ('token_embd.weight', (4, 8), 0, _f32((4, 8), 9)[1])])
    axes = torch.nn.functional.normalize(torch.randn(1, 8), dim=1)
    calls = []
    block_bytes, block_els, quantize, dequantize = quants.CODECS['F32']

    def counted(values, n, _fn=dequantize):
        calls.append(n)
        return _fn(values, n)

    monkeypatch.setitem(quants.CODECS, 'F32', (block_bytes, block_els, quantize, counted))
    report = spanfit.build_merged(
        str(src), str(tmp_path / 'out.gguf'), axes,
        ['model.layers.0.self_attn.o_proj.weight'],
        name_map={'model.layers.0.self_attn.o_proj.weight': 'blk.0.attn_output.weight'},
        axis_space='output', dequant_cache=False)
    assert report['tensors_edited'] == 1
    assert len(calls) == 1, 'one dequant pass per edited tensor'


def test_build_merged_warm_arm_reports_zero_dequant(tmp_path):
    from jdewitt import spanfit
    g = np.random.RandomState(6)
    tensors = {'blk.0.attn_output.weight': g.randn(8, 8).astype(np.float32),
               'token_embd.weight': g.randn(4, 8).astype(np.float32)}
    src = _writer(tmp_path / 'in.gguf', tensors)
    axes = torch.nn.functional.normalize(torch.randn(1, 8), dim=1)
    cache = DequantBases(tmp_path / 'cache')
    keys = ['model.layers.0.self_attn.o_proj.weight']
    name_map = {keys[0]: 'blk.0.attn_output.weight'}
    cold = spanfit.build_merged(str(src), str(tmp_path / 'cold.gguf'), axes, keys,
                                name_map=name_map, axis_space='output', dequant_cache=cache)
    warm = spanfit.build_merged(str(src), str(tmp_path / 'warm.gguf'), axes, keys,
                                name_map=name_map, axis_space='output', dequant_cache=cache)
    assert (cold['cache_misses'], cold['cache_hits']) == (1, 0)
    assert (warm['cache_misses'], warm['cache_hits']) == (0, 1)
    assert warm['dequant_seconds'] == 0.0
    assert (tmp_path / 'cold.gguf').read_bytes() == (tmp_path / 'warm.gguf').read_bytes()
    assert dequantize_tensor(str(tmp_path / 'warm.gguf'), 'blk.0.attn_output.weight').shape == (8, 8)
