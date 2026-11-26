"""The twin-transplant path: delta edits, rank-k reduction, and a revert that grows.

What is checked here is weight algebra and bytes only, the same contract the script
carries: an explicit float32 delta is what ``rewrite_with_edits`` stores, the
fingerprint tells two different deltas apart, a rank-k reduction is a real matrix of
rank k, and an in-place edit that CHANGED the file's layout still reverts byte-exactly.
"""
import json
import struct

import numpy as np
import pytest

from jdewitt.ggufio import compare_edit, revert_inplace, save_whole_file_revert, write_gguf
from jdewitt.rewrite import axis_fingerprint, rewrite_with_edits
from scripts.twin_transplant import (cos64, hf_key, rank_k, read_tensor,
                                     safetensors_index)


def _f32_gguf(path, shapes):
    """A tiny f32 artifact: [(name, ggml dims)] -> the logical (out, in) is reversed."""
    tensors = []
    rng = np.random.default_rng(7)
    for name, dims in shapes:
        logical = tuple(reversed(dims))
        array = rng.standard_normal(logical).astype(np.float32) * 0.05
        tensors.append((name, dims, 0, np.ascontiguousarray(array).reshape(-1).tobytes()))
    write_gguf(path, {'general.architecture': 'test', 'general.name': 'pytest'}, tensors)
    return path


def _delta(path, shapes, seed=3):
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(tuple(reversed(dims))).astype(np.float32) * 0.01
            for name, dims in shapes}


def test_delta_edit_reencodes_only_named_tensors_and_delivers_the_delta(tmp_path):
    shapes = [('blk.0.attn_output.weight', (64, 96)),
              ('blk.0.ffn_down.weight', (256, 96)),
              ('blk.0.attn_q.weight', (96, 96))]
    src = _f32_gguf(tmp_path / 'src.gguf', shapes)
    edits = _delta(src, shapes[:2])

    out = tmp_path / 'out.gguf'
    report = rewrite_with_edits(src, out, edits, edit_type='q8_0')
    assert report['tensors_edited'] == 2
    assert report['untouched']['verified'] is True
    assert report['untouched']['checked'] == 1
    # the fixture is f32 (4 B/weight) and Q8_0 is ~1.06 B/weight: only the two
    # edited tensors shrink, the untouched one keeps its bytes and its type
    assert report['bytes_out'] < report['bytes_in']
    delivered = compare_edit(src, out, edits, names=sorted(edits))
    assert delivered['min_cos'] > 0.99
    # the third tensor is untouched byte for byte, its type included
    assert report['edited'] and all(row['to_dtype'] == 'Q8_0' for row in report['edited'])


def test_fingerprint_separates_the_full_delta_from_its_rank_k_reduction(tmp_path):
    shapes = [('blk.1.ffn_down.weight', (64, 96))]
    edits = {'blk.1.ffn_down.weight': np.random.default_rng(1).standard_normal((96, 64))
             .astype(np.float32)}
    reduced = {'blk.1.ffn_down.weight': rank_k(edits['blk.1.ffn_down.weight'], 4)[0]}
    full_digest = axis_fingerprint(edits)['edit_sha256']
    reduced_digest = axis_fingerprint(reduced)['edit_sha256']
    assert full_digest != reduced_digest
    # an explicit delta cannot collide with an axis of the same numbers
    axis_digest = axis_fingerprint(
        {'blk.1.ffn_down.weight': (edits['blk.1.ffn_down.weight'].reshape(-1), 'output')}
    )['edit_sha256']
    assert axis_digest != full_digest


def test_rank_k_is_a_rank_k_matrix_and_cos_rises_with_k():
    delta = np.random.default_rng(5).standard_normal((96, 256)).astype(np.float32)
    cosines = []
    for k in (1, 4, 8, 16):
        reduced, cosine, energy = rank_k(delta, k)
        cosines.append(cosine)
        assert 0.0 <= energy <= 1.0
        assert reduced.shape == delta.shape
        assert abs(cosine - cos64(delta, reduced)) < 1e-6
        # a rank-k product is rank k: every singular value past k is float32 noise
        singular = np.linalg.svd(reduced.astype(np.float64), compute_uv=False)
        assert singular[k:].max() < singular[0] * 1e-5
    assert cosines == sorted(cosines)


def test_revert_restores_a_grown_artifact_byte_exactly(tmp_path):
    """An in-place re-encode moves every offset; the sidecar must cut the file back.

    The source is Q4_0 (0.5 B/weight), so re-encoding an edited tensor at Q8_0
    makes the artifact LONGER — the case a tile sidecar cannot express.
    """
    shapes = [('blk.0.attn_output.weight', (64, 96))]
    f32 = _f32_gguf(tmp_path / 'src-f32.gguf', shapes)
    pristine = tmp_path / 'inplace.gguf'
    rewrite_with_edits(f32, pristine, _delta(f32, shapes, seed=11), edit_type='q4_0')
    original = pristine.read_bytes()
    edits = _delta(pristine, shapes)
    edited = tmp_path / 'edited.gguf'
    rewrite_with_edits(pristine, edited, edits, edit_type='q8_0')
    assert edited.stat().st_size > len(original)

    sidecar = tmp_path / 'inplace.gguf.revert.npz'
    save_whole_file_revert(pristine, edited, sidecar)
    edited.replace(pristine)
    assert pristine.read_bytes() != original
    assert pristine.stat().st_size > len(original)

    result = revert_inplace(pristine, sidecar)
    assert result['byte_exact'] is True
    assert result['truncated_to'] == len(original)
    assert pristine.read_bytes() == original
    assert not sidecar.exists()


def test_revert_refuses_an_artifact_that_changed_since_the_patch(tmp_path):
    shapes = [('blk.0.attn_output.weight', (64, 96))]
    pristine = _f32_gguf(tmp_path / 'inplace.gguf', shapes)
    edits = _delta(pristine, shapes)
    edited = tmp_path / 'edited.gguf'
    rewrite_with_edits(pristine, edited, edits, edit_type='q8_0')
    sidecar = tmp_path / 'inplace.gguf.revert.npz'
    save_whole_file_revert(pristine, edited, sidecar)
    edited.replace(pristine)
    with open(pristine, 'r+b') as handle:  # someone touched it afterwards
        handle.seek(0)
        handle.write(b'XXXX')
    with pytest.raises(ValueError, match='changed since the patch'):
        revert_inplace(pristine, sidecar)


def _bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32) >> 16
    return bits.astype('<u2').tobytes()


def _safetensors(path, tensors):
    """Write a minimal safetensors file: {name: (dtype, shape, raw bytes)}."""
    header, offset = {}, 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {'dtype': dtype, 'shape': list(shape),
                        'data_offsets': [offset, offset + len(raw)]}
        offset += len(raw)
    blob = json.dumps(header).encode()
    with open(path, 'wb') as handle:
        handle.write(struct.pack('<Q', len(blob)))
        handle.write(blob)
        for _dtype, _shape, raw in tensors.values():
            handle.write(raw)
    return path


def test_safetensors_reader_returns_bf16_values_in_torch_layout(tmp_path):
    values = np.array([[1.0, 2.0, -0.5], [0.25, 8.0, -4.0]], dtype=np.float32)
    path = _safetensors(tmp_path / 'model.safetensors',
                        {'model.layers.0.mlp.down_proj.weight': ('BF16', values.shape,
                                                                 _bf16(values))})
    index, start = safetensors_index(path)
    read = read_tensor(path, index, start, 'model.layers.0.mlp.down_proj.weight')
    assert read.shape == values.shape
    assert np.allclose(read, values)


def test_hf_key_maps_only_the_writeback_projections():
    assert hf_key('blk.3.ffn_down.weight') == 'model.layers.3.mlp.down_proj.weight'
    assert hf_key('blk.11.attn_output.weight') == 'model.layers.11.self_attn.o_proj.weight'
    assert hf_key('blk.3.attn_q.weight') is None
    assert hf_key('token_embd.weight') is None
