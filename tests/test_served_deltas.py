import numpy as np
import pytest
import torch
from pathlib import Path

from b70_ablator.gguf_adapters import served_deltas
from b70_ablator.ggufio import edit_tensors, dequantize_tensor, write_gguf


def _gguf(tmp_path, dims, flat_bytes, dtype_id=0, name='t.weight'):
    p = tmp_path / 'm.gguf'
    write_gguf(p, {'general.architecture': 'test', 'general.name': 'pytest'},
               [(name, dims, dtype_id, flat_bytes)])
    return p


def _orth(rows, cols, seed):
    q = np.linalg.qr(np.random.RandomState(seed).randn(cols, rows))[0]  # [cols, rows]
    return q.T.astype(np.float32)  # [rows, cols] orthonormal rows


def test_layout_roundtrip_matches_gguf_writer(tmp_path):
    # Regression for the 2026-11-17 convention bug: tensor bytes must reshape to
    # the LOGICAL (reversed-dims) order the GGML loader and gguf-py agree on,
    # never the raw ne order. Written with the real GGUFWriter.
    from gguf import GGUFWriter
    W = np.random.RandomState(9).randn(6, 5).astype(np.float32)  # logical [out,in]
    p = tmp_path / 'real.gguf'
    wr = GGUFWriter(str(p), 'pytestarch')
    wr.add_tensor('t.weight', W)
    wr.open_output_file(); wr.write_header_to_file()
    wr.write_kv_data_to_file(); wr.write_tensors_to_file(); wr.close()
    got = dequantize_tensor(p, 't.weight')
    assert got.shape == W.shape
    np.testing.assert_allclose(got, W, atol=1e-5)


def test_output_side_direct_layout(tmp_path):
    W = np.random.RandomState(0).randn(4, 3).astype(np.float32)  # logical [out,in]
    d = _orth(2, 4, 1)
    p = _gguf(tmp_path, (3, 4), np.ascontiguousarray(W).reshape(-1).tobytes())
    deltas = served_deltas(p, d, ['t.weight'], strength=1.0)
    expected = -d.T @ (d @ W)
    np.testing.assert_allclose(deltas['t.weight'], expected, atol=1e-5)
    rep = edit_tensors(p, tmp_path / 'out.gguf', deltas)
    assert rep['tensors_edited'] == 1
    merged = dequantize_tensor(tmp_path / 'out.gguf', 't.weight')
    np.testing.assert_allclose(merged, W + expected, atol=1e-5)


def test_transposed_file_uses_transpose_path(tmp_path):
    # A hypothetical file stored with dims [out, in] (ne0=out): bytes C(W.T)
    # reshape to logical (in, out); the width rule must transpose around it.
    W = np.random.RandomState(2).randn(3, 4).astype(np.float32)
    d = _orth(3, 4, 3)
    p = _gguf(tmp_path, (3, 4), np.ascontiguousarray(W.T).reshape(-1).tobytes())
    logical = np.ascontiguousarray(W.T).reshape(4, 3)  # what served_deltas sees
    delta = served_deltas(p, d, ['t.weight'], strength=1.0)['t.weight']
    assert delta.shape == logical.shape
    expected = -d.T @ (d @ logical)  # width 3 == logical dim0; no transpose needed here
    np.testing.assert_allclose(delta, expected, atol=1e-5)


def test_width_mismatch_raises(tmp_path):
    W = np.random.RandomState(4).randn(4, 3).astype(np.float32)
    d = _orth(2, 5, 5)
    p = _gguf(tmp_path, (3, 4), np.ascontiguousarray(W).reshape(-1).tobytes())
    with pytest.raises(ValueError, match='matches neither dim'):
        served_deltas(p, d, ['t.weight'])


def test_bf16_roundtrip(tmp_path):
    W = np.random.RandomState(6).randn(4, 3).astype(np.float32)
    d = _orth(2, 4, 7)
    u16 = torch.from_numpy(W).to(torch.bfloat16).view(torch.uint16).numpy().reshape(-1)
    p = _gguf(tmp_path, (3, 4), u16.tobytes(), dtype_id=30)
    deltas = served_deltas(p, d, ['t.weight'], strength=1.0)
    edit_tensors(p, tmp_path / 'out.gguf', deltas)
    merged = dequantize_tensor(tmp_path / 'out.gguf', 't.weight')
    base = dequantize_tensor(p, 't.weight')
    expected = -d.T @ (d @ base)
    np.testing.assert_allclose(merged, base + expected, atol=5e-2)
