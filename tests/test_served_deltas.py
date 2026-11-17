import numpy as np
import pytest
import torch
from pathlib import Path

from b70_ablator.gguf_adapters import served_deltas
from b70_ablator.ggufio import edit_tensors, dequantize_tensor, write_gguf


def _f32_gguf(tmp_path, dims, W, name='t.weight'):
    p = tmp_path / 'm.gguf'
    write_gguf(p, {'general.architecture': 'test', 'general.name': 'pytest'},
               [(name, dims, 0, np.asarray(W, dtype=np.float32).reshape(-1).tobytes())])
    return p


def _orth(rows, cols, seed):
    q = np.linalg.qr(np.random.RandomState(seed).randn(cols, rows))[0]  # [cols, rows]
    return q.T.astype(np.float32)  # [rows, cols] orthonormal rows


def test_output_side_direct_layout(tmp_path):
    W = np.random.RandomState(0).randn(4, 3).astype(np.float32)
    d = _orth(2, 4, 1)
    p = _f32_gguf(tmp_path, (4, 3), W)
    deltas = served_deltas(p, d, ['t.weight'], strength=1.0)
    expected = -d.T @ (d @ W)
    got = served_deltas(p, d, ['t.weight'], strength=1.0)['t.weight']
    np.testing.assert_allclose(got, expected, atol=1e-5)
    rep = edit_tensors(p, tmp_path / 'out.gguf', deltas)
    assert rep['tensors_edited'] == 1
    merged = dequantize_tensor(tmp_path / 'out.gguf', 't.weight')
    np.testing.assert_allclose(merged, W + expected, atol=1e-5)


def test_width_matches_second_dim_transposes(tmp_path):
    W = np.random.RandomState(2).randn(3, 4).astype(np.float32)  # stored [in, out]
    d = _orth(2, 4, 3)
    p = _f32_gguf(tmp_path, (3, 4), W)
    delta = served_deltas(p, d, ['t.weight'], strength=1.0)['t.weight']
    expected = (-d.T @ (d @ W.T)).T
    np.testing.assert_allclose(delta, expected, atol=1e-5)


def test_width_mismatch_raises(tmp_path):
    W = np.random.RandomState(4).randn(4, 3).astype(np.float32)
    d = _orth(2, 5, 5)
    p = _f32_gguf(tmp_path, (4, 3), W)
    with pytest.raises(ValueError, match='matches neither dim'):
        served_deltas(p, d, ['t.weight'])


def test_bf16_roundtrip(tmp_path):
    W = np.random.RandomState(6).randn(4, 3).astype(np.float32)
    d = _orth(2, 4, 7)
    u16 = torch.from_numpy(W).to(torch.bfloat16).view(torch.uint16).numpy().reshape(-1)
    p = tmp_path / 'm.gguf'
    write_gguf(p, {'general.architecture': 'test'}, [('t.weight', (4, 3), 30, u16.tobytes())])
    deltas = served_deltas(p, d, ['t.weight'], strength=1.0)
    edit_tensors(p, tmp_path / 'out.gguf', deltas)
    merged = dequantize_tensor(tmp_path / 'out.gguf', 't.weight')
    base = dequantize_tensor(p, 't.weight')
    expected = -d.T @ (d @ base)
    np.testing.assert_allclose(merged, base + expected, atol=5e-2)
