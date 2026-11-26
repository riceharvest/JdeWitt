"""Unit-level checks for the static axis audit's math (numpy only, no torch).

The audit's defensible numbers are all matrix algebra on weight bytes, so the
only thing that can silently go wrong is the algebra itself: a mis-signed
aggregation, a BF16 decode off by an exponent, or a rank-1 share computed from
the wrong trace. These tests pin those three.
"""
import importlib.util
import json
import struct

import numpy as np
import pytest

from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'axis_audit.py'
_spec = importlib.util.spec_from_file_location('axis_audit', SCRIPT)
axis_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(axis_audit)


def _bf16_bits(values):
    u = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return (rounded & 0xFFFF).astype('<u2')


def _write_safetensors(path, tensors):
    """tensors: name -> (dtype, shape, raw bytes)."""
    header, offset = {}, 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {'dtype': dtype, 'shape': list(shape),
                        'data_offsets': [offset, offset + len(raw)]}
        offset += len(raw)
    blob = json.dumps(header).encode()
    blob += b' ' * (-len(blob) % 8)
    path.write_bytes(struct.pack('<Q', len(blob)) + blob + b''.join(t[2] for t in tensors.values()))


def test_rank1_delta_recovers_vector_and_share():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(64)
    r /= np.linalg.norm(r)
    v = rng.standard_normal(200)
    dW = 2.5 * np.outer(r, v)
    ax = axis_audit.top_axis(dW.astype(np.float32))
    assert ax['share1'] == pytest.approx(1.0, abs=1e-9)
    assert abs(float(np.dot(ax['u'], r))) == pytest.approx(1.0, abs=1e-6)
    assert ax['s1'] == pytest.approx(2.5 * np.linalg.norm(v), rel=1e-6)
    assert ax['cumulative_share']['k2'] == pytest.approx(1.0, abs=1e-9)


def test_two_orthogonal_components_split_energy():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    # dW = [[3, 0], [0, 4]]: singular values 4 and 3, so the energy share is 16/25.
    dW = 3.0 * np.outer(a, np.array([1.0, 0.0])) + 4.0 * np.outer(b, np.array([0.0, 1.0]))
    ax = axis_audit.top_axis(dW.astype(np.float32))
    assert ax['s1'] == pytest.approx(4.0, rel=1e-6)
    assert ax['share1'] == pytest.approx(16.0 / 25.0, rel=1e-6)
    assert ax['spectral_gap_s2_over_s1'] == pytest.approx(3.0 / 4.0, rel=1e-6)


def test_column_axis_is_the_rank1_vector():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(32)
    r /= np.linalg.norm(r)
    dW = np.outer(r, rng.standard_normal(50)) + 5.0 * np.outer(r, rng.standard_normal(50) * 1e-3)
    u, norm, j = axis_audit.column_axis(dW.astype(np.float32))
    assert abs(float(np.dot(u, r))) == pytest.approx(1.0, abs=1e-4)


def test_combine_sign_aligns_before_summing():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    out = axis_audit.combine([a, b], [1.0, 1.0])
    assert abs(float(out[0])) == pytest.approx(1.0, abs=1e-6)


def test_bf16_reader_decodes_round_to_nearest(tmp_path):
    values = np.array([1.0, -2.5, 0.1, 1024.0], dtype=np.float32)
    bits = _bf16_bits(values)
    raw = bits.tobytes()
    path = tmp_path / 'x.safetensors'
    _write_safetensors(path, {'w': ('BF16', values.shape, raw),
                              'f': ('F32', (2,), np.array([1.5, -0.25], np.float32).tobytes())})
    st = axis_audit.SafeTensors(path)
    got = st.get('w')
    expected = (bits.astype(np.uint32) << 16).view(np.float32)
    assert np.array_equal(got, expected)
    assert got[0] == 1.0 and got[1] == -2.5
    assert np.array_equal(st.get('f'), np.array([[1.5, -0.25]], np.float32).reshape(2))
    assert st.has('w') and not st.has('missing')


def test_cos_is_sign_sensitive_but_report_uses_abs():
    a = np.array([1.0, 0.0, 0.0])
    assert axis_audit.cos(a, -a) == pytest.approx(-1.0)
    assert abs(axis_audit.cos(a, -a)) == pytest.approx(1.0)
