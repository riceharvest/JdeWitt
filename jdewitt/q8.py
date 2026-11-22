# SPDX-License-Identifier: AGPL-3.0-only
# Q8_1 / Q8_K block codecs ported from llama.cpp ggml-quants.c master
# (sha256 1665bffa9c0ef0e692f8799523a729879960a6aa07adaa32f9efb41f7076a3da) and
# ggml-common.h (sha256 0061131b...). Copyright (C) ggml.org contributors, MIT.
"""GGML Q8_1 and Q8_K reference codecs, byte-exact against the compiled C.

Layouts at this pin:
  block_q8_1: fp16 d, fp16 s (= d * sum(qs)), int8 qs[32]  -> 36 bytes / 32 elements
  block_q8_K: f32 d, int8 qs[256], int16 bsums[16]          -> 292 bytes / 256 elements
Upstream has no dequantize_row_q8_1; dequantize_q8_1 is the obvious d*qs decode.
All arithmetic stays float32 (C order). Non-finite inputs are not handled (as in C
they hit assert/UB).
"""
import numpy as np

QK8_1 = 32
QK_K = 256
Q8_1_BYTES = 36
Q8_K_BYTES = 292
_F32 = np.float32



def _roundf_exact(v):
    # v + 0.5 can round up in f32 for v just below .5 (e.g. 0.49999997); use trunc/diff form.
    t = np.trunc(v)
    return t + np.where(np.abs(v - t) >= _F32(0.5), np.sign(v), _F32(0))


def _nearest_int(v):
    """ggml nearest_int: magic-number add, round-half-even; |v| <= 4194303."""
    return np.rint(v).astype(np.int32)


def _blocks(x, size):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32).reshape(-1))
    if x.size % size:
        raise ValueError(f'element count {x.size} not a multiple of {size}')
    return x.reshape(-1, size)


def quantize_q8_1(x):
    b = _blocks(x, QK8_1)
    nb = b.shape[0]
    amax = np.abs(b).max(axis=-1)
    d = (amax / _F32(127)).astype(np.float32)
    nz = d != 0
    idv = np.where(nz, _F32(1) / np.where(nz, d, _F32(1)), _F32(0)).astype(np.float32)
    q = _roundf_exact((b * idv[:, None]).astype(np.float32)).astype(np.int8)
    s = (q.astype(np.int32).sum(axis=-1).astype(np.float32) * d).astype(np.float32)
    out = np.empty((nb, Q8_1_BYTES), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(nb, 2)
    out[:, 2:4] = s.astype(np.float16).view(np.uint8).reshape(nb, 2)
    out[:, 4:] = q.view(np.uint8)
    return out.reshape(-1)


def dequantize_q8_1(raw, n):
    nb = np.asarray(raw, dtype=np.uint8).size // Q8_1_BYTES
    blk = np.frombuffer(np.ascontiguousarray(raw, dtype=np.uint8), np.uint8).reshape(nb, Q8_1_BYTES)
    d = blk[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb)
    q = blk[:, 4:].view(np.int8).astype(np.float32)
    return (q * d[:, None]).astype(np.float32).reshape(-1)[:n]


def quantize_q8_k(x):
    b = _blocks(x, QK_K)
    nb = b.shape[0]
    idx = np.argmax(np.abs(b), axis=-1)  # first strict maximum, like `ax > amax`
    amax = np.abs(b)[np.arange(nb), idx]
    mx = b[np.arange(nb), idx]
    zero = amax == 0
    iscale = (_F32(-127) / np.where(zero, _F32(1), mx)).astype(np.float32)
    v = _nearest_int((iscale[:, None] * b).astype(np.float32))
    q = np.minimum(v, 127).astype(np.int8)
    q[zero] = 0
    bsums = q.astype(np.int32).reshape(nb, QK_K // 16, 16).sum(axis=-1).astype(np.int16)
    bsums[zero] = 0
    d = np.where(zero, _F32(0), _F32(1) / iscale).astype(np.float32)
    out = np.empty((nb, Q8_K_BYTES), dtype=np.uint8)
    out[:, 0:4] = d.view(np.uint8).reshape(nb, 4)
    out[:, 4:260] = q.view(np.uint8)
    out[:, 260:292] = bsums.view(np.uint8).reshape(nb, 32)
    return out.reshape(-1)


def dequantize_q8_k(raw, n):
    nb = np.asarray(raw, dtype=np.uint8).size // Q8_K_BYTES
    blk = np.frombuffer(np.ascontiguousarray(raw, dtype=np.uint8), np.uint8).reshape(nb, Q8_K_BYTES)
    d = blk[:, 0:4].copy().view(np.float32).reshape(nb)
    q = blk[:, 4:260].view(np.int8).astype(np.float32)
    return (d[:, None] * q).astype(np.float32).reshape(-1)[:n]
