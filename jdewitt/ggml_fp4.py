# SPDX-License-Identifier: AGPL-3.0-or-later
"""GGML-side MXFP4 / NVFP4 block codecs, byte-exact with ggml-quants.c.

Layouts (ggml-common.h):
  block_mxfp4: u8 e (E8M0) + 16 bytes qs  = 17 bytes / 32 elements;
               qs[j] low nibble = element j, high nibble = element j+16.
  block_nvfp4: 4 x u8 d (UE4M3, one per 16-element sub-block) + 32 bytes qs
               = 36 bytes / 64 elements; within sub-block s,
               qs[s*8+j] low nibble = element j, high nibble = element j+8.
Nibbles index kvalues_fp4 (doubled E2M1, sign in bit 3) -- not the safetensors
digit layout in fp4.py. Quantize picks the nearest kvalue by float32 |kv*d - x|
with first-minimum tie-breaking (NaN/inf inputs land on index 0, as in C).
Scales are ggml's own (E8M0 = floor(log2(amax)) - 2, UE4M3 = round(amax/6)),
not fp4.py's ceil-based ones.
"""
import numpy as np

MXFP4_BLOCK, MXFP4_BYTES = 32, 17
NVFP4_BLOCK, NVFP4_SUB, NVFP4_BYTES = 64, 16, 36

KVALUES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int8)
_KV_F32 = KVALUES.astype(np.float32)


def _e8m0_half_table():
    x = np.arange(256, dtype=np.uint32)
    bits = np.where(x < 2, np.uint32(0x00200000) << np.minimum(x, 1), (x - 1) << 23).astype(np.uint32)
    return bits.view(np.float32)


def _ue4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        if b in (0, 0x7F):
            continue
        e, m = (b >> 3) & 0xF, b & 7
        raw = m * 2.0 ** -9 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = np.float32(raw * 0.5)
    return t


E8M0_HALF = _e8m0_half_table()
UE4M3_HALF = _ue4m3_table()


def _log2f(a):
    """float32 log2 the way a correctly rounded log2f gives it (via float64)."""
    return np.log2(a.astype(np.float64)).astype(np.float32)


def _e8m0_from_amax(amax):
    """(uint8_t)(floorf(log2f(amax)) - 2 + 127) for amax > 0, else 0 (x86 cvttss2si semantics)."""
    with np.errstate(all='ignore'):
        v = np.floor(_log2f(np.where(amax > 0, amax, np.float32(1)))) - np.float32(2) + np.float32(127)
    v = v.astype(np.float32)
    ok = np.isfinite(v)
    out = np.where(ok, np.trunc(np.where(ok, v, 0)).astype(np.int64) & 0xFF, 0)  # inf -> INT_MIN -> 0
    return np.where(amax > 0, out, 0).astype(np.uint8)


def _best_index(x, d):
    """x: (..., n) float32, d: (...,1) float32 -> uint8 index; mirrors best_index_mxfp4."""
    with np.errstate(all='ignore'):
        err = np.abs(_KV_F32 * d[..., None] - x[..., None])  # (..., n, 16) float32
    first_nan = np.isnan(err[..., 0])
    err = np.where(np.isnan(err), np.float32(np.inf), err)
    idx = np.argmin(err, axis=-1)  # first minimum == strict '<' scan
    return np.where(first_nan, 0, idx).astype(np.uint8)


def _amax(x):
    # C: `if (amax < fabsf(v)) amax = fabsf(v)` -- NaN never wins
    a = np.abs(x)
    return np.max(np.where(np.isnan(a), np.float32(0), a), axis=-1)


def quantize_mxfp4(x):
    """float32 array (size multiple of 32) -> uint8 array of nb*17 ggml block_mxfp4 bytes."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).reshape(-1)
    if x.size % MXFP4_BLOCK:
        raise ValueError('size must be a multiple of 32')
    b = x.reshape(-1, MXFP4_BLOCK)
    e = _e8m0_from_amax(_amax(b))
    d = E8M0_HALF[e][:, None]
    lo = _best_index(b[:, :16], d)
    hi = _best_index(b[:, 16:], d)
    out = np.empty((b.shape[0], MXFP4_BYTES), dtype=np.uint8)
    out[:, 0] = e
    out[:, 1:] = lo | (hi << 4)
    return out.reshape(-1)


def dequantize_mxfp4(raw, n=None):
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1, MXFP4_BYTES)
    d = E8M0_HALF[raw[:, 0]][:, None]
    qs = raw[:, 1:]
    out = np.empty((raw.shape[0], MXFP4_BLOCK), dtype=np.float32)
    with np.errstate(over='ignore'):
        out[:, :16] = _KV_F32[qs & 15] * d
        out[:, 16:] = _KV_F32[qs >> 4] * d
    out = out.reshape(-1)
    return out if n is None else out[:n]


def _ue4m3_from_f32(x):
    """ggml_fp32_to_ue4m3, vectorised."""
    x = np.asarray(x, dtype=np.float32)
    pos = x > 0  # False for NaN
    xc = np.minimum(np.where(pos, x, np.float32(1)), np.float32(448.0))
    bits = xc.view(np.uint32)
    fexp = ((bits >> 23) & 0xFF).astype(np.int64) - 127
    fman = ((bits >> 20) & 7).astype(np.int64)
    uexp = fexp + 7
    with np.errstate(all='ignore'):
        sub = (xc * np.float32(512.0) + np.float32(0.5)).astype(np.int64)
    sub = np.minimum(sub, 7)
    sub = np.where(sub < 1, 0, sub)
    rnd = ((bits >> 19) & 1).astype(np.int64)
    man = fman + rnd
    carry = man > 7
    man = np.where(carry, 0, man)
    e2 = uexp + carry
    normal = np.where(e2 >= 15, 0x7E, (e2 << 3) | man)
    res = np.where(uexp <= 0, sub, np.where(uexp >= 15, 0x7E, normal))
    return np.where(pos, res, 0).astype(np.uint8)


def quantize_nvfp4(x):
    """float32 array (size multiple of 64) -> uint8 array of nb*36 ggml block_nvfp4 bytes."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).reshape(-1)
    if x.size % NVFP4_BLOCK:
        raise ValueError('size must be a multiple of 64')
    sb = x.reshape(-1, 4, NVFP4_SUB)  # (nb, sub, 16)
    with np.errstate(all='ignore'):
        ue = _ue4m3_from_f32(_amax(sb) / np.float32(6.0))  # (nb, 4)
    d = UE4M3_HALF[ue][..., None]
    lo = _best_index(sb[..., :8], d)
    hi = _best_index(sb[..., 8:], d)
    out = np.empty((sb.shape[0], NVFP4_BYTES), dtype=np.uint8)
    out[:, :4] = ue
    out[:, 4:] = (lo | (hi << 4)).reshape(-1, 32)
    return out.reshape(-1)


def dequantize_nvfp4(raw, n=None):
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1, NVFP4_BYTES)
    d = UE4M3_HALF[raw[:, :4]][..., None]  # (nb,4,1)
    qs = raw[:, 4:].reshape(-1, 4, 8)
    out = np.empty((raw.shape[0], 4, NVFP4_SUB), dtype=np.float32)
    out[..., :8] = _KV_F32[qs & 15] * d
    out[..., 8:] = _KV_F32[qs >> 4] * d
    out = out.reshape(-1)
    return out if n is None else out[:n]
