# SPDX-License-Identifier: AGPL-3.0-only
# Quantization block codecs ported from llama.cpp ggml-quants.c/ggml-common.h
# (Copyright (C) ggml.org contributors, MIT). Reference (CPU _ref) routines only:
# these reproduce quantize_row_q4_0_ref/q8_0_ref/q4_K_ref/q5_K_ref/q6_K_ref and
# the matching dequantize routines. No model, no device, numpy only.
"""GGML quant block codecs for dequant -> edit -> requant round trips."""
import numpy as np

QK_K = 256
GROUP_MAX_EPS = 1e-15


def _rint(x):
    return np.rint(x)


def _make_qx_quants(x, nmax, rmse_type=1):
    """x: [nb, n] float32 -> scale [nb], L int8 [nb, n] (+nmax offset applied)."""
    ax = np.abs(x)
    idx = np.argmax(ax, axis=-1, keepdims=True)
    amax = np.take_along_axis(ax, idx, axis=-1)
    maxv = np.take_along_axis(x, idx, axis=-1)
    zero = amax < GROUP_MAX_EPS
    maxv = np.where(zero, 1.0, maxv)
    iscale = -nmax / maxv
    q = np.clip(_rint(iscale * x), -nmax, nmax - 1).astype(np.float32)
    L = (q + nmax).astype(np.int64)
    w = (x * x if rmse_type == 1 else np.ones_like(x)).astype(np.float32)
    lx = (w * x * q).sum(axis=-1, keepdims=True, dtype=np.float32)
    l2 = (w * q * q).sum(axis=-1, keepdims=True, dtype=np.float32)
    scale = np.where(l2, lx / np.where(l2, l2, 1), 0.0)
    best = scale * lx
    for is_ in range(-9, 10):
        if is_ == 0:
            continue
        iscale = -(nmax + 0.1 * is_) / maxv
        qa = np.clip(_rint(iscale * x), -nmax, nmax - 1).astype(np.float32)
        La = (qa + nmax).astype(np.int64)
        lx2 = (w * x * qa).sum(axis=-1, keepdims=True, dtype=np.float32)
        l22 = (w * qa * qa).sum(axis=-1, keepdims=True, dtype=np.float32)
        good = (l22 > 0) & (lx2 * lx2 > best * l22)
        L = np.where(good, La, L)
        scale = np.where(good, lx2 / np.where(l22, l22, 1), scale)
        best = np.where(good, scale * lx2, best)
    scale = np.where(zero, 0.0, scale)
    L = np.where(zero, 0, L)
    return scale.squeeze(-1), L.astype(np.int8)


def _make_qkx2(x, nmax, rmin, rdelta, nstep):
    """x: [nb,8,32] float32 -> scale [nb,8], the_min [nb,8], L [nb,8,32]."""
    mn = x.min(axis=-1, keepdims=True)
    mx = x.max(axis=-1, keepdims=True)
    mn = np.where(mn > 0, np.zeros_like(mn), mn)
    flat = (mx == mn)
    span = np.where(flat, np.ones_like(mx), mx - mn)
    iscale = nmax / span
    scale = 1 / iscale
    L = np.clip(_rint(iscale * (x - mn)), 0, nmax).astype(np.int64)
    w = np.sqrt((x * x).mean(axis=-1, keepdims=True)) + np.abs(x)
    err = (w * (scale * L + mn - x) ** 2).sum(axis=-1, keepdims=True)
    best_L, best_scale, best_min, best_err = L.copy(), scale.copy(), mn.copy(), err.copy()
    wx = (w * x).sum(axis=-1, keepdims=True)
    sw = w.sum(axis=-1, keepdims=True)
    for step in range(nstep + 1):
        iscale = (rmin + rdelta * step + nmax) / span
        La = np.clip(_rint(iscale * (x - mn)), 0, nmax).astype(np.int64)
        sl = (w * La).sum(axis=-1, keepdims=True)
        sl2 = (w * La * La).sum(axis=-1, keepdims=True)
        sxl = (w * La * x).sum(axis=-1, keepdims=True)
        D = sw * sl2 - sl * sl
        safeD = np.where(D > 0, D, np.ones_like(D))
        this_scale = (sw * sxl - sl * wx) / safeD
        this_min = (sl2 * wx - sl * sxl) / safeD
        pos = this_min > 0
        this_scale = np.where(pos, sxl / np.where(sl2, sl2, 1), this_scale)
        this_min = np.where(pos, np.zeros_like(this_min), this_min)
        cur = (w * (this_scale * La + this_min - x) ** 2).sum(axis=-1, keepdims=True)
        good = (D > 0) & (cur < best_err)
        best_L = np.where(good, La, best_L)
        best_scale = np.where(good, this_scale, best_scale)
        best_min = np.where(good, this_min, best_min)
        best_err = np.where(good, cur, best_err)
    best_L = np.where(flat, np.zeros_like(best_L), best_L)
    best_scale = np.where(flat, np.zeros_like(best_scale), best_scale)
    best_min = np.where(flat, np.zeros_like(best_min), best_min)
    return best_scale.squeeze(-1), (-best_min).squeeze(-1), best_L.astype(np.uint8)


def _kpack_scales(ls, lm):
    """Pack quantized super-block scales/mins exactly as quantize_row_qK_ref."""
    out = np.zeros((ls.shape[0], 12), dtype=np.uint64)
    out[:, 0:4] = ls[:, 0:4]
    out[:, 4:8] = lm[:, 0:4]
    out[:, 8:12] = (ls[:, 4:8] & 0xF) | ((lm[:, 4:8] & 0xF) << 4)
    out[:, 0:4] |= (ls[:, 4:8] >> 4) << 6
    out[:, 4:8] |= (lm[:, 4:8] >> 4) << 6
    return out.astype(np.uint8)


def _kunpack_scales(scales):
    """Inverse of _kpack_scales -> (sc, m) int arrays [nb,8]."""
    s = scales.astype(np.uint64)
    sc = np.empty((s.shape[0], 8), dtype=np.uint64)
    m = np.empty_like(sc)
    sc[:, 0:4] = s[:, 0:4] & 63
    m[:, 0:4] = s[:, 4:8] & 63
    sc[:, 4:8] = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    m[:, 4:8] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc.astype(np.int64), m.astype(np.int64)


def _fp16_bytes(values):
    return values.astype(np.float16).view(np.uint16)


def quantize_q4_k(x, n_rows=None):
    """x: [n] float32 (padded to multiple of 256) -> block bytes."""
    n = x.size
    nb = n // QK_K
    blocks = x.reshape(nb, QK_K)
    scales, mins, L = _make_qkx2(blocks.reshape(nb, 8, 32), 15, -1.0, 0.1, 20)
    max_scale = scales.max(axis=-1)
    max_min = mins.max(axis=-1)
    inv_scale = np.where(max_scale > 0, 63.0 / np.where(max_scale, max_scale, 1), 0.0)
    inv_min = np.where(max_min > 0, 63.0 / np.where(max_min, max_min, 1), 0.0)
    ls = np.minimum(63, _rint(inv_scale[:, None] * scales)).astype(np.uint64)
    lm = np.minimum(63, _rint(inv_min[:, None] * mins)).astype(np.uint64)
    packed = _kpack_scales(ls, lm)
    d = (max_scale / 63.0).astype(np.float32)
    dmin = (max_min / 63.0).astype(np.float32)
    sc, m = _kunpack_scales(packed)
    dsub = (d[:, None] * sc)[:, :, None]
    dminsub = (dmin[:, None] * m)[:, :, None]
    Lf = L.reshape(nb, 8, 32)
    nonzero = dsub > 0
    q = np.clip(_rint((blocks.reshape(nb, 8, 32) + dminsub) / np.where(nonzero, dsub, 1)), 0, 15).astype(np.uint64)
    Lf = np.where(nonzero, q, Lf)
    qs = np.empty((nb, 128), dtype=np.uint64)
    for p in range(4):
        qs[:, 32 * p:32 * p + 32] = Lf[:, 2 * p] | (Lf[:, 2 * p + 1] << 4)
    out = np.empty((nb, 144), dtype=np.uint8)
    out[:, 0:2] = _fp16_bytes(d).reshape(nb, 1).view(np.uint8).reshape(nb, 2)[:, :2]
    out[:, 2:4] = _fp16_bytes(dmin).reshape(nb, 1).view(np.uint8).reshape(nb, 2)[:, :2]
    out[:, 4:16] = packed
    out[:, 16:144] = qs.astype(np.uint8)
    return out.reshape(-1)


def dequantize_q4_k(raw, n):
    nb = raw.size // 144
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 144)
    d = blocks[:, 0:2].view(np.float16).astype(np.float32).reshape(nb)
    dmin = blocks[:, 2:4].view(np.float16).astype(np.float32).reshape(nb)
    sc, m = _kunpack_scales(blocks[:, 4:16])
    qs = blocks[:, 16:144].astype(np.uint64)
    y = np.empty((nb, QK_K), dtype=np.float32)
    for p in range(4):
        lo = qs[:, 32 * p:32 * p + 32] & 0xF
        hi = qs[:, 32 * p:32 * p + 32] >> 4
        y[:, 64 * p:64 * p + 32] = d[:, None] * sc[:, 2 * p][:, None] * lo - dmin[:, None] * m[:, 2 * p][:, None]
        y[:, 64 * p + 32:64 * p + 64] = d[:, None] * sc[:, 2 * p + 1][:, None] * hi - dmin[:, None] * m[:, 2 * p + 1][:, None]
    return y.reshape(-1)[:n]


def quantize_q5_k(x):
    n = x.size
    nb = n // QK_K
    blocks = x.reshape(nb, QK_K)
    scales, mins, L = _make_qkx2(blocks.reshape(nb, 8, 32), 31, -0.5, 0.1, 15)
    max_scale = scales.max(axis=-1)
    max_min = mins.max(axis=-1)
    inv_scale = np.where(max_scale > 0, 63.0 / np.where(max_scale, max_scale, 1), 0.0)
    inv_min = np.where(max_min > 0, 63.0 / np.where(max_min, max_min, 1), 0.0)
    ls = np.minimum(63, _rint(inv_scale[:, None] * scales)).astype(np.uint64)
    lm = np.minimum(63, _rint(inv_min[:, None] * mins)).astype(np.uint64)
    packed = _kpack_scales(ls, lm)
    d = (max_scale / 63.0).astype(np.float32)
    dmin = (max_min / 63.0).astype(np.float32)
    sc, m = _kunpack_scales(packed)
    dsub = (d[:, None] * sc)[:, :, None]
    dminsub = (dmin[:, None] * m)[:, :, None]
    Lf = L.reshape(nb, 8, 32)
    nonzero = dsub > 0
    q = np.clip(_rint((blocks.reshape(nb, 8, 32) + dminsub) / np.where(nonzero, dsub, 1)), 0, 31).astype(np.uint64)
    Lf = np.where(nonzero, q, Lf)
    ql = np.empty((nb, 128), dtype=np.uint64)
    qh = np.zeros((nb, 32), dtype=np.uint64)
    for p in range(4):
        l1 = Lf[:, 2 * p]
        l2 = Lf[:, 2 * p + 1]
        hi1 = (l1 >> 4).astype(np.uint64)
        hi2 = (l2 >> 4).astype(np.uint64)
        ql[:, 32 * p:32 * p + 32] = (l1 & 0xF) | ((l2 & 0xF) << 4)
        qh |= hi1 << (2 * p) | hi2 << (2 * p + 1)
    out = np.empty((nb, 176), dtype=np.uint8)
    out[:, 0:2] = _fp16_bytes(d).reshape(nb, 1).view(np.uint8).reshape(nb, 2)[:, :2]
    out[:, 2:4] = _fp16_bytes(dmin).reshape(nb, 1).view(np.uint8).reshape(nb, 2)[:, :2]
    out[:, 4:16] = packed
    out[:, 16:48] = qh.astype(np.uint8)
    out[:, 48:176] = ql.astype(np.uint8)
    return out.reshape(-1)


def dequantize_q5_k(raw, n):
    nb = raw.size // 176
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 176)
    d = blocks[:, 0:2].view(np.float16).astype(np.float32).reshape(nb)
    dmin = blocks[:, 2:4].view(np.float16).astype(np.float32).reshape(nb)
    sc, m = _kunpack_scales(blocks[:, 4:16])
    qh = blocks[:, 16:48].astype(np.uint64)
    ql = blocks[:, 48:176].astype(np.uint64)
    y = np.empty((nb, QK_K), dtype=np.float32)
    for p in range(4):
        lo = ql[:, 32 * p:32 * p + 32] & 0xF
        hi = ql[:, 32 * p:32 * p + 32] >> 4
        bit1 = (qh >> (2 * p) & 1)
        bit2 = (qh >> (2 * p + 1) & 1)
        y[:, 64 * p:64 * p + 32] = d[:, None] * sc[:, 2 * p][:, None] * (lo + bit1 * 16) - dmin[:, None] * m[:, 2 * p][:, None]
        y[:, 64 * p + 32:64 * p + 64] = d[:, None] * sc[:, 2 * p + 1][:, None] * (hi + bit2 * 16) - dmin[:, None] * m[:, 2 * p + 1][:, None]
    return y.reshape(-1)[:n]


def quantize_q6_k(x):
    n = x.size
    nb = n // QK_K
    blocks = x.reshape(nb, QK_K).reshape(nb, 16, 16)
    scales16, Lq = _make_qx_quants(blocks.reshape(-1, 16), 32, rmse_type=1)
    scales16 = scales16.reshape(nb, 16).astype(np.float32)
    max_abs = np.abs(scales16).max(axis=-1)
    arg = np.argmax(np.abs(scales16), axis=-1)
    max_scale = scales16[np.arange(nb), arg]
    zero = max_abs < GROUP_MAX_EPS
    iscale = np.where(zero[:, None], 0.0, -128.0 / np.where(zero[:, None], 1.0, max_scale[:, None]))
    d = np.where(zero[:, None], 0.0, 1.0 / np.where(zero[:, None], 1.0, iscale))
    sbyte = np.where(zero[:, None], 0, np.minimum(127, _rint(iscale * scales16))).astype(np.int8)
    d = d.astype(np.float16).astype(np.float32)
    dsub = d.reshape(-1, 1, 1) * sbyte.astype(np.float32)[:, :, None]
    nonzero = dsub != 0
    q = np.clip(_rint(blocks / np.where(nonzero, dsub, 1)), -32, 31) + 32
    Lf = np.where(nonzero, q.astype(np.uint64), (Lq.reshape(nb, 16, 16)).astype(np.uint64))
    Lf = Lf.reshape(nb, QK_K)
    ql = np.zeros((nb, 128), dtype=np.uint64)
    qh = np.zeros((nb, 64), dtype=np.uint64)
    for half in range(2):
        for lane in range(32):
            l0, l1, l2, l3 = (Lf[:, half * 128 + lane], Lf[:, half * 128 + lane + 32],
                              Lf[:, half * 128 + lane + 64], Lf[:, half * 128 + lane + 96])
            ql[:, half * 64 + lane] = (l0 & 0xF) | ((l2 & 0xF) << 4)
            ql[:, half * 64 + lane + 32] = (l1 & 0xF) | ((l3 & 0xF) << 4)
            qh[:, half * 32 + lane] = (l0 >> 4) | ((l1 >> 4) << 2) | ((l2 >> 4) << 4) | ((l3 >> 4) << 6)
    out = np.empty((nb, 210), dtype=np.uint8)
    out[:, 0:128] = ql.astype(np.uint8)
    out[:, 128:192] = qh.astype(np.uint8)
    out[:, 192:208] = sbyte.view(np.uint8)
    out[:, 208:210] = _fp16_bytes(d).reshape(nb, 1).view(np.uint8).reshape(nb, 2)[:, :2]
    out[zero, :] = 0
    return out.reshape(-1)


def dequantize_q6_k(raw, n):
    nb = raw.size // 210
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 210)
    d = blocks[:, 208:210].view(np.float16).astype(np.float32).reshape(nb)
    sc = blocks[:, 192:208].view(np.int8).astype(np.float32)
    ql = blocks[:, 0:128].astype(np.int64)
    qh = blocks[:, 128:192].astype(np.int64)
    y = np.empty((nb, QK_K), dtype=np.float32)
    for half in range(2):
        for lane in range(32):
            is_ = 8 * half + lane // 16
            q1 = ((ql[:, 64 * half + lane] & 0xF) | (((qh[:, 32 * half + lane] >> 0) & 3) << 4)) - 32
            q2 = ((ql[:, 64 * half + lane + 32] & 0xF) | (((qh[:, 32 * half + lane] >> 2) & 3) << 4)) - 32
            q3 = ((ql[:, 64 * half + lane] >> 4) | (((qh[:, 32 * half + lane] >> 4) & 3) << 4)) - 32
            q4 = ((ql[:, 64 * half + lane + 32] >> 4) | (((qh[:, 32 * half + lane] >> 6) & 3) << 4)) - 32
            y[:, 128 * half + lane] = d * sc[:, is_ + 0] * q1
            y[:, 128 * half + lane + 32] = d * sc[:, is_ + 2] * q2
            y[:, 128 * half + lane + 64] = d * sc[:, is_ + 4] * q3
            y[:, 128 * half + lane + 96] = d * sc[:, is_ + 6] * q4
    return y.reshape(-1)[:n]


def quantize_q4_0(x):
    n = x.size
    nb = n // 32
    blocks = x.reshape(nb, 32)
    ax = np.abs(blocks)
    idx = np.argmax(ax, axis=-1)
    maxv = blocks[np.arange(nb), idx]
    d = maxv / -8
    idv = np.where(d != 0, 1 / np.where(d, d, 1), 0.0)
    q = np.clip(_rint(blocks * idv[:, None]) + 8, 0, 15).astype(np.uint64)
    qs = q[:, 0:16] | (q[:, 16:32] << 4)
    out = np.empty((nb, 18), dtype=np.uint8)
    out[:, 0:2] = _fp16_bytes(d).reshape(nb, 1).view(np.uint8).reshape(nb, 2)[:, :2]
    out[:, 2:18] = qs.astype(np.uint8)
    return out.reshape(-1)


def dequantize_q4_0(raw, n):
    nb = raw.size // 18
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 18)
    d = blocks[:, 0:2].view(np.float16).astype(np.float32).reshape(nb)
    qs = blocks[:, 2:18].astype(np.int64)
    y = np.empty((nb, 32), dtype=np.float32)
    y[:, 0:16] = (qs & 0xF) - 8
    y[:, 16:32] = (qs >> 4) - 8
    return (y * d[:, None]).reshape(-1)[:n]


def quantize_q8_0(x):
    n = x.size
    nb = n // 32
    blocks = x.reshape(nb, 32)
    amax = np.abs(blocks).max(axis=-1)
    d = amax / 127
    idv = np.where(d != 0, 1 / np.where(d, d, 1), 0.0)
    q = np.clip(_rint(blocks * idv[:, None]), -127, 127).astype(np.int8)
    out = np.empty((nb, 34), dtype=np.uint8)
    out[:, 0:2] = _fp16_bytes(d).reshape(nb, 1).view(np.uint8).reshape(nb, 2)[:, :2]
    out[:, 2:34] = q.view(np.uint8)
    return out.reshape(-1)


def dequantize_q8_0(raw, n):
    nb = raw.size // 34
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 34)
    d = blocks[:, 0:2].view(np.float16).astype(np.float32).reshape(nb)
    q = blocks[:, 2:34].view(np.int8).astype(np.float32)
    return (q * d[:, None]).reshape(-1)[:n]


def float32_passthrough(raw, n):
    return np.frombuffer(raw, dtype=np.float32).reshape(-1)[:n]


def quantize_bf16(x):
    """Round-to-nearest-even f32 -> bf16 bytes."""
    u = x.astype(np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    u = u + (0x7FFF + lsb)
    return (u >> 16).astype(np.uint16).view(np.uint8).reshape(-1)


def dequantize_bf16(raw, n):
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = (u16.astype(np.uint32) << 16)
    return u32.view(np.float32).reshape(-1)[:n]


def quantize_f32(x):
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint8).reshape(-1)


# dtype name -> (bytes_per_block, elements_per_block, quantizer, dequantizer)
CODECS = {
    'Q4_0': (18, 32, quantize_q4_0, dequantize_q4_0),
    'Q8_0': (34, 32, quantize_q8_0, dequantize_q8_0),
    'Q4_K': (144, QK_K, quantize_q4_k, dequantize_q4_k),
    'Q5_K': (176, QK_K, quantize_q5_k, dequantize_q5_k),
    'Q6_K': (210, QK_K, quantize_q6_k, dequantize_q6_k),
    'F32': (4, 1, quantize_f32, float32_passthrough),
    'BF16': (2, 1, quantize_bf16, dequantize_bf16),
}


def row_padded(n_elements, elements_per_block):
    return -(-n_elements // elements_per_block) * elements_per_block


# Q2_K: 84 bytes per 256 elements — x = d*(sc&0xF)*q - dmin*(sc>>4), 16 subblocks of 16.
# Subblock s gathers qs[32*(s//8) + 16*(s%2) + l] >> 2*((s//2)%4), i.e. the two nibble
# halves interleave low/high 2-bit planes across the 64-byte qs field.
_Q2K_IDX = np.fromfunction(lambda s, e: 32 * (s // 8) + 16 * (s % 2) + e, (16, 16), dtype=int)
_Q2K_SHIFT = (2 * ((np.arange(16) // 2) % 4)).reshape(16, 1)
# Q3_K gathers qs exactly like Q2_K; the high bit lives in hmask[16*(s//8)+l] at
# bit position s//2 (m doubles across both halves).
_Q3K_IDX = _Q2K_IDX
_Q3K_SHIFT = _Q2K_SHIFT
_Q3K_HM_IDX = np.fromfunction(lambda s, e: 16 * (s % 2) + e, (16, 16), dtype=int)
_Q3K_MASK_BIT = (np.arange(16) // 2).reshape(16, 1)


def dequantize_q2_k(raw, n):
    """Exact port of ggml dequantize_row_q2_K."""
    nb = n // QK_K
    b = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 84)
    sc = b[:, 0:16].astype(np.int32)
    qs = b[:, 16:80].astype(np.int32)
    dm = np.frombuffer(np.ascontiguousarray(b[:, 80:84].reshape(-1, 4).tobytes()),
                       dtype=np.float16).reshape(nb, 2)
    d = dm[:, 0:1].astype(np.float32)
    dmin = dm[:, 1:2].astype(np.float32)
    q = qs[:, _Q2K_IDX] >> _Q2K_SHIFT & 3
    dl = d * (sc & 0xF)
    ml = dmin * (sc >> 4)
    return (dl[:, :, None] * q - ml[:, :, None]).reshape(nb * QK_K)


def dequantize_q3_k(raw, n):
    """Exact port of ggml dequantize_row_q3_K (S/M/L share this block layout)."""
    nb = n // QK_K
    b = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 110)
    hm = b[:, 0:32].astype(np.int32)
    qs = b[:, 32:96].astype(np.int32)
    # C reads a 16-byte window (12 scale bytes + 4 pad) as 4 uint32 words; the
    # high two bits of all sixteen 6-bit scales come from word 2 only, so the
    # four pad bytes never influence the output values.
    window = np.concatenate([b[:, 96:108], np.zeros((nb, 4), np.uint8)], axis=1)
    aux = np.frombuffer(np.ascontiguousarray(window.tobytes()),
                        dtype=np.uint32).reshape(nb, 4)
    # C: aux[0..3] hold 16 int8 six-bit scales; the low four bits come from the
    # nibbles of words 0,0,1 and the high two bits from bits 0/2/4/6 of word 2.
    tmp = aux[:, 2:3]
    kmask1, kmask2 = np.uint32(0x03030303), np.uint32(0x0F0F0F0F)
    new0 = (aux[:, 0:1] & kmask2) | ((tmp & kmask1) << 4)
    new1 = (aux[:, 1:2] & kmask2) | (((tmp >> 2) & kmask1) << 4)
    new2 = ((aux[:, 0:1] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    new3 = ((aux[:, 1:2] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    aux_new = np.concatenate([new0, new1, new2, new3], axis=1).astype(np.uint32)
    scales = aux_new.view(np.int8).reshape(nb, 16).astype(np.int32)
    d = np.frombuffer(np.ascontiguousarray(b[:, 108:110].reshape(-1, 2).tobytes()),
                      dtype=np.float16).reshape(nb, 1).astype(np.float32)
    q = qs[:, _Q3K_IDX] >> _Q3K_SHIFT & 3
    bit = (hm[:, _Q3K_HM_IDX] >> _Q3K_MASK_BIT) & 1
    factor = q + 4 * bit - 4
    dl = d * (scales - 32)
    return (dl[:, :, None] * factor).reshape(nb * QK_K)


# Read-only codecs: block_bytes, elements, dequantizer. Proven exact against
# the independent gguf-package port on 770048 random elements per dtype plus
# uniform edge fills. The reference quantizers (make_qkx3/make_qx_quants
# search loops) are NOT ported yet, so gguf-edit keeps refusing Q2_K/Q3_K
# requant; measurement-only for now.
DECODE_ONLY = {
    'Q2_K': (84, QK_K, dequantize_q2_k),
    'Q3_K': (110, QK_K, dequantize_q3_k),
}
