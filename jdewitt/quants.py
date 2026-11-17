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
    maxv = np.where(zero, np.float32(1.0), maxv)
    iscale = -nmax / maxv
    q = np.clip(_rint(iscale * x), -nmax, nmax - 1).astype(np.float32)
    L = (q + nmax).astype(np.int64)
    w = (x * x if rmse_type == 1 else np.ones_like(x)).astype(np.float32)
    zero1 = np.zeros_like(ax[:, :1])
    lx = _fold_last(w * x * q, zero1.copy())
    l2 = _fold_last(w * q * q, zero1.copy())
    scale = np.where(l2, lx / np.where(l2, l2, 1), np.float32(0.0))
    best = scale * lx
    for is_ in range(-9, 10):
        if is_ == 0:
            continue
        iscale = -(np.float32(nmax) + np.float32(0.1) * np.float32(is_)) / maxv  # C's f32 arithmetic
        qa = np.clip(_rint(iscale * x), -nmax, nmax - 1).astype(np.float32)
        La = (qa + nmax).astype(np.int64)
        lx2 = _fold_last(w * x * qa, zero1.copy())
        l22 = _fold_last(w * qa * qa, zero1.copy())
        good = (l22 > 0) & (lx2 * lx2 > best * l22)
        L = np.where(good, La, L)
        scale = np.where(good, lx2 / np.where(l22, l22, 1), scale)
        best = np.where(good, scale * lx2, best)
    scale = np.where(zero, 0.0, scale)
    L = np.where(zero, 0, L)
    return scale.squeeze(-1), L.astype(np.int8)


def _fold_last(a, init):
    """Left-fold accumulation over the last axis, matching C's `for i: sum += a[i]`
    scalar order — pairwise/tree summation flips float32 optimizer ties."""
    acc = init
    for i in range(a.shape[-1]):
        acc = acc + a[..., i:i + 1]
    return acc


def _make_qkx2(x, nmax, rmin, rdelta, nstep, use_mad=False, weights='av'):
    """x: [nb,8,32] float32 -> scale [nb,8], the_min [nb,8], L [nb,8,32].
    Byte-reproducible mirror of make_qkx2_quants (ggml-quants.c master bcda336):
    left-fold float32 sums in C's order, int-as-float without float64 promotion,
    f32 step arithmetic in the iscale numerator, NaN/inf l -> 0 like nearest_int's
    magic-add on degenerate spans, and the C early-return semantics for flat groups.
    weights='av' mirrors q4_K/q5_K callers (av_x + |x|); 'abs' mirrors the q2_K
    caller (plain |x|) — C uses the caller-provided weights verbatim."""
    x = x.astype(np.float32)
    n = x.shape[-1]
    zero = np.zeros_like(x[..., :1])
    w = np.abs(x)
    if weights == 'av':
        av_x = np.sqrt(_fold_last(x * x, zero.copy()) / n)
        w = av_x + w

    mn = mx = x[..., :1].copy()
    sw = w[..., :1].copy()
    wx = w[..., 0:1] * x[..., 0:1]
    for i in range(1, n):
        xi, wi = x[..., i:i + 1], w[..., i:i + 1]
        mn = np.minimum(mn, xi)
        mx = np.maximum(mx, xi)
        sw = sw + wi
        wx = wx + wi * xi
    mn = np.where(mn > 0, np.zeros_like(mn), mn)
    flat = mx == mn
    span = np.where(flat, np.ones_like(mx), mx - mn)
    iscale = nmax / span
    scale = 1 / iscale

    def _L(iscale_):
        q = _rint(iscale_ * (x - mn))
        q = np.where(np.isfinite(q), q, 0.0)
        return np.clip(q, 0, nmax)

    L = _L(iscale)
    best_L = L
    best_scale = scale
    def _err(s_, l_, m_):
        diff = s_ * l_ + m_ - x
        return _fold_last(w * (np.abs(diff) if use_mad else diff * diff), zero.copy())
    best_err = _err(scale, L, mn)
    for step in range(nstep + 1):
        span = mx - mn  # C uses the *running* min; it drifts after an accepted refinement
        num = np.float32(rmin) + np.float32(rdelta) * np.float32(step)  # C's f32 step math
        num = np.float32(num) + np.float32(nmax)
        iscale = num / span
        Laf = _L(iscale)
        sl = _fold_last(w * Laf, zero.copy())
        sl2 = _fold_last(w * Laf * Laf, zero.copy())
        sxl = _fold_last(w * Laf * x, zero.copy())
        D = sw * sl2 - sl * sl
        safeD = np.where(D > 0, D, np.ones_like(D))
        this_scale = (sw * sxl - sl * wx) / safeD
        this_min = (sl2 * wx - sl * sxl) / safeD
        pos = this_min > 0
        this_scale = np.where(pos, sxl / np.where(sl2 != 0, sl2, 1), this_scale)
        this_min = np.where(pos, np.zeros_like(this_min), this_min)
        cur = _err(this_scale, Laf, this_min)
        good = (D > 0) & (cur < best_err)
        best_L = np.where(good, Laf, best_L)
        best_scale = np.where(good, this_scale, best_scale)
        mn = np.where(good, this_min, mn)
        best_err = np.where(good, cur, best_err)
    best_L = np.where(flat, np.zeros_like(best_L), best_L)
    best_scale = np.where(flat, np.zeros_like(best_scale), best_scale)
    return best_scale.squeeze(-1), (-mn).squeeze(-1), best_L.astype(np.uint8)


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
    return sc.astype(np.float32), m.astype(np.float32)  # C multiplies these as float; int64 would promote divisors to float64 and flip ties


def _fp16_bytes(values):
    return values.astype(np.float16).view(np.uint16)


def quantize_q4_k(x, n_rows=None):
    """x: [n] float32 (padded to multiple of 256) -> block bytes."""
    n = x.size
    nb = n // QK_K
    blocks = x.reshape(nb, QK_K)
    scales, mins, L = _make_qkx2(blocks.reshape(nb, 8, 32), 15, -1.0, 0.1, 20)
    max_scale = np.where(scales > 0, scales, 0.0).max(axis=-1)  # C folds strict > onto +0
    max_min = np.where(mins > 0, mins, 0.0).max(axis=-1)        # (numpy max() can return -0.0)
    inv_scale = np.where(max_scale > 0, 63.0 / np.where(max_scale, max_scale, 1), 0.0)
    inv_min = np.where(max_min > 0, 63.0 / np.where(max_min, max_min, 1), 0.0)
    ls = np.minimum(63, _rint(inv_scale[:, None] * scales)).astype(np.uint64)
    lm = np.minimum(63, _rint(inv_min[:, None] * mins)).astype(np.uint64)
    packed = _kpack_scales(ls, lm)
    d = (max_scale / 63.0).astype(np.float32)
    dmin = (max_min / 63.0).astype(np.float32)
    # C requantizes from the stored f16-rounded scales, not the float ones
    d = d.astype(np.float16).astype(np.float32)
    dmin = dmin.astype(np.float16).astype(np.float32)
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
    max_scale = np.where(scales > 0, scales, 0.0).max(axis=-1)  # C folds strict > onto +0
    max_min = np.where(mins > 0, mins, 0.0).max(axis=-1)        # (numpy max() can return -0.0)
    inv_scale = np.where(max_scale > 0, 63.0 / np.where(max_scale, max_scale, 1), 0.0)
    inv_min = np.where(max_min > 0, 63.0 / np.where(max_min, max_min, 1), 0.0)
    ls = np.minimum(63, _rint(inv_scale[:, None] * scales)).astype(np.uint64)
    lm = np.minimum(63, _rint(inv_min[:, None] * mins)).astype(np.uint64)
    packed = _kpack_scales(ls, lm)
    d = (max_scale / 63.0).astype(np.float32)
    dmin = (max_min / 63.0).astype(np.float32)
    # C requantizes from the stored f16-rounded scales, not the float ones
    d = d.astype(np.float16).astype(np.float32)
    dmin = dmin.astype(np.float16).astype(np.float32)
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
    # C: MIN(15, (int8_t)(x0 + 8.5f)) — truncation toward zero, not rounding
    q = np.clip(np.trunc(blocks * idv[:, None] + np.float32(8.5)), 0, 15).astype(np.uint64)
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
    x0 = blocks * idv[:, None]
    # C: roundf — half away from zero (np.rint rounds half to even, a real difference)
    q = np.where(x0 >= 0, np.trunc(x0 + np.float32(0.5)),
                 -np.trunc(-x0 + np.float32(0.5))).astype(np.int8)
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


def quantize_q4_1(x):
    """ggml quantize_row_q4_1_ref byte-exact: d=(max-min)/15, m=min,
    xi = MIN(15, trunc_toward_zero((x-min)/d + 0.5))."""
    nb = x.size // 32
    b = x.reshape(nb, 32)
    mn = b.min(axis=-1)
    mx = b.max(axis=-1)
    d = (mx - mn) / 15.0
    idv = np.where(d != 0, 1 / np.where(d, d, 1), 0.0)
    q = np.clip(np.trunc((b - mn[:, None]) * idv[:, None] + 0.5), 0, 15).astype(np.uint64)
    qs = q[:, 0:16] | (q[:, 16:32] << 4)
    out = np.empty((nb, 20), dtype=np.uint8)
    out[:, 0:2] = _fp16_bytes(d).view(np.uint8).reshape(nb, 2)
    out[:, 2:4] = _fp16_bytes(mn).view(np.uint8).reshape(nb, 2)
    out[:, 4:20] = qs.astype(np.uint8)
    return out.reshape(-1)


def dequantize_q4_1(raw, n):
    nb = raw.size // 20
    b = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 20)
    d = b[:, 0:2].view(np.float16).astype(np.float32).reshape(nb)
    m = b[:, 2:4].view(np.float16).astype(np.float32).reshape(nb)
    qs = b[:, 4:20].astype(np.int64)
    y = np.empty((nb, 32), dtype=np.float32)
    y[:, 0:16] = qs & 0xF
    y[:, 16:32] = qs >> 4
    return (y * d[:, None] + m[:, None]).reshape(-1)[:n]


def _pack_qh_bit4(q5):
    """C memcpy(qh, uint32): bit 4 of quant j -> bit j, quant 16+j -> bit 16+j."""
    bits = ((q5 >> 4) & 1).astype(np.uint32)
    packed = (bits.reshape(-1, 32) << np.arange(32, dtype=np.uint32)).sum(axis=-1)
    return packed.astype(np.uint32).view(np.uint8).reshape(-1, 4)


def quantize_q5_0(x):
    """ggml quantize_row_q5_0_ref byte-exact: d=max/-16 (signed max),
    xi = MIN(31, trunc(x/d + 16.5)); low nibbles in qs, bit 4 in qh."""
    nb = x.size // 32
    b = x.reshape(nb, 32)
    ax = np.abs(b)
    idx = np.argmax(ax, axis=-1)
    maxv = b[np.arange(nb), idx]
    d = maxv / -16
    idv = np.where(d != 0, 1 / np.where(d, d, 1), 0.0)
    q = np.clip(np.trunc(b * idv[:, None] + 16.5), 0, 31).astype(np.uint64)
    qs = (q[:, 0:16] & 0xF) | ((q[:, 16:32] & 0xF) << 4)
    out = np.empty((nb, 22), dtype=np.uint8)
    out[:, 0:2] = _fp16_bytes(d).view(np.uint8).reshape(nb, 2)
    out[:, 2:6] = _pack_qh_bit4(q)
    out[:, 6:22] = qs.astype(np.uint8)
    return out.reshape(-1)


def dequantize_q5_0(raw, n):
    nb = raw.size // 22
    b = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 22)
    d = b[:, 0:2].view(np.float16).astype(np.float32).reshape(nb)
    qs = b[:, 6:22].astype(np.int64)
    hi_bits = np.unpackbits(b[:, 2:6], bitorder='little', axis=1).reshape(nb, 32)
    y = np.empty((nb, 32), dtype=np.float32)
    y[:, 0:16] = (qs & 0xF) | (hi_bits[:, 0:16] << 4)
    y[:, 16:32] = (qs >> 4) | (hi_bits[:, 16:32] << 4)
    # ggml dequantize_row_q5_0 stores an offset code: x0 = (code - 16) * d
    return ((y - 16.0) * d[:, None]).reshape(-1)[:n]


def quantize_q5_1(x):
    """ggml quantize_row_q5_1_ref byte-exact: d=(max-min)/31, m=min,
    xi = trunc((x-min)/d + 0.5) with C uint8 cast (no MIN)."""
    nb = x.size // 32
    b = x.reshape(nb, 32)
    mn = b.min(axis=-1)
    mx = b.max(axis=-1)
    d = (mx - mn) / 31.0
    idv = np.where(d != 0, 1 / np.where(d, d, 1), 0.0)
    q = (np.trunc((b - mn[:, None]) * idv[:, None] + 0.5) % 256).astype(np.uint64)
    qs = (q[:, 0:16] & 0xF) | ((q[:, 16:32] & 0xF) << 4)
    out = np.empty((nb, 24), dtype=np.uint8)
    out[:, 0:2] = _fp16_bytes(d).view(np.uint8).reshape(nb, 2)
    out[:, 2:4] = _fp16_bytes(mn).view(np.uint8).reshape(nb, 2)
    out[:, 4:8] = _pack_qh_bit4(q)
    out[:, 8:24] = qs.astype(np.uint8)
    return out.reshape(-1)


def dequantize_q5_1(raw, n):
    nb = raw.size // 24
    b = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 24)
    d = b[:, 0:2].view(np.float16).astype(np.float32).reshape(nb)
    m = b[:, 2:4].view(np.float16).astype(np.float32).reshape(nb)
    qs = b[:, 8:24].astype(np.int64)
    hi_bits = np.unpackbits(b[:, 4:8], bitorder='little', axis=1).reshape(nb, 32)
    y = np.empty((nb, 32), dtype=np.float32)
    y[:, 0:16] = (qs & 0xF) | (hi_bits[:, 0:16] << 4)
    y[:, 16:32] = (qs >> 4) | (hi_bits[:, 16:32] << 4)
    return (y * d[:, None] + m[:, None]).reshape(-1)[:n]


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
    'Q4_1': (20, 32, quantize_q4_1, dequantize_q4_1),
    'Q5_0': (22, 32, quantize_q5_0, dequantize_q5_0),
    'Q5_1': (24, 32, quantize_q5_1, dequantize_q5_1),
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


def _pack_2bit_qk(L):
    """L: [nb,256] uint64 -> ggml 2-bit packing: byte 32g+l holds quants
    128g+l | 128g+32+l <<2 | 128g+64+l <<4 | 128g+96+l <<6 (both q2/q3_K)."""
    q4 = L.reshape(-1, 2, 4, 32)
    return (q4 << np.array([0, 2, 4, 6], dtype=np.uint64).reshape(1, 1, 4, 1)).sum(axis=2).reshape(-1, 64)


def quantize_q2_k(x):
    """Byte-reproducible mirror of ggml quantize_row_q2_K_ref: per 16-element
    subblock make_qkx2 (n=16, nmax=3, rmin=-0.5, rdelta=0.1, nstep=15, use_mad),
    f16-packed 4+4-bit scales/mins, then requant with the stored bytes."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, QK_K)
    xs = x.reshape(-1, 16, 16)
    scale, mins, Lq = _make_qkx2(xs, 3, -0.5, 0.1, 15, use_mad=True, weights='abs')
    L = Lq.astype(np.uint64).reshape(xs.shape)
    max_scale = np.where(scale > 0, scale, 0.0).max(axis=-1)
    max_min = np.where(mins > 0, mins, 0.0).max(axis=-1)
    pos_s = max_scale > 0
    pos_m = max_min > 0
    sdiv = np.where(pos_s, np.float32(15.0) / np.where(pos_s, max_scale, 1.0), 0.0)
    mdiv = np.where(pos_m, np.float32(15.0) / np.where(pos_m, max_min, 1.0), 0.0)
    low = np.where(pos_s[:, None], _rint(sdiv[:, None] * scale), 0.0).astype(np.int64)
    high = np.where(pos_m[:, None], _rint(mdiv[:, None] * mins), 0.0).astype(np.int64)
    sc = ((low | (high << 4)) & 0xFF)
    dg = np.where(pos_s, max_scale / np.float32(15.0), 0.0)
    mg = np.where(pos_m, max_min / np.float32(15.0), 0.0)
    d_f16 = dg.astype(np.float16).astype(np.float32)
    m_f16 = mg.astype(np.float16).astype(np.float32)
    dsub = d_f16[:, None] * (sc & 0xF)
    dmsub = m_f16[:, None] * (sc >> 4)
    mask = dsub != 0
    with np.errstate(invalid='ignore', divide='ignore'):
        lq = np.clip(_rint((xs + dmsub[:, :, None]) /
                           np.where(mask[:, :, None], dsub[:, :, None], 1.0)), 0, 3)
    L = np.where(mask[:, :, None], lq.astype(np.uint64), L)
    out = np.empty((x.shape[0], 84), dtype=np.uint8)
    out[:, 0:16] = sc.astype(np.uint8)
    out[:, 16:80] = _pack_2bit_qk(L).astype(np.uint8)
    out[:, 80:82] = _fp16_bytes(dg).view(np.uint8).reshape(-1, 2)
    out[:, 82:84] = _fp16_bytes(mg).view(np.uint8).reshape(-1, 2)
    return out.reshape(-1)


# Q2_K joins CODECS here (not the literal above) because its quantizer is defined
# after CODECS; Q3_K stays decode-only: a quantize_row_q3_K_ref port was written
# and compiled-C falsified but is only ~16% byte-exact on gaussian data (zero and
# near-zero cases pass) — kept out of CODECS until byte-exact. See docs/PLAN-NEXT.
CODECS['Q2_K'] = (84, QK_K, quantize_q2_k, dequantize_q2_k)


# Q2_K's quantizer above is a byte-exact mirror of ggml quantize_row_q2_K_ref
# (compiled-C falsified, scripts/ggml_ref/ref_23k.c, 13 distributions incl. 4
# real F32 checkpoint tensors). It exposed a latent caller-weights issue in
# _make_qkx2: q2_K passes plain |x| weights while q4_K/q5_K pass av_x+|x|; the
# dequantizers below remain the previously proven ports. Q3_K stays decode-only
# — see the quantize_q2_k docstring region and docs/PLAN-NEXT.md for the
# falsifier verdict.
# IQ4_XS: 256 elements per 136 bytes. Layout (ggml-common.h block_iq4_xs):
#   d (f16) | scales_h (u16) | scales_l[4] | qs[128]
# per 32-element sub-block ib: ls = low nibble pair | 2 high bits from scales_h,
# dl = d * (ls - 32); values from kvalues_iq4nl (int8, unscaled).
# Ported from dequantize_row_iq4_xs (vendored ggml-quants.c:2759); cross-checked
# byte-for-byte against the pinned gguf.quants port below.
_KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                           1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


def dequantize_iq4_xs(raw, n):
    nb = raw.size // 136
    if nb * 136 != raw.size:
        raise ValueError(f'iq4_xs input {raw.size} is not a whole number of 136-byte blocks')
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 136)
    d = blocks[:, 0:2].view(np.float16).astype(np.float32).reshape(nb)
    scales_h = blocks[:, 2:4].view(np.uint16).astype(np.uint32).reshape(nb)
    scales_l = blocks[:, 4:8].astype(np.uint32)
    qs = blocks[:, 8:].reshape(nb, 8, 16)
    y = np.empty((nb, QK_K), dtype=np.float32)
    for ib in range(8):
        lo = (scales_l[:, ib // 2] >> (4 * (ib % 2))) & 0xF
        hi = (scales_h >> (2 * ib)) & 3
        dl = d * ((lo | (hi << 4)).astype(np.int32) - 32)
        q = qs[:, ib, :]
        y[:, 32 * ib:32 * ib + 16] = dl[:, None] * _KVALUES_IQ4NL[q & 0xF]
        y[:, 32 * ib + 16:32 * ib + 32] = dl[:, None] * _KVALUES_IQ4NL[q >> 4]
    out = y.reshape(-1)
    if out.size < n:
        raise ValueError(f'iq4_xs decoded {out.size} values, wanted {n}')
    return out[:n]


def dequantize_iq4_nl(raw: np.ndarray, n: int) -> np.ndarray:
    """numpy port of dequantize_row_iq4_nl from ggml-quants.c (master
    bcda336): 18-byte blocks (ggml_half d + 16 packed nibble bytes) of 32
    elements. Bit-exact vs a compiled C harness of the upstream function on
    65,536 random blocks and 32,768 real checkpoint blocks; numpy and
    compiled-C and the pinned HF port all agree element-for-element."""
    blocks = raw.reshape(-1, 18)
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    qs = blocks[:, 2:18]
    lo = (qs & 0xF).astype(np.intp)
    hi = (qs >> 4).astype(np.intp)
    out = np.empty((blocks.shape[0], 32), dtype=np.float32)
    out[:, 0:16] = d * _KVALUES_IQ4NL[lo]
    out[:, 16:32] = d * _KVALUES_IQ4NL[hi]
    return out.reshape(n)


DECODE_ONLY = {
    'IQ4_XS': (136, QK_K, dequantize_iq4_xs),
    'Q3_K': (110, QK_K, dequantize_q3_k),
    'IQ4_NL': (18, 32, dequantize_iq4_nl),
}
