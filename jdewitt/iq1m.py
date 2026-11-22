# SPDX-License-Identifier: AGPL-3.0-or-later
"""IQ1_M encoder: iq2 data construction (2048 1-bit grid, nwant=3) and the reference quantizer.

Ported from ggml-quants.c at the revision pinned in jdewitt/iq_tables.py
(iq2xs_init_impl for GGML_TYPE_IQ1_M, iq1_find_best_neighbour2 and
quantize_row_iq1_m_impl), with quant_weights == NULL (weight = x*x). The
verbatim C lives in scripts/ggml_ref/ref_iq1m.c; every f32 accumulation order
and acceptance test is reproduced, and the C `qsort` tie order (glibc >= 2.37
mergesort, i.e. stable) is reproduced with a stable argsort.

kgrid_1bit_2048 is derived from IQ1S_GRID (signed-byte form of the same grid:
2-bit code = byte + 1). The neighbour lists are built lazily and only for the
6561 tuples whose digits are all in {0,1,2}, the only ones the quantizer can
query (upstream builds all 43692; the other entries are never read here).
"""
from __future__ import annotations

import numpy as np

from .iq_tables import IQ1S_GRID

QK_K = 256
NGRID = 2048
KMAP_SIZE = 43692
NWANT = 3
BLOCK_BYTES = 56
GROUP_MAX_EPS_IQ1_M = np.float32(1e-7)
_F = np.float32
IQ1M_DELTA = _F(0.125)
X_P = np.array([-1 + IQ1M_DELTA, IQ1M_DELTA, 1 + IQ1M_DELTA], dtype=np.float32)
X_M = np.array([-1 - IQ1M_DELTA, -IQ1M_DELTA, 1 - IQ1M_DELTA], dtype=np.float32)
MASKS = np.array([0x00, 0x80, 0x08, 0x88], dtype=np.int64)

_CACHE = {}


def _build():
    if 'g' in _CACHE:
        return _CACHE['g']
    signed = np.ascontiguousarray(IQ1S_GRID.astype('<u8')).view(np.int8).reshape(NGRID, 8)
    code = signed.astype(np.int64) + 1          # (pg - 1) / 2
    grid = (2 * code + 1).astype(np.int8)       # upstream the_grid bytes
    if code.min() < 0 or code.max() > 2:
        raise ValueError('IQ1_M grid has codes outside {0,1,2}')
    pw = (1 << (2 * np.arange(8))).astype(np.int64)
    packed = (code * pw).sum(axis=1)
    kmap = np.full(KMAP_SIZE, -1, dtype=np.int64)
    for i in range(NGRID):
        kmap[packed[i]] = i
    # tuples with digits in {0,1,2} that are off the grid
    digits = np.indices((3,) * 8).reshape(8, -1).T  # (6561, 8) digit k = column k
    tuples = (digits.astype(np.int64) * pw).sum(axis=1)
    miss = tuples[kmap[tuples] < 0]
    g64 = grid.astype(np.int64)
    idx = np.arange(NGRID)
    neigh = {}
    for t in miss:
        pos = 2 * ((int(t) >> (2 * np.arange(8))) & 3) + 1
        dist = ((g64 - pos) ** 2).sum(axis=1)
        order = np.lexsort((idx, dist))      # == qsort on (d2, j) pairs
        sd = dist[order]
        # distance classes: keep entries until the (NWANT+1)-th distinct distance
        cls = np.concatenate(([0], np.cumsum(sd[1:] > sd[:-1])))
        neigh[int(t)] = order[cls < NWANT]
    width = max(len(v) for v in neigh.values())
    padded = np.full((KMAP_SIZE, width), -1, dtype=np.int64)
    for t, v in neigh.items():
        padded[t, :len(v)] = v
    _CACHE['g'] = (grid, kmap, padded)
    return _CACHE['g']


def _nearest_int(v):
    return np.rint(v)  # == ggml nearest_int for |v| <= 4194303


def _pack8(codes):
    out = np.zeros(codes.shape[:-1], dtype=np.int64)
    for i in range(8):
        out |= codes[..., i].astype(np.int64) << (2 * i)
    return out


def _search(xb, weight, sorted_idx):
    """Exhaustive 3-group split search. xb, weight: (M,16) f32. Returns scale, i1, i2, k, found."""
    m = xb.shape[0]
    rows = np.arange(m)[:, None]
    xs = xb[rows, sorted_idx]
    ws = weight[rows, sorted_idx]
    wx = (ws * xs).astype(_F)
    half = (sorted_idx >= 8)
    px = np.zeros((2, m, 17), dtype=np.float32)
    pw = np.zeros((2, m, 17), dtype=np.float32)
    for j in range(16):
        for h in range(2):
            here = (half[:, j] == bool(h))
            px[h, :, j + 1] = np.where(here, px[h, :, j] + wx[:, j], px[h, :, j])
            pw[h, :, j + 1] = np.where(here, pw[h, :, j] + ws[:, j], pw[h, :, j])
    txs = px[:, :, 16]
    tws = pw[:, :, 16]
    p2, m2 = X_P[2], X_M[2]
    cp1 = _F(X_P[0] * X_P[0] - X_P[1] * X_P[1])
    cp2 = _F(X_P[1] * X_P[1] - X_P[2] * X_P[2])
    cm1 = _F(X_M[0] * X_M[0] - X_M[1] * X_M[1])
    cm2 = _F(X_M[1] * X_M[1] - X_M[2] * X_M[2])
    pairs = [(i1, i2) for i1 in range(17) for i2 in range(i1, 17)]
    I1 = np.array([p[0] for p in pairs])
    I2 = np.array([p[1] for p in pairs])
    qx_p, qx_m, q2_p, q2_m = [], [], [], []
    for h in range(2):
        sx = (px[h][:, I1] + px[h][:, I2]).astype(_F)
        qx_p.append((txs[h][:, None] * p2 - sx).astype(_F))
        qx_m.append((txs[h][:, None] * m2 - sx).astype(_F))
        q2_p.append((((tws[h][:, None] * p2) * p2 + pw[h][:, I1] * cp1) + pw[h][:, I2] * cp2).astype(_F))
        q2_m.append((((tws[h][:, None] * m2) * m2 + pw[h][:, I1] * cm1) + pw[h][:, I2] * cm2).astype(_F))
    sumqx = np.stack([qx_p[0] + qx_p[1], qx_p[0] + qx_m[1], qx_m[0] + qx_p[1], qx_m[0] + qx_m[1]], axis=-1)
    sumq2 = np.stack([q2_p[0] + q2_p[1], q2_p[0] + q2_m[1], q2_m[0] + q2_p[1], q2_m[0] + q2_m[1]], axis=-1)
    sumqx = sumqx.reshape(m, -1).astype(_F)
    sumq2 = sumq2.reshape(m, -1).astype(_F)
    cand_i1 = np.repeat(I1, 4)
    cand_i2 = np.repeat(I2, 4)
    cand_k = np.tile(np.arange(4), len(pairs))
    best = np.full(m, -np.finfo(np.float32).max, dtype=np.float32)
    scale = np.zeros(m, dtype=np.float32)
    bi1 = np.full(m, -1)
    bi2 = np.full(m, -1)
    bk = np.full(m, -1)
    with np.errstate(all='ignore'):
        for c in range(sumqx.shape[1]):
            qx = sumqx[:, c]
            q2 = sumq2[:, c]
            upd = (q2 > 0) & ((qx * qx) > (best * q2))
            if upd.any():
                ns = (qx / q2).astype(_F)
                scale = np.where(upd, ns, scale)
                best = np.where(upd, (ns * qx).astype(_F), best)
                bi1 = np.where(upd, cand_i1[c], bi1)
                bi2 = np.where(upd, cand_i2[c], bi2)
                bk = np.where(upd, cand_k[c], bk)
    return scale.astype(_F), bi1, bi2, bk


def _xx_for(best_k, k):
    """(M,3) f32 value table for half k given shift code (0: ++, 1: +-, 2: -+, 3: --)."""
    use_p = (best_k < 2) if k == 0 else (best_k % 2 == 0)
    return np.where(use_p[:, None], X_P[None, :], X_M[None, :]).astype(np.float32)


def _neighbour2(padded, grid, tuples, xval, weight, scale, xg):
    """iq1_find_best_neighbour2 for off-grid tuples. xval/weight (n,8), scale (n,), xg (n,3)."""
    idxs = padded[tuples]
    valid = idxs >= 0
    safe = np.where(valid, idxs, 0)
    code = (grid[safe].astype(np.int64) - 1) // 2              # (n,w,8)
    q = xg[np.arange(code.shape[0])[:, None, None], code]
    d2 = np.zeros(idxs.shape, dtype=np.float32)
    for i in range(8):
        diff = (scale[:, None] * q[:, :, i] - xval[:, None, i]).astype(_F)
        d2 = (d2 + (weight[:, None, i] * diff) * diff).astype(_F)
    d2 = np.where(valid, d2, _F(np.inf))
    best = np.argmin(d2, axis=1)
    chosen = np.take_along_axis(idxs, best[:, None], axis=1)[:, 0]
    # C: best_score starts at FLT_MAX with strict <, so an all-inf row finds nothing
    if not (np.take_along_axis(d2, best[:, None], axis=1)[:, 0] < np.finfo(np.float32).max).all():
        raise ValueError('IQ1_M: no grid point found (undefined upstream path)')
    return chosen


def _quantize_superblocks(x):
    nb = x.shape[0]
    grid, kmap, padded = _build()
    xb = x.reshape(nb * 16, 16)
    m = xb.shape[0]
    weight = (xb * xb).astype(_F)
    maxv = np.abs(xb).max(axis=1)
    dead = maxv < GROUP_MAX_EPS_IQ1_M
    order = np.argsort(xb, axis=1, kind='stable')
    scale, i1, i2, bk = _search(xb, weight, order)
    dead = dead | (i1 < 0)
    # L from the sorted order
    rank = np.empty_like(order)
    rank[np.arange(m)[:, None], order] = np.arange(16)[None, :]
    L = np.where(rank < i1[:, None], 0, np.where(rank < i2[:, None], 1, 2)).astype(np.int64)
    neg = scale < 0
    L = np.where(neg[:, None], 2 - L, L)
    scale = np.where(neg, -scale, scale).astype(_F)
    bk = np.where(neg, 3 - bk, bk)
    all_on = np.ones(m, dtype=bool)
    index = np.zeros((m, 2), dtype=np.int64)
    xgs = []
    for k in range(2):
        xg = _xx_for(np.where(dead, 0, bk), k)
        xgs.append(xg)
        tup = _pack8(L[:, 8 * k:8 * k + 8])
        gi = kmap[tup]
        miss = (gi < 0) & ~dead
        if miss.any():
            sel = np.nonzero(miss)[0]
            if (padded[tup[sel], 0] < 0).any():
                raise ValueError('IQ1_M: tuple outside the neighbour table')
            chosen = _neighbour2(padded, grid, tup[sel], xb[sel, 8 * k:8 * k + 8],
                                 weight[sel, 8 * k:8 * k + 8], scale[sel], xg[sel])
            gi = gi.copy()
            gi[sel] = chosen
            all_on[sel] = False
        index[:, k] = gi
    # rescale blocks that left the grid
    sumqx = np.zeros(m, dtype=np.float32)
    sumq2 = np.zeros(m, dtype=np.float32)
    with np.errstate(all='ignore'):
        for k in range(2):
            code = (grid[np.where(dead, 0, index[:, k])].astype(np.int64) - 1) // 2
            q = xgs[k][np.arange(m)[:, None], code]
            for j in range(8):
                w = weight[:, 8 * k + j]
                sumqx = (sumqx + (w * q[:, j]) * xb[:, 8 * k + j]).astype(_F)
                sumq2 = (sumq2 + (w * q[:, j]) * q[:, j]).astype(_F)
    resc = ~all_on & ~dead & (sumqx > 0) & (sumq2 > 0)
    with np.errstate(all='ignore'):
        scale = np.where(resc, (sumqx / sumq2).astype(_F), scale)
    scale = np.where(dead, _F(0), scale).astype(_F)
    shifts = np.where(dead, 0, bk).astype(np.int64)
    index = np.where(dead[:, None], 0, index)
    return xb, weight, scale.reshape(nb, 16), shifts.reshape(nb, 16), index.reshape(nb, 16, 2)


def quantize_iq1_m(x):
    """Byte-reproducible mirror of quantize_iq1_m(..., quant_weights=NULL) (56-byte blocks)."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, QK_K)
    out = np.zeros((x.shape[0], BLOCK_BYTES), dtype=np.uint8)
    chunk = 256
    for start in range(0, x.shape[0], chunk):
        out[start:start + chunk] = _quantize_chunk(x[start:start + chunk])
    return out.reshape(-1)


def _quantize_chunk(x):
    nb = x.shape[0]
    grid, kmap, padded = _build()
    xb, weight, scales, shifts, index = _quantize_superblocks(x)
    qs = np.zeros((nb, 32), dtype=np.uint8)
    qh = np.zeros((nb, 16), dtype=np.int64)
    for ib in range(16):
        qs[:, 2 * ib] = (index[:, ib, 0] & 255).astype(np.uint8)
        qs[:, 2 * ib + 1] = (index[:, ib, 1] & 255).astype(np.uint8)
        qh[:, ib] = (index[:, ib, 0] >> 8) | ((index[:, ib, 1] >> 8) << 4)
    max_scale = np.zeros(nb, dtype=np.float32)
    for ib in range(16):
        s = scales[:, ib]
        max_scale = np.where(s > max_scale, s, max_scale)  # ggml MAX
    live = max_scale > 0
    sc = np.zeros((nb, 4), dtype=np.int64)
    f16 = np.zeros(nb, dtype=np.uint16)
    with np.errstate(all='ignore'):
        d = (max_scale / _F(15)).astype(_F)
        id_ = (_F(1) / d).astype(_F)
        sumqx = np.zeros(nb, dtype=np.float32)
        sumq2 = np.zeros(nb, dtype=np.float32)
        xbb = xb.reshape(nb, 16, 16)
        wb = weight.reshape(nb, 16, 16)
        for ib in range(16):
            lv = _nearest_int(_F(0.5) * (id_ * scales[:, ib] - _F(1)))
            lv = np.clip(lv, 0, 7).astype(np.int64)
            sc[:, ib // 4] |= lv << (3 * (ib % 4))
            qh[:, ib] |= MASKS[shifts[:, ib]]
            lf = (2 * lv + 1).astype(np.float32)
            for k in range(2):
                xg = _xx_for(shifts[:, ib], k)
                gidx = qs[:, 2 * ib + k].astype(np.int64) + ((qh[:, ib] << (8 - 4 * k)) & 0x700)
                code = (grid[gidx].astype(np.int64) - 1) // 2
                q = xg[np.arange(nb)[:, None], code] * lf[:, None]
                for j in range(8):
                    w = wb[:, ib, 8 * k + j]
                    sumqx = (sumqx + (w * q[:, j]) * xbb[:, ib, 8 * k + j]).astype(_F)
                    sumq2 = (sumq2 + (w * q[:, j]) * q[:, j]).astype(_F)
        d2 = np.where(sumq2 > 0, (sumqx / sumq2).astype(_F), d)
        f16 = (d2 * _F(1.1125)).astype(np.float16).view(np.uint16).astype(np.int64)
    # superblocks with max_scale == 0 `continue` before masks and scales are applied
    qh_dead = qh.copy()
    qh_dead[:] = 0
    for ib in range(16):
        qh_dead[:, ib] = (index[:, ib, 0] >> 8) | ((index[:, ib, 1] >> 8) << 4)
    qh = np.where(live[:, None], qh, qh_dead)
    sc[:, 0] |= (f16 & 0x000f) << 12
    sc[:, 1] |= (f16 & 0x00f0) << 8
    sc[:, 2] |= (f16 & 0x0f00) << 4
    sc[:, 3] |= (f16 & 0xf000)
    sc = np.where(live[:, None], sc, 0).astype('<u2')
    out = np.zeros((nb, BLOCK_BYTES), dtype=np.uint8)
    out[:, 0:32] = qs
    out[:, 32:48] = qh.astype(np.uint8)
    out[:, 48:56] = sc.view(np.uint8).reshape(nb, 8)
    return out
