# SPDX-License-Identifier: AGPL-3.0-or-later
"""IQ1_S encoder: iq1 data construction (2048 grid, nwant=3) and the reference quantizer.

Ported from ggml-quants.c at the revision pinned in jdewitt/iq_tables.py
(iq2xs_init_impl for GGML_TYPE_IQ1_S, iq1_find_best_neighbour2 and
quantize_row_iq1_s_impl), verified byte-for-byte against the compiled C harness
scripts/ggml_ref/ref_iq1s.c.

WEIGHTS. Upstream asserts ``quant_weights`` for IQ1_S (the real path uses an
importance matrix). This port takes an optional ``weights`` argument (float32,
same element count as ``x``); when omitted it is the deterministic NEUTRAL
vector of all 1.0f, which is also what ref_iq1s.c passes. Byte-exactness is
therefore proven for neutral weights only; arbitrary imatrix weights use the same
f32 arithmetic but are not part of the falsification battery.

The encoder grid (kgrid_1bit_2048) is derived from the vendored decoder table
IQ1S_GRID (int8 bytes -1/0/1 -> 2-bit codes 0/1/2); the derivation is checked
against the C source table by the verification battery.

Every f32 accumulation order is kept: the super-block sum of squares, the sorted
prefix sums, the exhaustive (i1, i2, shift) scan with its running best_score
comparison, the neighbour distance sums and the scale refit. The qsort of
(value, index) pairs is emulated with a stable sort (equal values carry equal
weights, so only which tied element lands in which group could differ).
"""
from __future__ import annotations

import numpy as np

from .iq_tables import IQ1S_GRID

QK_K = 256
BLOCK = 32
NGRID = 2048
KMAP_SIZE = 43692
GROUP_MAX_EPS_IQ1_S = np.float32(1e-12)
IQ1S_DELTA = np.float32(0.125)
BLOCK_BYTES = 50
NWANT = 3

_F = np.float32
_X_P = np.array([-1 + IQ1S_DELTA, IQ1S_DELTA, 1 + IQ1S_DELTA], dtype=np.float32)
_X_M = np.array([-1 - IQ1S_DELTA, -IQ1S_DELTA, 1 - IQ1S_DELTA], dtype=np.float32)

_DATA = {}


def _build():
    """Vectorised iq2xs_init_impl(GGML_TYPE_IQ1_S): grid levels, kmap, padded neighbours."""
    if 'd' in _DATA:
        return _DATA['d']
    raw = np.ascontiguousarray(IQ1S_GRID, dtype=np.uint64).view(np.uint8).reshape(NGRID, 8)
    levels = (raw.view(np.int8).astype(np.int64) + 1)  # codes 0/1/2 (kgrid 2-bit fields)
    assert levels.min() >= 0 and levels.max() <= 2
    pos = 2 * levels + 1  # int8 grid values 1/3/5
    kmap = np.full(KMAP_SIZE, -1, dtype=np.int64)
    pw = 1 << (2 * np.arange(8))
    tuples = (levels * pw).sum(axis=1)
    kmap[tuples] = np.arange(NGRID)
    # every 8-digit {0,1,2} tuple that is not a grid point (the only ones queried)
    digits = np.indices((3,) * 8).reshape(8, -1).T[:, ::-1]  # digit k = element k
    all_t = (digits * pw).sum(axis=1)
    miss_t = np.sort(all_t[kmap[all_t] < 0])
    row_of = np.full(KMAP_SIZE, -1, dtype=np.int64)
    row_of[miss_t] = np.arange(len(miss_t))
    pos_t = np.array([[2 * ((t >> (2 * k)) & 3) + 1 for k in range(8)] for t in miss_t], dtype=np.int64)
    lists = []
    idx = np.arange(NGRID)
    for r in range(len(miss_t)):
        dist = ((pos - pos_t[r]) ** 2).sum(axis=1)
        order = np.lexsort((idx, dist))  # (d2, j) like iq2_compare_func
        ds = dist[order]
        n, d2, nhave = 0, int(ds[0]), 1
        for v in ds:
            if v > d2:
                if nhave == NWANT:
                    break
                d2 = int(v)
                nhave += 1
            n += 1
        lists.append(order[:n])
    width = max(len(v) for v in lists)
    padded = np.full((len(lists), width), -1, dtype=np.int64)
    for r, v in enumerate(lists):
        padded[r, :len(v)] = v
    _DATA['d'] = (levels, kmap, row_of, padded)
    return _DATA['d']


def _scan(sumqx_c, sumq2_c, max_v):
    """Sequential acceptance scan over the (i1, i2, shift) candidates, vectorised over blocks."""
    m, nc = sumqx_c.shape
    best = np.full(m, -np.finfo(np.float32).max, dtype=np.float32)
    scale = max_v.astype(np.float32).copy()
    bidx = np.full(m, -1, dtype=np.int64)
    with np.errstate(all='ignore'):
        for c in range(nc):
            sx = sumqx_c[:, c]
            s2 = sumq2_c[:, c]
            upd = (s2 > 0) & (sx * sx > best * s2)
            if upd.any():
                ns = (sx / s2).astype(np.float32)
                scale = np.where(upd, ns, scale)
                best = np.where(upd, (ns * sx).astype(np.float32), best)
                bidx = np.where(upd, c, bidx)
    return scale, bidx


_CAND = {}


def _cands():
    if 'c' not in _CAND:
        i1s, i2s = [], []
        for i1 in range(BLOCK + 1):
            for i2 in range(i1, BLOCK + 1):
                i1s.append(i1)
                i2s.append(i2)
        _CAND['c'] = (np.array(i1s), np.array(i2s))
    return _CAND['c']


def _quantize_blocks(xb, weight, levels, kmap, row_of, padded):
    """xb, weight: (M,32) float32. Returns scale (M,), shift (M,), index (M,4), live (M,)."""
    m = xb.shape[0]
    maxv = np.abs(xb).max(axis=1)
    live = ~(maxv < GROUP_MAX_EPS_IQ1_S)
    order = np.argsort(xb, axis=1, kind='stable')
    xs = np.take_along_axis(xb, order, axis=1)  # noqa: F841 (documenting the pair sort)
    wx = (weight * xb).astype(np.float32)
    sumx = np.zeros((m, BLOCK + 1), dtype=np.float32)
    sumw = np.zeros((m, BLOCK + 1), dtype=np.float32)
    wo = np.take_along_axis(weight, order, axis=1)
    wxo = np.take_along_axis(wx, order, axis=1)
    for j in range(BLOCK):
        sumx[:, j + 1] = sumx[:, j] + wxo[:, j]
        sumw[:, j + 1] = sumw[:, j] + wo[:, j]
    i1s, i2s = _cands()
    s0, s1, s2 = sumx[:, i1s], sumx[:, i2s], sumx[:, BLOCK:BLOCK + 1]
    w0, w1, w2 = sumw[:, i1s], sumw[:, i2s], sumw[:, BLOCK:BLOCK + 1]
    nc = len(i1s)
    sumqx = np.empty((m, 2 * nc), dtype=np.float32)
    sumq2 = np.empty((m, 2 * nc), dtype=np.float32)
    for sel, xx in ((0, _X_P), (1, _X_M)):
        qx = ((s0 - sumx[:, 0:1]) * xx[0] + (s1 - s0) * xx[1]) + (s2 - s1) * xx[2]
        q2 = (((w0 - sumw[:, 0:1]) * xx[0] * xx[0] + (w1 - w0) * xx[1] * xx[1])
              + (w2 - w1) * xx[2] * xx[2])
        sumqx[:, sel::2] = qx
        sumq2[:, sel::2] = q2
    scale, bidx = _scan(sumqx, sumq2, maxv)
    found = (bidx >= 0) & live
    c = np.where(found, bidx // 2, 0)
    shift = np.where(bidx % 2 == 0, 1, -1)
    besti1, besti2 = i1s[c], i2s[c]
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(BLOCK)[None, :].repeat(m, axis=0), axis=1)
    L = np.where(rank < besti1[:, None], 0, np.where(rank < besti2[:, None], 1, 2))
    neg = scale < 0
    L = np.where(neg[:, None], 2 - L, L)
    scale = np.where(neg, -scale, scale).astype(np.float32)
    shift = np.where(neg, -shift, shift)
    # degenerate blocks: scale 0, shift 1, L = 1 (qs/qh stay zero upstream)
    scale = np.where(found, scale, _F(0)).astype(np.float32)
    shift = np.where(found, shift, 1)
    L = np.where(found[:, None], L, 1)
    xx = np.where((shift == 1)[:, None], _X_P[None, :], _X_M[None, :])  # (M,3)
    pw = (1 << (2 * np.arange(8)))
    L4 = L.reshape(m, 4, 8)
    tuples = (L4 * pw).sum(axis=2)
    index = kmap[tuples]
    off = (index < 0) & found[:, None]
    if off.any():
        bi, ki = np.nonzero(off)
        rows = padded[row_of[tuples[bi, ki]]]
        valid = rows >= 0
        safe = np.where(valid, rows, 0)
        lv = levels[safe]  # (G,W,8)
        xxg = xx[bi]  # (G,3)
        q = np.take_along_axis(xxg[:, None, :].repeat(lv.shape[1], axis=1), lv, axis=2)
        xv = xb.reshape(m, 4, 8)[bi, ki]
        wv = weight.reshape(m, 4, 8)[bi, ki]
        sc = scale[bi]
        d2 = np.zeros(valid.shape, dtype=np.float32)
        for i in range(8):
            diff = (sc[:, None] * q[:, :, i] - xv[:, None, i]).astype(np.float32)
            d2 = (d2 + wv[:, None, i] * diff * diff).astype(np.float32)
        d2 = np.where(valid, d2, np.float32(np.inf))
        pick = np.argmin(d2, axis=1)
        assert np.isfinite(d2[np.arange(len(pick)), pick]).all(), 'IQ1_S: no finite neighbour'
        chosen = rows[np.arange(len(pick)), pick]
        index[bi, ki] = chosen
        # L updated in place upstream; only index is consumed afterwards
    all_on_grid = ~off.any(axis=1)
    refit = ~all_on_grid
    if refit.any():
        rr = np.nonzero(refit)[0]
        lvl = levels[index[rr]]  # (R,4,8)
        qv = np.take_along_axis(xx[rr][:, None, None, :].repeat(4, axis=1).repeat(8, axis=2), lvl[..., None], axis=3)[..., 0]
        w = weight[rr].reshape(-1, 4, 8)
        x = xb[rr].reshape(-1, 4, 8)
        sqx = np.zeros(len(rr), dtype=np.float32)
        sq2 = np.zeros(len(rr), dtype=np.float32)
        for k in range(4):
            for j in range(8):
                wq = (w[:, k, j] * qv[:, k, j]).astype(np.float32)
                sqx = (sqx + wq * x[:, k, j]).astype(np.float32)
                sq2 = (sq2 + wq * qv[:, k, j]).astype(np.float32)
        ok = (sqx > 0) & (sq2 > 0)
        with np.errstate(all='ignore'):
            ns = (sqx / sq2).astype(np.float32)
        scale[rr] = np.where(ok, ns, scale[rr])
    return scale, shift, index, found


def quantize_iq1_s(x, weights=None):
    """Byte-reproducible mirror of quantize_iq1_s (50-byte blocks), neutral weights by default."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, QK_K)
    nb = x.shape[0]
    if weights is None:
        w_all = np.ones_like(x)
    else:
        w_all = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1, QK_K)
        if w_all.shape != x.shape:
            raise ValueError('weights must have the same element count as x')
    levels, kmap, row_of, padded = _build()
    out = np.zeros((nb, BLOCK_BYTES), dtype=np.uint8)
    chunk = 512
    for start in range(0, nb, chunk):
        stop = min(start + chunk, nb)
        blk = x[start:stop]
        cnt = stop - start
        sumx2 = np.zeros(cnt, dtype=np.float32)
        for i in range(QK_K):
            sumx2 = (sumx2 + blk[:, i] * blk[:, i]).astype(np.float32)
        sigma2 = (np.float32(2) * sumx2 / np.float32(QK_K)).astype(np.float32)
        xb = blk.reshape(cnt * 8, BLOCK)
        qw = w_all[start:stop].reshape(cnt * 8, BLOCK)
        sig = np.repeat(sigma2, 8)
        with np.errstate(all='ignore'):
            weight = (qw * np.sqrt(sig[:, None] + xb * xb).astype(np.float32)).astype(np.float32)
        scale, shift, index, found = _quantize_blocks(xb, weight, levels, kmap, row_of, padded)
        scales = scale.reshape(cnt, 8)
        shifts = shift.reshape(cnt, 8)
        index = np.where(found[:, None], index, 0).reshape(cnt, 8, 4)
        qs = (index & 255).astype(np.uint8).reshape(cnt, 32)
        qh = np.zeros((cnt, 8), dtype=np.int64)
        for k in range(4):
            qh |= (index[:, :, k] >> 8) << (3 * k)
        max_scale = scales.max(axis=1)
        live = max_scale > 0
        d = np.zeros(cnt, dtype=np.float32)
        with np.errstate(all='ignore'):
            d = (max_scale / np.float32(15)).astype(np.float32)
            dh = (d * np.float32(1.125)).astype(np.float16)
            idv = (np.float32(1) / d).astype(np.float32)
            lq = np.rint(np.float32(0.5) * (idv[:, None] * scales - np.float32(1)))
        lq = np.where(live[:, None], lq, 0)
        lq = np.clip(lq, 0, 7).astype(np.int64)
        lq = np.where(shifts == -1, lq | 8, lq)
        qh = np.where(live[:, None], qh | (lq << 12), qh)
        dh = np.where(live, dh, np.float16(0)).astype(np.float16)
        out[start:stop, 0:2] = np.ascontiguousarray(dh).view(np.uint8).reshape(cnt, 2)
        out[start:stop, 2:34] = qs
        out[start:stop, 34:50] = qh.astype('<u2').view(np.uint8).reshape(cnt, 16)
    return out.reshape(-1)
