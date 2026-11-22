# SPDX-License-Identifier: AGPL-3.0-or-later
"""IQ2_S encoder: iq2 data construction (1024 grid, nwant=1) and the reference quantizer.

Ported from ggml-quants.c at the revision pinned in jdewitt/iq_tables.py
(iq2xs_init_impl for GGML_TYPE_IQ2_S, iq2_find_best_neighbour and
quantize_row_iq2_s_impl). The tune-level skips in the C (`same candidate as the
previous scale`) are numerically neutral, so they are not reproduced; every
acceptance test and every f32 accumulation order is.

The map/neighbour tables are built lazily: iq2xs_init_impl is ~44k tuples x
1,024 grid points, seconds in numpy and never paid unless an IQ2_S tensor is
actually written.
"""
from __future__ import annotations

import numpy as np

from .iq_tables import KGRID_2BIT_1024

QK_K = 256
K_MAX_Q = 3
GROUP_MAX_EPS_IQ2_S = 1e-8
KMAP_SIZE = 43692  # the upstream map allocation; tuples beyond it are upstream UB
_rint = np.rint

_MAP = {}


def _build():
    if 'map' in _MAP:
        return _MAP['map']
    kgrid = KGRID_2BIT_1024.astype(np.int64)
    grid = np.empty((1024, 8), dtype=np.int8)
    for k in range(1024):
        for i in range(8):
            grid[k, i] = 2 * ((int(kgrid[k]) >> (2 * i)) & 3) + 1
    kmap = np.full(KMAP_SIZE, -1, dtype=np.int64)
    for i in range(1024):
        index = 0
        for k in range(8):
            index |= (((int(grid[i, k]) - 1) // 2) << (2 * k))
        kmap[index] = i
    neighbours = {}
    grid64 = grid.astype(np.int64)
    indices = np.arange(1024)
    for i in range(KMAP_SIZE):
        if kmap[i] >= 0:
            continue
        pos = np.array([2 * ((i >> (2 * k)) & 3) + 1 for k in range(8)], dtype=np.int64)
        dist = ((grid64 - pos) ** 2).sum(axis=1)
        order = np.lexsort((indices, dist))
        chosen, current = [], int(dist[order[0]])
        for index in order:
            if int(dist[index]) > current:
                break  # nwant == 1: only the closest distance class
            chosen.append(int(index))
        neighbours[i] = np.asarray(chosen, dtype=np.int64)
    _MAP['map'] = (grid, kmap, neighbours)
    return _MAP['map']


_PADDED = {}


def _padded(neighbours):
    if 'p' in _PADDED:
        return _PADDED['p']
    width = max(len(v) for v in neighbours.values())
    padded = np.full((KMAP_SIZE, width), -1, dtype=np.int64)
    for i, chosen in neighbours.items():
        padded[i, :len(chosen)] = chosen
    _PADDED['p'] = padded
    return padded


def _pack8(values):
    v = values.astype(np.int64)
    out = np.zeros(v.shape[:-1], dtype=np.int64)
    for i in range(8):
        out |= (v[..., i] << (2 * i))
    return out


def _pick(padded, grid, tuples, xval_t, waux_t, scale, kmap):
    gi = kmap[tuples]
    on_grid = gi >= 0
    codes = np.empty(tuples.shape + (8,), dtype=np.int64)
    if on_grid.any():
        codes[on_grid] = (grid[gi[on_grid]].astype(np.int64) - 1) // 2
    miss = ~on_grid
    if miss.any():
        idxs = padded[tuples[miss]]
        valid = idxs >= 0
        safe = np.where(valid, idxs, 0)
        pg = grid[safe].astype(np.float32)
        sc = np.broadcast_to(scale[:, None], miss.shape)[miss]
        diff = sc[:, None, None] * pg - xval_t[miss][:, None, :]
        term = (waux_t[miss][:, None, :] * diff * diff).astype(np.float32)
        d2 = np.zeros(term.shape[:-1], dtype=np.float32)
        for i in range(8):  # sequential f32 accumulation, as the C loop (ndarray.sum is unrolled)
            d2 = (d2 + term[..., i]).astype(np.float32)
        d2 = np.where(valid, d2, np.float32(np.inf))
        best = np.argmin(d2, axis=1)
        chosen = np.take_along_axis(idxs, best[:, None], axis=1)[:, 0]
        gi = gi.copy()
        gi[miss] = chosen
        codes[miss] = (grid[chosen].astype(np.int64) - 1) // 2
    return gi, codes, on_grid


def _quantize_batch(xb, sigma2, grid, kmap, padded):
    """xb: (M,16) float32 blocks of one 256-block each; sigma2: (M,) super-block value."""
    m = xb.shape[0]
    weight = (np.float32(0.25) * sigma2[:, None] + xb * xb).astype(np.float32)
    waux = np.sqrt(weight).astype(np.float32)
    xv = np.abs(xb).copy()
    signs = np.zeros((m, 2), dtype=np.uint8)
    for k in range(2):
        neg = xb[:, 8 * k:8 * k + 8] < 0
        s = np.zeros(m, dtype=np.uint8)
        for i in range(8):
            s = s | (neg[:, i].astype(np.uint8) << i)
        signs[:, k] = s
    maxv = xv.max(axis=1)
    zero = maxv < np.float32(GROUP_MAX_EPS_IQ2_S)
    safe_max = np.where(zero, np.float32(1.0), maxv)
    scale = np.where(zero, np.float32(0.0), maxv / np.float32(2 * K_MAX_Q - 1)).astype(np.float32)
    best = np.zeros(m, dtype=np.float32)
    L = np.zeros((m, 16), dtype=np.int64)
    on_grid = np.ones((m, 2), dtype=bool)
    xv_t = xv.reshape(m, 2, 8)
    waux_t = waux.reshape(m, 2, 8)
    wx = (weight * xv).astype(np.float32)
    for is_ in range(-9, 10):
        id_ = (np.float32(2 * K_MAX_Q - 1) + np.float32(is_) * np.float32(0.1)) / safe_max
        with np.errstate(divide='ignore', invalid='ignore'):
            this_scale = (np.float32(1.0) / id_).astype(np.float32)
        lq = np.clip(_rint(np.float32(0.5) * (id_[:, None] * xv - np.float32(1.0))), 0, K_MAX_Q - 1)
        tuples = _pack8(lq.reshape(m, 2, 8))
        if tuples.max() >= KMAP_SIZE:
            raise ValueError('IQ2_S: tuple index beyond the upstream map (undefined upstream)')
        _, codes, on_aux = _pick(padded, grid, tuples, xv_t, waux_t, this_scale, kmap)
        Laux = codes.reshape(m, 16)
        q = (2 * Laux + 1).astype(np.float32)
        sumqx = np.zeros(m, dtype=np.float32)
        sumq2 = np.zeros(m, dtype=np.float32)
        for j in range(16):
            sumqx = (sumqx + wx[:, j] * q[:, j]).astype(np.float32)
            sumq2 = (sumq2 + weight[:, j] * q[:, j] * q[:, j]).astype(np.float32)
        update = (sumq2 > 0) & (sumqx * sumqx > best * sumq2) & ~zero
        if update.any():
            ns = (sumqx / np.where(sumq2 > 0, sumq2, np.float32(1.0))).astype(np.float32)
            scale = np.where(update, ns, scale).astype(np.float32)
            best = np.where(update, (ns * sumqx).astype(np.float32), best).astype(np.float32)
            L = np.where(update[:, None], Laux, L)
            on_grid = np.where(update[:, None], on_aux, on_grid)
    need = (~on_grid.all(axis=1)) & (scale > 0)
    if need.any():
        id_ = (np.float32(1.0) / np.where(need, scale, np.float32(1.0))).astype(np.float32)
        lq = np.clip(_rint(np.float32(0.5) * (id_[:, None] * xv - np.float32(1.0))), 0, K_MAX_Q - 1)
        tuples = _pack8(lq.reshape(m, 2, 8))
        if tuples.max() >= KMAP_SIZE:
            raise ValueError('IQ2_S: tuple index beyond the upstream map (undefined upstream)')
        _, codes, _ = _pick(padded, grid, tuples, xv_t, waux_t, scale, kmap)
        Lnew = codes.reshape(m, 16)
        take = np.repeat(need[:, None] & ~on_grid, 8, axis=1)
        L = np.where(take, Lnew, L)
        q = (2 * L + 1).astype(np.float32)
        sumqx = np.zeros(m, dtype=np.float32)
        sumq2 = np.zeros(m, dtype=np.float32)
        for j in range(16):
            sumqx = (sumqx + wx[:, j] * q[:, j]).astype(np.float32)
            sumq2 = (sumq2 + weight[:, j] * q[:, j] * q[:, j]).astype(np.float32)
        positive = need & (sumq2 > 0)
        if positive.any():
            scale = np.where(positive,
                             (sumqx / np.where(sumq2 > 0, sumq2, np.float32(1.0))).astype(np.float32),
                             scale)
    neg_scale = scale < 0
    if neg_scale.any():
        scale = np.where(neg_scale, -scale, scale).astype(np.float32)
        signs = np.where(neg_scale[:, None], ~signs, signs).astype(np.uint8)
    return scale.astype(np.float32), L, signs


def quantize_iq2_s(x):
    """Byte-reproducible mirror of quantize_row_iq2_s_ref (82-byte blocks)."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, QK_K)
    nb = x.shape[0]
    grid, kmap, neighbours = _build()
    padded = _padded(neighbours)
    out = np.zeros((nb, 82), dtype=np.uint8)
    chunk = 1024
    for start in range(0, nb, chunk):
        stop = min(start + chunk, nb)
        block = x[start:stop]
        sq = (block * block).astype(np.float32)
        sumx2 = np.zeros(stop - start, dtype=np.float32)
        for i in range(QK_K):  # sequential f32 accumulation, as the C loop (np.sum is pairwise)
            sumx2 = (sumx2 + sq[:, i]).astype(np.float32)
        sigma2 = (np.float32(2) * sumx2 / np.float32(QK_K)).astype(np.float32)
        xb = block.reshape(-1, 16)
        sig = np.repeat(sigma2, 16)
        scales, L, signs = _quantize_batch(xb, sig, grid, kmap, padded)
        count = stop - start
        scales = scales.reshape(count, 16)
        L = L.reshape(count, 16, 16)
        signs = signs.reshape(count, 16, 2)
        tuples = _pack8(L.reshape(count * 16, 2, 8)).reshape(count, 16, 2)
        indices = kmap[tuples]
        if (indices < 0).any():
            raise ValueError('IQ2_S: point not on the grid after selection')
        qs = np.zeros((count, 64), dtype=np.uint8)
        qh = np.zeros((count, 8), dtype=np.uint8)
        for ib in range(16):
            for k in range(2):
                i8 = 2 * ib + k
                gi = indices[:, ib, k]
                qs[:, i8] = (gi & 255).astype(np.uint8)
                qh[:, i8 // 4] |= ((gi >> 8).astype(np.uint8) << (2 * (i8 % 4)))
                qs[:, 32 + i8] = signs[:, ib, k]
        max_scale = scales.max(axis=1)
        live = max_scale > 0
        if live.any():
            d = (max_scale / np.float32(31.0)).astype(np.float32)
            dh = (d * np.float32(0.9875)).astype(np.float16)
            id_ = (np.float32(1.0) / np.where(live, d, np.float32(1.0))).astype(np.float32)
            lv = np.clip(_rint(np.float32(0.5) * (id_[:, None] * scales - np.float32(1.0))),
                         0, 15).astype(np.uint8)
            scale_bytes = np.zeros((count, 8), dtype=np.uint8)
            for pair in range(8):
                scale_bytes[:, pair] = lv[:, 2 * pair] | (lv[:, 2 * pair + 1] << 4)
            out[start:stop, 0:2] = np.ascontiguousarray(dh).view(np.uint16).view(np.uint8).reshape(count, 2)
            out[start:stop, 2:66] = qs
            out[start:stop, 66:74] = qh
            out[start:stop, 74:82] = scale_bytes
            if not live.all():
                out[start:stop][~live] = 0
    return out.reshape(-1)
