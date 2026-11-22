# SPDX-License-Identifier: AGPL-3.0-or-later
"""IQ3_XXS encoder: iq3xs map/neighbour construction and the reference quantizer.

The C reference derives two runtime tables from the static D4-lattice grid:
`kmap` maps a 4-value tuple to its grid index (or to a neighbour-list offset)
and `kneighbours` holds, for every tuple off the lattice, the entries of the
closest `nwant` distance classes. Both are functions of the pinned grid, so
they are rebuilt here at import instead of being vendored as opaque blobs.

Ported from ggml-quants.c at the revision pinned in jdewitt/iq_tables.py
(iq3xs_init_impl, iq3_find_best_neighbour, quantize_row_iq3_xxs_impl). The
quantizer's per-block sums are in C element order; a pairwise numpy sum flips
tie decisions, so they use an explicit f32 fold.

NOT WIRED into CODECS, deliberately: this port is byte-exact against the
compiled reference (scripts/ggml_ref/ref_iq3.c; 8 distributions x 64 blocks
and real F32 tensors) but it walks 32 elements at a time in Python, and on a
4M-element tensor that is minutes per tensor, not seconds. An editable IQ3_XXS
needs the is-loop and the neighbour search vectorised across blocks first.
Until then IQ3_XXS stays decode-only and edits re-encode the touched tensors
at a writable type, which is the documented behaviour for the whole IQ1/IQ2/IQ3
family. Keep this module as the falsifier target for the vectorised version.
"""
from __future__ import annotations

import numpy as np

from .iq_tables import KGRID_256, KGRID_512

# Local copies (no import of quants: quants registers this encoder at its end,
# so importing back would be a circular import).
QK_K = 256
GROUP_MAX_EPS_IQ3_XXS = 1e-8  # the constant quantize_row_iq3_xxs_impl tests
_rint = np.rint

K_MAX_Q = 8

_MAPS = {}


def _f32_fold(values):
    acc = np.float32(0.0)
    for value in values:
        acc = np.float32(acc + np.float32(value))
    return acc


def _build(grid_size):
    """Grid values, tuple map and neighbour lists exactly as iq3xs_init_impl builds them."""
    if grid_size in _MAPS:
        return _MAPS[grid_size]
    kgrid = (KGRID_256 if grid_size == 256 else KGRID_512).astype(np.int64)
    grid = np.empty((grid_size, 4), dtype=np.int8)
    for k in range(grid_size):
        for i in range(4):
            grid[k, i] = 2 * ((int(kgrid[k]) >> (3 * i)) & 7) + 1
    kmap = np.full(4096, -1, dtype=np.int64)
    for i in range(grid_size):
        index = 0
        for k in range(4):
            index |= (((int(grid[i, k]) - 1) // 2) << (3 * k))
        kmap[index] = i
    nwant = 2 if grid_size == 256 else 3
    neighbours = {}
    cursor = 0
    for i in range(4096):
        if kmap[i] >= 0:
            continue
        pos = np.array([2 * ((i >> (3 * k)) & 7) + 1 for k in range(4)], dtype=np.int64)
        dist = ((grid.astype(np.int64) - pos) ** 2).sum(axis=1)
        order = np.lexsort((np.arange(grid_size), dist))
        chosen, nhave, current = [], 1, int(dist[order[0]])
        for index in order:
            d = int(dist[index])
            if d > current:
                if nhave == nwant:
                    break
                current = d
                nhave += 1
            chosen.append(int(index))
        neighbours[i] = np.asarray(chosen, dtype=np.int64)
        kmap[i] = -(cursor + 1)
        cursor += 1 + len(chosen)
    _MAPS[grid_size] = (grid, kmap, neighbours)
    return _MAPS[grid_size]


def _find_best(neigh, grid, xval, weight, scale):
    idxs = neigh
    pg = grid[idxs].astype(np.float32)
    diff = scale * pg - xval[None, :]
    d2 = (weight[None, :] * diff * diff).sum(axis=1)
    best = int(np.argmin(d2))
    grid_index = int(idxs[best])
    return grid_index, (grid[grid_index].astype(np.int32) - 1) // 2


def _padded_neighbours(neighbours, size=4096):
    """(size, Lmax) neighbour-index matrix padded with -1, plus per-tuple lengths."""
    lengths = np.zeros(size, dtype=np.int64)
    for i, chosen in neighbours.items():
        lengths[i] = len(chosen)
    width = max(1, int(lengths.max()))
    padded = np.full((size, width), -1, dtype=np.int64)
    for i, chosen in neighbours.items():
        padded[i, :len(chosen)] = chosen
    return padded, lengths


def _pack_tuples(values):
    """(M, 32) per-element values -> (M, 8) packed 3-bit tuples, C bit order."""
    v = values.reshape(-1, 8, 4).astype(np.int64)
    return v[..., 0] | (v[..., 1] << 3) | (v[..., 2] << 6) | (v[..., 3] << 9)


def _neighbour_pick(padded, grid, tuples, xval_t, waux_t, scale, kmap):
    """Direct map hit, else the nearest neighbour entry (C's iq3_find_best_neighbour).

    Returns (mapped tuple count M, 8) grid indices, (M, 8, 4) per-element codes,
    and the on-grid mask.
    """
    gi = kmap[tuples]
    on_grid = gi >= 0
    codes = np.empty(tuples.shape + (4,), dtype=np.int64)
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
        d2 = (waux_t[miss][:, None, :] * diff * diff).sum(axis=-1)
        d2 = np.where(valid, d2, np.float32(np.inf))
        best = np.argmin(d2, axis=1)
        chosen = np.take_along_axis(idxs, best[:, None], axis=1)[:, 0]
        gi = gi.copy()
        gi[miss] = chosen
        codes[miss] = (grid[chosen].astype(np.int64) - 1) // 2
    return gi, codes, on_grid


def _quantize_batch(xb, grid, kmap, padded):
    """Vectorised quantize_row_iq3_xxs_impl for M 32-element blocks."""
    m = xb.shape[0]
    weight = (xb * xb).astype(np.float32)
    waux = np.sqrt(weight).astype(np.float32)
    xv = np.abs(xb).copy()
    signs = np.zeros((m, 4), dtype=np.uint8)
    for k in range(4):
        sl = slice(8 * k, 8 * k + 8)
        seg = xb[:, sl]
        neg = seg < 0
        s = np.zeros(m, dtype=np.uint8)
        for i in range(8):
            s = s | (neg[:, i].astype(np.uint8) << i)
        odd = (neg.sum(axis=1) % 2).astype(bool)
        if odd.any():
            local = weight[:, sl] * seg * seg
            imin = np.argmin(np.where(odd[:, None], local, np.float32(np.inf)), axis=1)
            rows = np.arange(m)[odd]
            cols = imin[odd]
            xv[rows, 8 * k + cols] = -xv[rows, 8 * k + cols]
            s[odd] ^= (1 << cols).astype(np.uint8)
        signs[:, k] = s & 127
    maxv = xv.max(axis=1)
    zero = maxv < np.float32(GROUP_MAX_EPS_IQ3_XXS)
    safe_max = np.where(zero, np.float32(1.0), maxv)
    scale = np.where(zero, np.float32(0.0), maxv / np.float32(2 * K_MAX_Q - 1)).astype(np.float32)
    best = np.zeros(m, dtype=np.float32)
    L = np.zeros((m, 32), dtype=np.int64)
    on_grid = np.ones((m, 8), dtype=bool)
    xv_t = xv.reshape(m, 8, 4)
    waux_t = waux.reshape(m, 8, 4)
    wx = (weight * xv).astype(np.float32)
    for is_ in range(-15, 16):
        id_ = (np.float32(2 * K_MAX_Q - 1) + np.float32(is_) * np.float32(0.2)) / safe_max
        with np.errstate(divide='ignore', invalid='ignore'):
            this_scale = (np.float32(1.0) / id_).astype(np.float32)
        lq = np.clip(_rint(np.float32(0.5) * (id_[:, None] * xv - np.float32(1.0))), 0, K_MAX_Q - 1)
        tuples = _pack_tuples(lq)
        _, codes, on_aux = _neighbour_pick(padded, grid, tuples, xv_t, waux_t, this_scale, kmap)
        Laux = codes.reshape(m, 32)
        q = (2 * Laux + 1).astype(np.float32)
        sumqx = np.zeros(m, dtype=np.float32)
        sumq2 = np.zeros(m, dtype=np.float32)
        for j in range(32):
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
        tuples = _pack_tuples(lq)
        _, codes, on_new = _neighbour_pick(padded, grid, tuples, xv_t, waux_t, scale, kmap)
        Lnew = codes.reshape(m, 32)
        take = np.repeat(need[:, None] & ~on_grid, 4, axis=1)
        L = np.where(take, Lnew, L)
        q = (2 * L + 1).astype(np.float32)
        sumqx = np.zeros(m, dtype=np.float32)
        sumq2 = np.zeros(m, dtype=np.float32)
        for j in range(32):
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
        signs = np.where(neg_scale[:, None], (~signs) & 127, signs).astype(np.uint8)
    return scale.astype(np.float32), L, signs


def quantize_iq3_xxs(x):
    """Byte-reproducible mirror of quantize_row_iq3_xxs_ref (98-byte blocks)."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, QK_K)
    nb = x.shape[0]
    grid, kmap, neighbours = _build(256)
    padded, _lengths = _padded_neighbours(neighbours)
    out = np.empty((nb, 98), dtype=np.uint8)
    chunk = 2048  # 256-blocks per batch keeps the padded neighbour gather bounded
    for start in range(0, nb, chunk):
        stop = min(start + chunk, nb)
        xb = x[start:stop].reshape(-1, 32)
        scales, L, signs = _quantize_batch(xb, grid, kmap, padded)
        count = stop - start
        scales = scales.reshape(count, 8)
        L = L.reshape(count, 8, 32)
        signs = signs.reshape(count, 8, 4)
        bodies = np.zeros((count, 96), dtype=np.uint8)
        sas = np.zeros((count, 8), dtype=np.uint32)
        tuples = _pack_tuples(L.reshape(count * 8, 32)).reshape(count, 8, 8)
        indices = kmap[tuples]
        if (indices < 0).any():
            raise ValueError('IQ3_XXS: point not on the grid after selection')
        for ib in range(8):
            bodies[:, 8 * ib:8 * ib + 8] = indices[:, ib].astype(np.uint8)
        sas = (signs[..., 0].astype(np.uint32) |
               (signs[..., 1].astype(np.uint32) << 7) |
               (signs[..., 2].astype(np.uint32) << 14) |
               (signs[..., 3].astype(np.uint32) << 21))
        max_scale = scales.max(axis=1)
        live = max_scale > 0
        if live.any():
            d = (max_scale / np.float32(31.0)).astype(np.float32)
            dh = (d * np.float32(1.0125)).astype(np.float16)
            id_ = (np.float32(1.0) / np.where(live, d, np.float32(1.0))).astype(np.float32)
            lv = np.clip(_rint(np.float32(0.5) * (id_[:, None] * scales - np.float32(1.0))),
                         0, 15).astype(np.uint32)
            sas |= (lv << 28)
            bodies[:, 64:96] = np.ascontiguousarray(sas).view(np.uint8).reshape(count, 32)
            out[start:stop, 0:2] = np.ascontiguousarray(dh).view(np.uint16).view(np.uint8).reshape(count, 2)
            out[start:stop, 2:98] = bodies
            if not live.all():
                out[start:stop][~live] = 0
        else:
            out[start:stop, :] = 0
    return out.reshape(-1)


def _quantize_batch_s(xb, grid, kmap, padded):
    """Vectorised quantize_row_iq3_s_impl for M 32-element blocks (512-entry grid)."""
    m = xb.shape[0]
    weight = (xb * xb).astype(np.float32)
    waux = np.sqrt(weight).astype(np.float32)
    xv = np.abs(xb).copy()
    signs = np.zeros((m, 4), dtype=np.uint8)
    for k in range(4):
        neg = xb[:, 8 * k:8 * k + 8] < 0
        s = np.zeros(m, dtype=np.uint8)
        for i in range(8):
            s = s | (neg[:, i].astype(np.uint8) << i)
        signs[:, k] = s
    maxv = xv.max(axis=1)
    zero = maxv == 0
    safe_max = np.where(zero, np.float32(1.0), maxv)
    scale = np.where(zero, np.float32(0.0), maxv / np.float32(2 * K_MAX_Q - 1)).astype(np.float32)
    best = np.zeros(m, dtype=np.float32)
    L = np.zeros((m, 32), dtype=np.int64)
    on_grid = np.zeros((m, 8), dtype=bool)
    xv_t = xv.reshape(m, 8, 4)
    waux_t = waux.reshape(m, 8, 4)
    wx = (weight * xv).astype(np.float32)
    for is_ in range(-9, 10):
        id_ = (np.float32(2 * K_MAX_Q - 1) + np.float32(is_) * np.float32(0.2)) / safe_max
        with np.errstate(divide='ignore', invalid='ignore'):
            this_scale = (np.float32(1.0) / id_).astype(np.float32)
        lq = np.clip(_rint(np.float32(0.5) * (id_[:, None] * xv - np.float32(1.0))), 0, K_MAX_Q - 1)
        tuples = _pack_tuples(lq)
        _, codes, on_aux = _neighbour_pick(padded, grid, tuples, xv_t, waux_t, this_scale, kmap)
        Laux = codes.reshape(m, 32)
        q = (2 * Laux + 1).astype(np.float32)
        sumqx = np.zeros(m, dtype=np.float32)
        sumq2 = np.zeros(m, dtype=np.float32)
        for j in range(32):
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
        tuples = _pack_tuples(lq)
        _, codes, _ = _neighbour_pick(padded, grid, tuples, xv_t, waux_t, scale, kmap)
        Lnew = codes.reshape(m, 32)
        # The C reference redoes every sub-block here (its is_on_grid skip is commented out).
        L = np.where(need[:, None], Lnew, L)
        q = (2 * L + 1).astype(np.float32)
        sumqx = np.zeros(m, dtype=np.float32)
        sumq2 = np.zeros(m, dtype=np.float32)
        for j in range(32):
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
    return scale.astype(np.float32), L, signs, zero


def quantize_iq3_s(x):
    """Byte-reproducible mirror of quantize_row_iq3_s_ref (110-byte blocks)."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, QK_K)
    nb = x.shape[0]
    grid, kmap, neighbours = _build(512)
    padded, _lengths = _padded_neighbours(neighbours)
    out = np.zeros((nb, 110), dtype=np.uint8)
    chunk = 2048
    for start in range(0, nb, chunk):
        stop = min(start + chunk, nb)
        xb = x[start:stop].reshape(-1, 32)
        scales, L, signs, skip = _quantize_batch_s(xb, grid, kmap, padded)
        count = stop - start
        scales = scales.reshape(count, 8)
        L = L.reshape(count, 8, 32)
        signs = signs.reshape(count, 8, 4)
        skip = skip.reshape(count, 8)
        qs = np.zeros((count, 64), dtype=np.uint8)
        qh = np.zeros((count, 8), dtype=np.uint8)
        sign_bytes = np.zeros((count, 32), dtype=np.uint8)
        tuples = _pack_tuples(L.reshape(count * 8, 32)).reshape(count, 8, 8)
        indices = kmap[tuples]
        if (indices < 0).any():
            raise ValueError('IQ3_S: point not on the grid after selection')
        # Upstream pointer bookkeeping: a zero sub-block `continue`s BEFORE qs/signs
        # advance, so later sub-blocks shift left. Replicate it; qh is indexed by ib
        # and does not shift.
        advance = (~skip).astype(np.int64)
        before = np.cumsum(advance, axis=1) - advance
        rows = np.arange(count)
        for ib in range(8):
            live = ~skip[:, ib]
            if not live.any():
                continue
            base = before[live, ib]
            for k in range(8):
                gi = indices[live, ib, k]
                qs[rows[live], 8 * base + k] = (gi & 255).astype(np.uint8)
                qh[rows[live], (ib * 8 + k) // 8] |= ((gi >> 8).astype(np.uint8) << ((ib * 8 + k) % 8))
            sign_bytes[rows[live][:, None], 4 * base[:, None] + np.arange(4)[None, :]] = signs[live, ib]
        max_scale = scales.max(axis=1)
        live = max_scale > 0
        if live.any():
            d = (max_scale / np.float32(31.0)).astype(np.float32)
            dh = (d * np.float32(1.033)).astype(np.float16)
            id_ = (np.float32(1.0) / np.where(live, d, np.float32(1.0))).astype(np.float32)
            lv = np.clip(_rint(np.float32(0.5) * (id_[:, None] * scales - np.float32(1.0))),
                         0, 15).astype(np.uint8)
            scale_bytes = np.zeros((count, 4), dtype=np.uint8)
            for pair in range(4):
                scale_bytes[:, pair] = lv[:, 2 * pair] | (lv[:, 2 * pair + 1] << 4)
            out[start:stop, 0:2] = np.ascontiguousarray(dh).view(np.uint16).view(np.uint8).reshape(count, 2)
            out[start:stop, 2:66] = qs
            out[start:stop, 66:74] = qh
            out[start:stop, 74:106] = sign_bytes
            out[start:stop, 106:110] = scale_bytes
    return out.reshape(-1)
