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


def _quantize_32(xb, grid, kmap, neighbours):
    """One 32-element sub-block -> (scale, L[32], block_signs[4])."""
    weight = (xb * xb).astype(np.float32)
    waux = np.sqrt(weight).astype(np.float32)
    xval = np.abs(xb).copy()
    block_signs = np.zeros(4, dtype=np.uint8)
    for k in range(4):
        segment = xb[8 * k:8 * k + 8]
        negative = segment < 0
        s = 0
        for i in range(8):
            if negative[i]:
                s |= 1 << i
        if int(negative.sum() % 2):
            local = weight[8 * k:8 * k + 8] * xb[8 * k:8 * k + 8] ** 2
            imin = int(np.argmin(local))  # C keeps the first strict minimum
            xval[8 * k + imin] = -xval[8 * k + imin]
            s ^= 1 << imin
        block_signs[k] = s & 127
    maxv = np.float32(xval.max())
    L = np.zeros(32, dtype=np.int64)
    if maxv < np.float32(GROUP_MAX_EPS_IQ3_XXS):
        return np.float32(0.0), L, block_signs
    best = np.float32(0.0)
    scale = np.float32(maxv / np.float32(2 * K_MAX_Q - 1))
    on_grid = np.ones(8, dtype=bool)
    for is_ in range(-15, 16):
        id_ = (np.float32(2 * K_MAX_Q - 1) + np.float32(is_) * np.float32(0.2)) / maxv
        this_scale = np.float32(1.0) / id_
        Laux = np.empty(32, dtype=np.int64)
        on_aux = np.ones(8, dtype=bool)
        for k in range(8):
            vals = np.float32(0.5) * (id_ * xval[4 * k:4 * k + 4] - np.float32(1.0))
            lq = np.clip(_rint(vals), 0, K_MAX_Q - 1).astype(np.int64)
            Laux[4 * k:4 * k + 4] = lq
            u = int(lq[0] | (lq[1] << 3) | (lq[2] << 6) | (lq[3] << 9))
            grid_index = int(kmap[u])
            if grid_index < 0:
                on_aux[k] = False
                chosen = neighbours[u]
                _, lq = _find_best(chosen, grid, xval[4 * k:4 * k + 4], waux[4 * k:4 * k + 4], this_scale)
                Laux[4 * k:4 * k + 4] = lq
        q = (2 * Laux + 1).astype(np.float32)
        sumqx = _f32_fold(weight * xval * q)
        sumq2 = _f32_fold(weight * q * q)
        if sumq2 > 0 and sumqx * sumqx > best * sumq2:
            scale = np.float32(sumqx / sumq2)
            best = np.float32(scale * sumqx)
            L = Laux.copy()
            on_grid = on_aux.copy()
    if not on_grid.all() and scale > 0:
        id_ = np.float32(1.0) / scale
        for k in range(8):
            if on_grid[k]:
                continue
            vals = np.float32(0.5) * (id_ * xval[4 * k:4 * k + 4] - np.float32(1.0))
            lq = np.clip(_rint(vals), 0, K_MAX_Q - 1).astype(np.int64)
            u = int(lq[0] | (lq[1] << 3) | (lq[2] << 6) | (lq[3] << 9))
            grid_index = int(kmap[u])
            if grid_index < 0:
                chosen = neighbours[u]
                grid_index, lq = _find_best(chosen, grid, xval[4 * k:4 * k + 4],
                                            waux[4 * k:4 * k + 4], scale)
            else:
                lq = (grid[grid_index].astype(np.int64) - 1) // 2
            L[4 * k:4 * k + 4] = lq
        q = (2 * L + 1).astype(np.float32)
        sumqx = _f32_fold(weight * xval * q)
        sumq2 = _f32_fold(weight * q * q)
        if sumq2 > 0:
            scale = np.float32(sumqx / sumq2)
    if scale < 0:
        scale = -scale
        for k in range(4):
            block_signs[k] = (~block_signs[k]) & 127
    return scale, L, block_signs


def quantize_iq3_xxs(x):
    """Byte-reproducible mirror of quantize_row_iq3_xxs_ref (98-byte blocks)."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, QK_K)
    nb = x.shape[0]
    grid, kmap, neighbours = _build(256)
    out = np.empty((nb, 98), dtype=np.uint8)
    for row in range(nb):
        body = np.zeros(96, dtype=np.uint8)  # q3: 64 grid indices + 32 scale/sign bytes
        scales_and_signs = np.zeros(8, dtype=np.uint32)
        scales = np.zeros(8, dtype=np.float32)
        max_scale = np.float32(0.0)
        for ib in range(8):
            scale, L, block_signs = _quantize_32(x[row, 32 * ib:32 * ib + 32], grid, kmap, neighbours)
            indices = np.empty(8, dtype=np.uint8)
            for k in range(8):
                u = int(L[4 * k] | (L[4 * k + 1] << 3) | (L[4 * k + 2] << 6) | (L[4 * k + 3] << 9))
                grid_index = int(kmap[u])
                if grid_index < 0:
                    raise ValueError(f'IQ3_XXS: point {u} not on the grid after selection')
                indices[k] = grid_index
            body[8 * ib:8 * ib + 8] = indices
            scales_and_signs[ib] = (int(block_signs[0]) | (int(block_signs[1]) << 7) |
                                    (int(block_signs[2]) << 14) | (int(block_signs[3]) << 21))
            scales[ib] = scale
            max_scale = max(max_scale, scale)
        if max_scale:
            d = np.float32(max_scale / np.float32(31.0))
            dh = np.float16(d * np.float32(1.0125))
            id_ = np.float32(1.0) / d
            for ib in range(8):
                lv = int(np.clip(_rint(np.float32(0.5) * (id_ * scales[ib] - np.float32(1.0))), 0, 15))
                scales_and_signs[ib] |= np.uint32(lv << 28)
            body[64:96] = np.ascontiguousarray(scales_and_signs).view(np.uint8)
            out[row, 0:2] = np.ascontiguousarray(dh).view(np.uint16).view(np.uint8)
            out[row, 2:98] = body
        else:
            out[row, :] = 0
    return out.reshape(-1)
