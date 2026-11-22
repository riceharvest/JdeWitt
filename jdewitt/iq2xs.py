# SPDX-License-Identifier: AGPL-3.0-or-later
"""IQ2_XS encoder: iq2 data construction (512 grid, nwant=2) and the reference quantizer.

Ported from ggml-quants.c at the revision pinned in jdewitt/iq_tables.py
(iq2xs_init_impl for GGML_TYPE_IQ2_XS, iq2_find_best_neighbour and
quantize_row_iq2_xs_impl). The upstream impl asserts quant_weights; the optional
`quant_weights` argument here defaults to the deterministic neutral vector of all
1.0f, which is what scripts/ggml_ref/ref_iq2xs.c passes. The tune-level skips in
the C (`same candidate as the previous scale`, the neighbour cache) are
numerically neutral, so they are not reproduced; every acceptance test and every
f32 accumulation order is: all sums are element-order folds (never np.sum).

kgrid_2bit_512 is vendored here (extracted from the pinned ggml-quants.c, sha256
1665bffa...) because jdewitt/iq_tables.py does not carry it.
"""
from __future__ import annotations

import numpy as np

QK_K = 256
K_MAX_Q = 3
GROUP_MAX_EPS = 1e-15
KMAP_SIZE = 43692  # the upstream map allocation
BLOCK_BYTES = 74
NWANT = 2
_rint = np.rint

KGRID_2BIT_512 = np.array([
    0, 2, 5, 8, 10, 17, 20, 22, 25, 32, 34, 37,
    40, 65, 68, 70, 73, 80, 82, 85, 88, 97, 100, 128,
    130, 133, 136, 145, 148, 153, 160, 257, 260, 262, 265, 272,
    274, 277, 280, 282, 289, 292, 320, 322, 325, 328, 337, 340,
    352, 360, 385, 388, 400, 512, 514, 517, 520, 529, 532, 544,
    577, 580, 592, 597, 640, 650, 1025, 1028, 1030, 1033, 1040, 1042,
    1045, 1048, 1057, 1060, 1088, 1090, 1093, 1096, 1105, 1108, 1110, 1120,
    1153, 1156, 1168, 1280, 1282, 1285, 1288, 1297, 1300, 1312, 1345, 1348,
    1360, 1377, 1408, 1537, 1540, 1552, 1574, 1600, 1602, 1668, 2048, 2050,
    2053, 2056, 2058, 2065, 2068, 2080, 2085, 2113, 2116, 2128, 2136, 2176,
    2208, 2218, 2305, 2308, 2320, 2368, 2433, 2441, 2560, 2592, 2600, 2710,
    2720, 4097, 4100, 4102, 4105, 4112, 4114, 4117, 4120, 4129, 4132, 4160,
    4162, 4165, 4168, 4177, 4180, 4192, 4202, 4225, 4228, 4240, 4352, 4354,
    4357, 4360, 4369, 4372, 4384, 4417, 4420, 4432, 4480, 4500, 4502, 4609,
    4612, 4614, 4624, 4672, 4704, 5120, 5122, 5125, 5128, 5137, 5140, 5152,
    5185, 5188, 5193, 5200, 5220, 5248, 5377, 5380, 5392, 5440, 5632, 5652,
    5705, 6145, 6148, 6160, 6162, 6208, 6228, 6278, 6400, 6405, 6502, 6737,
    6825, 8192, 8194, 8197, 8200, 8202, 8209, 8212, 8224, 8257, 8260, 8272,
    8320, 8352, 8449, 8452, 8464, 8512, 8520, 8549, 8704, 8738, 8832, 8872,
    9217, 9220, 9232, 9257, 9280, 9472, 9537, 9554, 9625, 9729, 9754, 9894,
    10240, 10248, 10250, 10272, 10325, 10376, 10402, 10600, 10640, 10760, 10784, 10882,
    10888, 10890, 16385, 16388, 16390, 16393, 16400, 16402, 16405, 16408, 16417, 16420,
    16448, 16450, 16453, 16456, 16458, 16465, 16468, 16480, 16485, 16513, 16516, 16528,
    16640, 16642, 16645, 16648, 16657, 16660, 16672, 16705, 16708, 16720, 16768, 16773,
    16802, 16897, 16900, 16912, 16914, 16937, 16960, 17408, 17410, 17413, 17416, 17425,
    17428, 17433, 17440, 17473, 17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700,
    17728, 17818, 17920, 17930, 17988, 18000, 18433, 18436, 18448, 18496, 18501, 18516,
    18530, 18688, 18705, 18756, 18768, 18793, 18948, 20480, 20482, 20485, 20488, 20497,
    20500, 20512, 20520, 20545, 20548, 20560, 20608, 20737, 20740, 20752, 20757, 20800,
    20802, 20992, 21060, 21162, 21505, 21508, 21520, 21537, 21568, 21600, 21633, 21665,
    21760, 21768, 21888, 21896, 22049, 22120, 22177, 22528, 22548, 22593, 22608, 22681,
    22810, 22848, 22850, 23173, 24577, 24580, 24592, 24640, 24660, 24674, 24710, 24745,
    24832, 25124, 25162, 25234, 25600, 25622, 25872, 25920, 25925, 26020, 26625, 26730,
    26917, 27142, 27220, 27234, 32768, 32770, 32773, 32776, 32785, 32788, 32800, 32810,
    32833, 32836, 32848, 32896, 32898, 32936, 32938, 33025, 33028, 33030, 33040, 33088,
    33105, 33113, 33280, 33312, 33408, 33410, 33440, 33448, 33793, 33796, 33808, 33810,
    33813, 33856, 33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853,
    34906, 34944, 34946, 34984, 35078, 35362, 35456, 35464, 35478, 35496, 36865, 36868,
    36880, 36928, 36950, 36996, 37120, 37154, 37220, 37462, 37513, 37888, 37893, 37956,
    37968, 37976, 38185, 38288, 38290, 38465, 38993, 39078, 39241, 39445, 39520, 40960,
    40962, 40968, 40970, 40992, 41002, 41120, 41297, 41305, 41382, 41472, 41474, 41480,
    41514, 41600, 41632, 42048, 42133, 42597, 42648, 43018, 43040, 43042, 43048, 43168,
    43176, 43268, 43396, 43398, 43560, 43562, 43665, 43690,
], dtype=np.uint16)
assert KGRID_2BIT_512.shape == (512,)

_MAP = {}


def _build():
    if 'map' in _MAP:
        return _MAP['map']
    kgrid = KGRID_2BIT_512.astype(np.int64)
    ngrid = len(kgrid)
    grid = np.empty((ngrid, 8), dtype=np.int8)
    for k in range(ngrid):
        for i in range(8):
            grid[k, i] = 2 * ((int(kgrid[k]) >> (2 * i)) & 3) + 1
    kmap = np.full(KMAP_SIZE, -1, dtype=np.int64)
    for i in range(ngrid):
        index = 0
        for k in range(8):
            index |= (((int(grid[i, k]) - 1) // 2) << (2 * k))
        kmap[index] = i
    neighbours = {}
    grid64 = grid.astype(np.int64)
    indices = np.arange(ngrid)
    shifts = 2 * np.arange(8)
    for i in range(KMAP_SIZE):
        if kmap[i] >= 0:
            continue
        pos = 2 * ((i >> shifts) & 3) + 1
        dist = ((grid64 - pos) ** 2).sum(axis=1)
        order = np.lexsort((indices, dist))  # qsort by (dist, index)
        chosen, d2, nhave = [], int(dist[order[0]]), 1
        for index in order:
            if int(dist[index]) > d2:
                if nhave == NWANT:
                    break
                d2 = int(dist[index])
                nhave += 1
            chosen.append(int(index))
        neighbours[i] = np.asarray(chosen, dtype=np.int64)
    width = max(len(v) for v in neighbours.values())
    padded = np.full((KMAP_SIZE, width), -1, dtype=np.int64)
    for i, chosen in neighbours.items():
        padded[i, :len(chosen)] = chosen
    _MAP['map'] = (grid, kmap, padded)
    return _MAP['map']


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
        idxs = padded[tuples[miss]]                      # (T, W)
        valid = idxs >= 0
        safe = np.where(valid, idxs, 0)
        sc = np.broadcast_to(scale[:, None], miss.shape)[miss][:, None]
        xm = xval_t[miss]
        wm = waux_t[miss]
        d2 = np.zeros(idxs.shape, dtype=np.float32)
        for i in range(8):                               # C: d2 += w*diff*diff, element order
            q = grid[safe, i].astype(np.float32)
            diff = (sc * q - xm[:, i:i + 1]).astype(np.float32)
            d2 = (d2 + (wm[:, i:i + 1] * diff) * diff).astype(np.float32)
        d2 = np.where(valid, d2, np.float32(np.inf))
        best = np.argmin(d2, axis=1)                     # first minimum == C strict <
        chosen = np.take_along_axis(idxs, best[:, None], axis=1)[:, 0]
        gi = gi.copy()
        gi[miss] = chosen
        codes[miss] = (grid[chosen].astype(np.int64) - 1) // 2
    return gi, codes, on_grid


def _fold(terms):
    acc = np.zeros(terms[0].shape, dtype=np.float32)
    for t in terms:
        acc = (acc + t).astype(np.float32)
    return acc


def _quantize_batch(xb, qw, sigma2, grid, kmap, padded):
    """xb, qw: (M,16) float32 sub-blocks; sigma2: (M,) per-super-block value."""
    m = xb.shape[0]
    weight = (qw * np.sqrt((sigma2[:, None] + (xb * xb).astype(np.float32)).astype(np.float32))).astype(np.float32)
    waux = np.sqrt(weight).astype(np.float32)
    neg = xb < 0
    xv = np.where(neg, -xb, xb).astype(np.float32)
    xv = xv.copy()
    signs = np.zeros((m, 2), dtype=np.int64)
    for k in range(2):
        sl = slice(8 * k, 8 * k + 8)
        s = np.zeros(m, dtype=np.int64)
        for i in range(8):
            s |= neg[:, 8 * k + i].astype(np.int64) << i
        odd = (neg[:, sl].sum(axis=1) % 2) == 1
        ax = ((weight[:, sl] * xb[:, sl]) * xb[:, sl]).astype(np.float32)
        imin = np.argmin(ax, axis=1)                     # first minimum, strict <
        rows = np.nonzero(odd)[0]
        xv[rows, 8 * k + imin[rows]] = -xv[rows, 8 * k + imin[rows]]
        s[rows] ^= (1 << imin[rows])
        signs[:, k] = s & 127
    maxv = xv.max(axis=1)
    zero = maxv < np.float32(GROUP_MAX_EPS)
    safe_max = np.where(zero, np.float32(1.0), maxv).astype(np.float32)
    scale = np.where(zero, np.float32(0.0), maxv / np.float32(2 * K_MAX_Q - 1)).astype(np.float32)
    best = np.zeros(m, dtype=np.float32)
    L = np.zeros((m, 16), dtype=np.int64)
    on_grid = np.ones((m, 2), dtype=bool)
    xv_t = xv.reshape(m, 2, 8)
    waux_t = waux.reshape(m, 2, 8)
    wx = (weight * xv).astype(np.float32)
    for is_ in range(-9, 10):
        id_ = ((np.float32(2 * K_MAX_Q - 1) + np.float32(is_) * np.float32(0.1)) / safe_max).astype(np.float32)
        this_scale = (np.float32(1.0) / id_).astype(np.float32)
        lq = np.clip(_rint(np.float32(0.5) * ((id_[:, None] * xv).astype(np.float32) - np.float32(1.0))), 0, K_MAX_Q - 1)
        tuples = _pack8(lq.reshape(m, 2, 8))
        _, codes, on_aux = _pick(padded, grid, tuples, xv_t, waux_t, this_scale, kmap)
        Laux = codes.reshape(m, 16)
        q = (2 * Laux + 1).astype(np.float32)
        sumqx = _fold([(wx[:, j] * q[:, j]).astype(np.float32) for j in range(16)])
        sumq2 = _fold([((weight[:, j] * q[:, j]) * q[:, j]).astype(np.float32) for j in range(16)])
        update = (sumq2 > 0) & ((sumqx * sumqx).astype(np.float32) > (best * sumq2).astype(np.float32)) & ~zero
        if update.any():
            ns = (sumqx / np.where(sumq2 > 0, sumq2, np.float32(1.0))).astype(np.float32)
            scale = np.where(update, ns, scale).astype(np.float32)
            best = np.where(update, (ns * sumqx).astype(np.float32), best).astype(np.float32)
            L = np.where(update[:, None], Laux, L)
            on_grid = np.where(update[:, None], on_aux, on_grid)
    need = (~on_grid.all(axis=1)) & (scale > 0) & ~zero
    if need.any():
        id_ = (np.float32(1.0) / np.where(need, scale, np.float32(1.0))).astype(np.float32)
        lq = np.clip(_rint(np.float32(0.5) * ((id_[:, None] * xv).astype(np.float32) - np.float32(1.0))), 0, K_MAX_Q - 1)
        tuples = _pack8(lq.reshape(m, 2, 8))
        _, codes, _ = _pick(padded, grid, tuples, xv_t, waux_t, scale, kmap)
        Lnew = codes.reshape(m, 16)
        take = np.repeat(need[:, None] & ~on_grid, 8, axis=1)
        L = np.where(take, Lnew, L)
        q = (2 * L + 1).astype(np.float32)
        sumqx = _fold([(wx[:, j] * q[:, j]).astype(np.float32) for j in range(16)])
        sumq2 = _fold([((weight[:, j] * q[:, j]) * q[:, j]).astype(np.float32) for j in range(16)])
        positive = need & (sumq2 > 0)
        if positive.any():
            scale = np.where(positive,
                             (sumqx / np.where(sumq2 > 0, sumq2, np.float32(1.0))).astype(np.float32),
                             scale).astype(np.float32)
    neg_scale = scale < 0
    if neg_scale.any():
        scale = np.where(neg_scale, -scale, scale).astype(np.float32)
        signs = np.where(neg_scale[:, None], (~signs) & 127, signs)
    return scale.astype(np.float32), L, signs, zero


def quantize_iq2_xs(x, quant_weights=None):
    """Byte-reproducible mirror of quantize_row_iq2_xs_impl (74-byte blocks).

    quant_weights: float32 vector like x (importance weights); None means the
    neutral all-1.0f vector (what the C harness passes).
    """
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, QK_K)
    nb = x.shape[0]
    if quant_weights is None:
        qw_all = np.ones_like(x)
    else:
        qw_all = np.ascontiguousarray(quant_weights, dtype=np.float32).reshape(-1, QK_K)
        if qw_all.shape != x.shape:
            raise ValueError('IQ2_XS: quant_weights must match x')
    grid, kmap, padded = _build()
    out = np.zeros((nb, BLOCK_BYTES), dtype=np.uint8)
    chunk = 128
    with np.errstate(all='ignore'):
        for start in range(0, nb, chunk):
            stop = min(start + chunk, nb)
            count = stop - start
            block = x[start:stop]
            sumx2 = _fold([(block[:, i] * block[:, i]).astype(np.float32) for i in range(QK_K)])
            sigma2 = (sumx2 / np.float32(QK_K)).astype(np.float32)
            sig = np.repeat(sigma2, 16)
            scales, L, signs, zero = _quantize_batch(
                block.reshape(-1, 16), qw_all[start:stop].reshape(-1, 16), sig, grid, kmap, padded)
            scales = np.where(zero, np.float32(0.0), scales).astype(np.float32).reshape(count, 16)
            L = L.reshape(count, 16, 16)
            signs = signs.reshape(count, 16, 2)
            zero = zero.reshape(count, 16)
            tuples = _pack8(L.reshape(count * 16, 2, 8)).reshape(count, 16, 2)
            indices = kmap[tuples]
            q2 = np.zeros((count, 32), dtype=np.uint16)
            ok = zero[:, :, None] | (indices >= 0)
            if not ok.all():
                raise ValueError('IQ2_XS: point not on the grid after selection')
            for ib in range(16):
                for k in range(2):
                    live = ~zero[:, ib]
                    val = (indices[:, ib, k] | (signs[:, ib, k] << 9)).astype(np.uint16)
                    q2[:, 2 * ib + k] = np.where(live, val, np.uint16(0))
            max_scale = scales.max(axis=1)
            live = max_scale > 0
            if live.any():
                d = (max_scale / np.float32(31.0)).astype(np.float32)
                dh = d.astype(np.float16)
                id_ = (np.float32(1.0) / np.where(live, d, np.float32(1.0))).astype(np.float32)
                lv = np.clip(_rint(np.float32(0.5) * ((id_[:, None] * scales).astype(np.float32) - np.float32(1.0))),
                             0, 15).astype(np.uint8)
                scale_bytes = np.zeros((count, 8), dtype=np.uint8)
                for pair in range(8):
                    scale_bytes[:, pair] = lv[:, 2 * pair] | (lv[:, 2 * pair + 1] << 4)
                blk = np.zeros((count, BLOCK_BYTES), dtype=np.uint8)
                blk[:, 0:2] = np.ascontiguousarray(dh).view(np.uint16).view(np.uint8).reshape(count, 2)
                blk[:, 2:66] = q2.astype('<u2').view(np.uint8).reshape(count, 64)
                blk[:, 66:74] = scale_bytes
                blk[~live] = 0
                out[start:stop] = blk
    return out.reshape(-1)
