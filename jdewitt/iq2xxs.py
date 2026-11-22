# SPDX-License-Identifier: AGPL-3.0-or-later
"""IQ2_XXS encoder: iq2 data construction (256 grid, nwant=2) and the reference quantizer.

Ported from ggml-quants.c at the revision pinned in jdewitt/iq_tables.py
(iq2xs_init_impl for GGML_TYPE_IQ2_XXS, iq2_find_best_neighbour, make_qp_quants
and quantize_row_iq2_xxs_impl). Upstream asserts ``quant_weights``; the default
here is the neutral all-1.0f vector (the compiled reference harness
scripts/ggml_ref/ref_iq2xxs.c passes the same). The cached neighbour search in
the C is specified as identical to the exact one, so the exact search is ported;
the ``same candidate as the previous scale`` skip is reproduced because a repeated
candidate can re-fire the strict compare on f32 rounding. Every f32 accumulation
is an element-order left fold.
"""
from __future__ import annotations

import numpy as np

QK_K = 256
K_MAX_Q = 3
GROUP_MAX_EPS = np.float32(1e-15)
KMAP_SIZE = 43692
BLOCK_BYTES = 66
F = np.float32
_rint = np.rint

# kgrid_2bit_256 (ggml-quants.c, iq2xs_init_impl)
KGRID_2BIT_256 = np.array([
    0, 2, 5, 8, 10, 17, 20, 32, 34, 40, 42, 65, 68, 80, 88, 97,
    100, 128, 130, 138, 162, 257, 260, 272, 277, 320, 388, 408, 512, 514, 546, 642,
    1025, 1028, 1040, 1057, 1060, 1088, 1090, 1096, 1120, 1153, 1156, 1168, 1188, 1280, 1282, 1288,
    1312, 1350, 1385, 1408, 1425, 1545, 1552, 1600, 1668, 1700, 2048, 2053, 2056, 2068, 2088, 2113,
    2116, 2128, 2130, 2184, 2308, 2368, 2562, 2580, 4097, 4100, 4112, 4129, 4160, 4192, 4228, 4240,
    4245, 4352, 4360, 4384, 4432, 4442, 4480, 4644, 4677, 5120, 5128, 5152, 5157, 5193, 5248, 5400,
    5474, 5632, 5654, 6145, 6148, 6160, 6208, 6273, 6400, 6405, 6560, 6737, 8192, 8194, 8202, 8260,
    8289, 8320, 8322, 8489, 8520, 8704, 8706, 9217, 9220, 9232, 9280, 9302, 9472, 9537, 9572, 9872,
    10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
    16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
    17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
    20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
    22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
    25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
    33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
    37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
], dtype=np.uint16)

_DATA = {}


def _build():
    if 'd' in _DATA:
        return _DATA['d']
    kgrid = KGRID_2BIT_256.astype(np.int64)
    grid = np.empty((256, 8), dtype=np.int64)
    for i in range(8):
        grid[:, i] = 2 * ((kgrid >> (2 * i)) & 3) + 1
    kmap = np.full(KMAP_SIZE, -1, dtype=np.int64)
    for i in range(256):
        index = 0
        for k in range(8):
            index |= (((int(grid[i, k]) - 1) // 2) << (2 * k))
        kmap[index] = i
    missing = np.nonzero(kmap < 0)[0]
    nbr_row = np.full(KMAP_SIZE, -1, dtype=np.int64)
    nbr_row[missing] = np.arange(len(missing))
    parts, width = [], 0
    for lo in range(0, len(missing), 4096):
        tup = missing[lo:lo + 4096]
        pos = np.stack([2 * ((tup >> (2 * k)) & 3) + 1 for k in range(8)], axis=1)
        dist = ((grid[None, :, :] - pos[:, None, :]) ** 2).sum(axis=2)
        key = np.sort(dist * 256 + np.arange(256)[None, :], axis=1)
        d, j = key >> 8, key & 255
        gt = d > d[:, :1]
        has = gt.any(axis=1)
        second = np.where(has, d[np.arange(len(tup)), np.argmax(gt, axis=1)], d[:, :1].ravel())
        include = d <= second[:, None]
        cnt = include.sum(axis=1)
        width = max(width, int(cnt.max()))
        parts.append(np.where(include, j, -1))
    padded = np.concatenate([p[:, :width] if p.shape[1] >= width else p for p in parts])
    padded = padded[:, :width]
    _DATA['d'] = (grid, kmap, nbr_row, padded)
    return _DATA['d']


def _nearest_int(v):
    """nearest_int(): RNE via the 1.5*2^23 trick; upstream asserts |v| <= 4194303."""
    if v.size and not (np.abs(v) <= F(4194303.0)).all():
        raise ValueError('IQ2_XXS: nearest_int argument beyond 4194303 (assert upstream)')
    return _rint(v).astype(np.int64)


def _make_qp_quants(x, w, nmax):
    """make_qp_quants for n=32 rows (R,32) with max >= eps. Returns (scale (R,), L (R,32) as uint8 values)."""
    r = x.shape[0]
    nm = F(nmax)
    mx = x.max(axis=1)
    iscale = (nm / mx).astype(F)
    L = _nearest_int(iscale[:, None] * x) & 255
    scale = (F(1.0) / iscale).astype(F)
    best_mse = np.zeros(r, dtype=F)
    for i in range(32):
        diff = (x[:, i] - scale * L[:, i].astype(F)).astype(F)
        best_mse = (best_mse + w[:, i] * diff * diff).astype(F)
    for is_ in range(-4, 5):
        if is_ == 0:
            continue
        iscale_is = ((F(0.1) * F(is_) + nm) / mx).astype(F)
        scale_is = (F(1.0) / iscale_is).astype(F)
        mse = np.zeros(r, dtype=F)
        for i in range(32):
            lv_ = np.minimum(nmax, _nearest_int(iscale_is * x[:, i]))
            diff = (x[:, i] - scale_is * lv_.astype(F)).astype(F)
            mse = (mse + w[:, i] * diff * diff).astype(F)
        better = mse < best_mse
        best_mse = np.where(better, mse, best_mse).astype(F)
        iscale = np.where(better, iscale_is, iscale).astype(F)
    sumlx = np.zeros(r, dtype=F)
    suml2 = np.zeros(r, dtype=F)
    L = np.empty((r, 32), dtype=np.int64)
    for i in range(32):
        lv_ = np.minimum(nmax, _nearest_int(iscale * x[:, i]))
        L[:, i] = lv_ & 255
        lf = lv_.astype(F)
        sumlx = (sumlx + w[:, i] * x[:, i] * lf).astype(F)
        suml2 = (suml2 + w[:, i] * lf * lf).astype(F)
    live = np.ones(r, dtype=bool)
    for _ in range(5):
        changed = np.zeros(r, dtype=np.int64)
        for i in range(32):
            wi, xi = w[:, i], x[:, i]
            Lf = L[:, i].astype(F)
            slx = (sumlx - wi * xi * Lf).astype(F)
            sl2 = (suml2 - wi * Lf * Lf).astype(F)
            cond = live & (slx > 0) & (sl2 > 0)
            if not cond.any():
                continue
            safe = np.where(cond, slx, F(1.0))
            new_l = np.where(cond, np.minimum(nmax, _nearest_int(
                np.where(cond, (xi * sl2 / safe).astype(F), F(0.0)))), 0)
            diff = cond & (new_l != L[:, i])
            if not diff.any():
                continue
            nl = new_l.astype(F)
            slx2 = (slx + wi * xi * nl).astype(F)
            sl22 = (sl2 + wi * nl * nl).astype(F)
            take = diff & ((slx2 * slx2 * suml2) > (sumlx * sumlx * sl22))
            L[:, i] = np.where(take, new_l & 255, L[:, i])
            sumlx = np.where(take, slx2, sumlx).astype(F)
            suml2 = np.where(take, sl22, suml2).astype(F)
            changed += take
        live = live & (changed > 0)
        if not live.any():
            break
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(suml2 > 0, sumlx / np.where(suml2 > 0, suml2, F(1.0)), F(0.0)).astype(F)
    return out, L


def _pick(tables, Lg, xv, wa, scale):
    """Per group (G,8) quantised levels -> (grid index (G,), levels (G,8)); neighbour search for off-grid tuples.

    Lg: (G,8) int levels 0..2; xv, wa: (G,8) f32; scale: (G,) f32."""
    grid, kmap, nbr_row, padded = tables
    u = np.zeros(Lg.shape[0], dtype=np.int64)
    for i in range(8):
        u |= Lg[:, i] << (2 * i)
    gi = kmap[u].copy()
    codes = np.empty(Lg.shape, dtype=np.int64)
    on = gi >= 0
    if on.any():
        codes[on] = (grid[gi[on]] - 1) // 2
    miss = ~on
    if miss.any():
        idxs = padded[nbr_row[u[miss]]]
        valid = idxs >= 0
        pg = grid[np.where(valid, idxs, 0)].astype(F)
        sc = scale[miss][:, None]
        xm, wm = xv[miss], wa[miss]
        d2 = np.zeros(idxs.shape, dtype=F)
        for i in range(8):
            diff = (sc * pg[:, :, i] - xm[:, i:i + 1]).astype(F)
            d2 = (d2 + wm[:, i:i + 1] * diff * diff).astype(F)
        d2 = np.where(valid, d2, F(np.inf))
        chosen = np.take_along_axis(idxs, np.argmin(d2, axis=1)[:, None], axis=1)[:, 0]
        gi[miss] = chosen
        codes[miss] = (grid[chosen] - 1) // 2
    return gi, codes


def _fold_sums(weight, xval, L):
    q = (2 * L + 1).astype(F)
    sumqx = np.zeros(L.shape[0], dtype=F)
    sumq2 = np.zeros(L.shape[0], dtype=F)
    for i in range(32):
        w = weight[:, i]
        sumqx = (sumqx + w * xval[:, i] * q[:, i]).astype(F)
        sumq2 = (sumq2 + w * q[:, i] * q[:, i]).astype(F)
    return sumqx, sumq2


def _quantize_subblocks(weight, waux, xval, tables):
    """Active 32-wide sub-blocks (max >= eps). Returns scale (R,), L (R,32), flip (R,) bool, gi (R,4)."""
    r = xval.shape[0]
    scale, _ = _make_qp_quants(xval, weight, K_MAX_Q + 1)
    L = np.zeros((r, 32), dtype=np.int64)
    ok = (scale * F(K_MAX_Q)) > 0  # eff_max > 0
    eff_max = (scale * F(K_MAX_Q)).astype(F)
    best = np.zeros(r, dtype=F)
    prev = None
    xg, wg = xval.reshape(r * 4, 8), waux.reshape(r * 4, 8)
    for is_ in range(-6, 7):
        with np.errstate(divide='ignore', invalid='ignore'):
            id_ = ((F(2 * K_MAX_Q - 1) + F(is_) * F(0.1)) / np.where(ok, eff_max, F(1.0))).astype(F)
            this_scale = (F(1.0) / id_).astype(F)
        lq = np.clip(_nearest_int(F(0.5) * (id_[:, None] * xval - F(1.0))), 0, K_MAX_Q - 1)
        _, codes = _pick(tables, lq.reshape(r * 4, 8), xg, wg, np.repeat(this_scale, 4))
        Laux = codes.reshape(r, 32)
        skip = np.zeros(r, dtype=bool) if prev is None else (Laux == prev).all(axis=1)
        prev = Laux
        sumqx, sumq2 = _fold_sums(weight, xval, Laux)
        upd = ok & ~skip & (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        if upd.any():
            ns = (sumqx / np.where(sumq2 > 0, sumq2, F(1.0))).astype(F)
            scale = np.where(upd, ns, scale).astype(F)
            best = np.where(upd, (ns * sumqx).astype(F), best).astype(F)
            L = np.where(upd[:, None], Laux, L)
    pos = ok & (scale > 0)
    if pos.any():
        id_ = (F(1.0) / np.where(pos, scale, F(1.0))).astype(F)
        lq = np.clip(_nearest_int(F(0.5) * (id_[:, None] * xval - F(1.0))), 0, K_MAX_Q - 1)
        _, codes = _pick(tables, lq.reshape(r * 4, 8), xg, wg, np.repeat(scale, 4))
        Lnew = codes.reshape(r, 32)
        L = np.where(pos[:, None], Lnew, L)
        sumqx, sumq2 = _fold_sums(weight, xval, L)
        ns = (sumqx / np.where(sumq2 > 0, sumq2, F(1.0))).astype(F)
        scale = np.where(pos & (sumq2 > 0), ns, scale).astype(F)
    flip = ok & (scale < 0)
    scale = np.where(flip, -scale, scale).astype(F)
    # final grid indices (upstream aborts if a point is off-grid)
    gi = np.zeros((r, 4), dtype=np.int64)
    if ok.any():
        Lg = L.reshape(r * 4, 8)
        u = np.zeros(r * 4, dtype=np.int64)
        for i in range(8):
            u |= Lg[:, i] << (2 * i)
        g = tables[1][u]
        if (g[np.repeat(ok, 4)] < 0).any():
            raise ValueError('IQ2_XXS: point not on the grid after selection')
        gi = g.reshape(r, 4)
    return scale, flip, gi, ok


def quantize_iq2_xxs(x, quant_weights=None):
    """Byte-reproducible mirror of quantize_row_iq2_xxs_impl (66-byte blocks).

    quant_weights: per-element importance, same length as x; None means the neutral
    all-1.0f vector (upstream requires weights; the C harness passes the same)."""
    x = np.ascontiguousarray(x, dtype=F).reshape(-1, QK_K)
    nb = x.shape[0]
    if quant_weights is None:
        qw = np.ones_like(x)
    else:
        qw = np.ascontiguousarray(quant_weights, dtype=F).reshape(-1, QK_K)
        if qw.shape != x.shape:
            raise ValueError('quant_weights must match x')
    tables = _build()
    out = np.zeros((nb, BLOCK_BYTES), dtype=np.uint8)
    chunk = 256
    for start in range(0, nb, chunk):
        stop = min(start + chunk, nb)
        n = stop - start
        xb, wb = x[start:stop], qw[start:stop]
        sumx2 = np.zeros(n, dtype=F)
        for i in range(QK_K):
            sumx2 = (sumx2 + xb[:, i] * xb[:, i]).astype(F)
        sigma2 = (sumx2 / F(QK_K)).astype(F)
        xs = xb.reshape(n * 8, 32)
        sig = np.repeat(sigma2, 8)
        weight = (wb.reshape(n * 8, 32) * np.sqrt(sig[:, None] + xs * xs)).astype(F)
        waux = np.sqrt(weight).astype(F)
        xval = np.empty_like(xs)
        signs = np.zeros((n * 8, 4), dtype=np.int64)
        for k in range(4):
            seg = xs[:, 8 * k:8 * k + 8]
            neg = seg < 0
            xv = np.where(neg, -seg, seg).astype(F)
            s = np.zeros(n * 8, dtype=np.int64)
            for i in range(8):
                s |= neg[:, i].astype(np.int64) << i
            odd = neg.sum(axis=1) % 2 == 1
            wseg = weight[:, 8 * k:8 * k + 8]
            ax = (wseg * seg * seg).astype(F)
            imin = np.argmin(ax, axis=1)  # first strict minimum
            rows = np.arange(n * 8)
            xv[rows, imin] = np.where(odd, -xv[rows, imin], xv[rows, imin])
            s = np.where(odd, s ^ (1 << imin), s)
            xval[:, 8 * k:8 * k + 8] = xv
            signs[:, k] = s & 127
        mx = xval.max(axis=1)
        act = np.nonzero(mx >= GROUP_MAX_EPS)[0]
        scales = np.zeros(n * 8, dtype=F)
        q2 = np.zeros((n * 8, 2), dtype=np.uint64)
        if len(act):
            sc, flip, gi, ok = _quantize_subblocks(weight[act], waux[act], xval[act], tables)
            sg = np.where(flip[:, None], ~signs[act] & 127, signs[act])
            lo = np.zeros(len(act), dtype=np.uint64)
            hi = np.zeros(len(act), dtype=np.uint64)
            for k in range(4):
                lo |= gi[:, k].astype(np.uint64) << np.uint64(8 * k)
                hi |= sg[:, k].astype(np.uint64) << np.uint64(7 * k)
            keep = ok
            sc = np.where(keep, sc, F(0.0)).astype(F)
            scales[act] = sc
            q2[act, 0] = np.where(keep, lo, np.uint64(0))
            q2[act, 1] = np.where(keep, hi, np.uint64(0))
        scales = scales.reshape(n, 8)
        q2 = q2.reshape(n, 16)
        max_scale = scales.max(axis=1)
        live = max_scale > 0
        if live.any():
            d = (max_scale / F(31.0)).astype(F)
            dh = d.astype(np.float16)
            id_ = (F(1.0) / np.where(live, d, F(1.0))).astype(F)
            lv = np.clip(_nearest_int(F(0.5) * (id_[:, None] * scales - F(1.0))), 0, 15)
            for ib in range(8):
                q2[:, 2 * ib + 1] |= lv[:, ib].astype(np.uint64) << np.uint64(28)
            q2 = np.where(live[:, None], q2, np.uint64(0)).astype(np.uint32)
            blk = out[start:stop]
            dbytes = np.ascontiguousarray(dh).view(np.uint16).view(np.uint8).reshape(n, 2)
            qbytes = np.ascontiguousarray(q2.astype('<u4')).view(np.uint8).reshape(n, 64)
            blk[live, 0:2] = dbytes[live]
            blk[live, 2:66] = qbytes[live]
    return out.reshape(-1)
