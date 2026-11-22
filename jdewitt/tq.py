"""TQ1_0 / TQ2_0 ternary codecs, byte-exact ports of ggml-quants.c @ master
(sha256 1665bffa...a3da): quantize_row_tq{1,2}_0_ref, dequantize_row_tq{1,2}_0.

Block layouts: TQ1_0 = qs[48] + qh[4] + f16 d (54 B); TQ2_0 = qs[64] + f16 d (66 B).
Scale is d = amax (f32), id = 1/d in f32, trit = lroundf(x*id)+1 (f32 product).
"""

from __future__ import annotations

import numpy as np

QK_K = 256
TQ1_0_BLOCK_BYTES = 54
TQ2_0_BLOCK_BYTES = 66


def _blocks(x):
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    if x.size % QK_K:
        raise ValueError('TQ quantization needs a multiple of 256 values')
    return x.reshape(-1, QK_K)


def _scale_and_trits(xb):
    """Return (d_f16 as uint16 [nb], xi int64 [nb,256]) = lroundf(x*id)+1."""
    with np.errstate(all='ignore'):
        amax = np.zeros(xb.shape[0], dtype=np.float32)
        for j in range(QK_K):  # C: amax = MAX(amax, fabsf(v)); NaN-faithful
            v = np.abs(xb[:, j])
            amax = np.where(amax > v, amax, v).astype(np.float32)
        d = amax
        one = np.float32(1.0)
        idv = np.where(d != 0, one / d, np.float32(0.0)).astype(np.float32)
        d16 = d.astype(np.float16).view(np.uint16)
        prod = (xb * idv[:, None]).astype(np.float32)
        # lroundf: round half away from zero (exact in f32/f64)
        p = prod.astype(np.float64)
        t = np.trunc(p)
        r = t + np.where(np.abs(p - t) >= 0.5, np.sign(p), 0.0)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)  # UB in C; unreachable for finite sane input
    return d16, r.astype(np.int64) + 1


def _ceil243(q):
    return ((q.astype(np.int64) * 256 + 242) // 243).astype(np.uint8)


def quantize_tq1_0(x):
    xb = _blocks(x)
    nb = xb.shape[0]
    d16, xi = _scale_and_trits(xb)
    out = np.zeros((nb, TQ1_0_BLOCK_BYTES), dtype=np.uint8)
    # trit order is the C loop order; uint8 wraparound as in C
    def pack(cols):  # cols: list of [nb] int arrays, first is most significant
        q = np.zeros(nb, dtype=np.int64)
        for c in cols:
            q = ((q * 3) & 255)
            q = (q + c) & 255
        return q
    # qs[0:32]: x[m + n*32]
    for m in range(32):
        out[:, m] = _ceil243(pack([xi[:, m + n * 32] for n in range(5)]))
    # qs[32:48]: x[160 + m + n*16]
    for m in range(16):
        out[:, 32 + m] = _ceil243(pack([xi[:, 160 + m + n * 16] for n in range(5)]))
    # qh[0:4]: x[240 + j + m*4], then one extra *3
    for j in range(4):
        q = pack([xi[:, 240 + j + m * 4] for m in range(4)])
        out[:, 48 + j] = _ceil243((q * 3) & 255)
    out[:, 52:54] = d16.astype('<u2').view(np.uint8).reshape(nb, 2)
    return out.reshape(-1)


def quantize_tq2_0(x):
    xb = _blocks(x)
    nb = xb.shape[0]
    d16, xi = _scale_and_trits(xb)
    out = np.zeros((nb, TQ2_0_BLOCK_BYTES), dtype=np.uint8)
    for j in range(0, 64, 32):
        base = j * 4
        for m in range(32):
            q = np.zeros(nb, dtype=np.int64)
            for n in range(4):
                q += (xi[:, base + m + n * 32] & 3) << (2 * n)
            out[:, j + m] = (q & 255).astype(np.uint8)
    out[:, 64:66] = d16.astype('<u2').view(np.uint8).reshape(nb, 2)
    return out.reshape(-1)


def _split(raw, size):
    raw = np.frombuffer(bytes(raw) if not isinstance(raw, np.ndarray) else raw.tobytes(), dtype=np.uint8)
    if raw.size % size:
        raise ValueError('byte length is not a multiple of the block size')
    return raw.reshape(-1, size)


def _f16_scale(blk, off):
    return np.ascontiguousarray(blk[:, off:off + 2]).view('<f2').reshape(-1).astype(np.float32)


def _trit(q, n):
    qq = (q.astype(np.uint16) * np.uint16([1, 3, 9, 27, 81, 243][n])) & 255  # uint8 wrap
    return (((qq.astype(np.uint32) * 3) >> 8).astype(np.int16) - 1).astype(np.float32)


def dequantize_tq1_0(raw):
    blk = _split(raw, TQ1_0_BLOCK_BYTES)
    nb = blk.shape[0]
    d = _f16_scale(blk, 52)
    out = np.empty((nb, QK_K), dtype=np.float32)
    pos = 0
    for j, width in ((0, 32), (32, 16)):
        for n in range(5):
            out[:, pos:pos + width] = _trit(blk[:, j:j + width], n)
            pos += width
    for n in range(4):
        out[:, pos:pos + 4] = _trit(blk[:, 48:52], n)
        pos += 4
    with np.errstate(all='ignore'):
        return (out * d[:, None]).astype(np.float32).reshape(-1)


def dequantize_tq2_0(raw):
    blk = _split(raw, TQ2_0_BLOCK_BYTES)
    nb = blk.shape[0]
    d = _f16_scale(blk, 64)
    out = np.empty((nb, QK_K), dtype=np.float32)
    for j in range(0, 64, 32):
        for lane in range(4):
            q = ((blk[:, j:j + 32] >> (2 * lane)) & 3).astype(np.int16) - 1
            out[:, j * 4 + lane * 32: j * 4 + lane * 32 + 32] = q.astype(np.float32)
    with np.errstate(all='ignore'):
        return (out * d[:, None]).astype(np.float32).reshape(-1)
