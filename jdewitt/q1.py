# SPDX-License-Identifier: AGPL-3.0-or-later
"""GGML Q1_0 codec: 128-element blocks, f16 scale d = mean(|x|), one sign bit per weight.

Port of quantize_row_q1_0_ref / dequantize_row_q1_0 (ggml-quants.c master sha256
1665bffa...). Despite the 1-bit-ternary label it is binary: bit j (LSB first within
byte j//8) is set iff x >= 0 (so +0.0 and -0.0 both map to +d; NaN maps to -d),
and dequant is +d / -d (never 0). Block = 2-byte f16 d + 16 bytes of bits = 18 bytes.
"""
import numpy as np

QK1_0 = 128
BLOCK_BYTES = 18


def quantize_q1_0(x):
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    if x.size % QK1_0:
        raise ValueError(f'Q1_0 needs a multiple of {QK1_0} values, got {x.size}')
    nb = x.size // QK1_0
    blocks = x.reshape(nb, QK1_0)
    # C: sequential float32 left fold of fabsf; cumsum is a sequential fold.
    sum_abs = np.cumsum(np.abs(blocks), axis=1, dtype=np.float32)[:, -1]
    with np.errstate(over='ignore', invalid='ignore'):
        d = (sum_abs / np.float32(QK1_0)).astype(np.float16)
    bits = (blocks >= 0).astype(np.uint8)
    out = np.empty((nb, BLOCK_BYTES), dtype=np.uint8)
    out[:, 0:2] = d.reshape(nb, 1).view(np.uint8).reshape(nb, 2)
    out[:, 2:] = np.packbits(bits, axis=1, bitorder='little')
    return out.reshape(-1)


def dequantize_q1_0(raw, n):
    raw = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw.reshape(-1).view(np.uint8)
    nb = raw.size // BLOCK_BYTES
    blocks = raw[:nb * BLOCK_BYTES].reshape(nb, BLOCK_BYTES)
    d = np.ascontiguousarray(blocks[:, 0:2]).view(np.float16).astype(np.float32).reshape(nb)
    # hardware f16->f32 (F16C) quiets signalling NaNs; numpy's software path keeps bit 22 clear
    d = np.where(np.isnan(d), (d.view(np.uint32) | np.uint32(0x00400000)).view(np.float32), d)
    bits = np.unpackbits(blocks[:, 2:], axis=1, bitorder='little').astype(bool)
    y = np.where(bits, d[:, None], -d[:, None]).astype(np.float32)
    return y.reshape(-1)[:n]
