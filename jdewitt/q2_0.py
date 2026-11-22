"""Q2_0 (GGML raw type id 42) codec. Ported from ggml-quants.c @ master
sha256 1665bffa...3da (quantize_row_q2_0_ref, dequantize_row_q2_0) and
ggml-common.h sha256 0061131b... (block_q2_0: QK2_0 = 64, half d + 16 qs = 18 bytes).

The published gguf package (0.19.0) does not know this type, so it is keyed by
NAME / TYPE_ID, not GGMLQuantizationType.
"""
import numpy as np

NAME = 'Q2_0'
TYPE_ID = 42
BLOCK_ELEMENTS = 64
BLOCK_BYTES = 18


def quantize_q2_0(x):
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    if x.size % BLOCK_ELEMENTS:
        raise ValueError(f'Q2_0 needs a multiple of {BLOCK_ELEMENTS} values, got {x.size}')
    blocks = x.reshape(-1, BLOCK_ELEMENTS)
    d = np.abs(blocks).max(axis=1)
    with np.errstate(divide='ignore'):
        idv = np.where(d > 0, np.float32(1) / np.where(d > 0, d, np.float32(1)), np.float32(0)).astype(np.float32)
    # roundf = half away from zero; product is float32 like C
    prod = (blocks * idv[:, None]).astype(np.float32)
    a = np.abs(prod)
    fl = np.floor(a)
    r = np.where(a - fl >= np.float32(0.5), fl + 1, fl) * np.sign(prod)
    q = np.clip(r.astype(np.int64) + 1, 0, 3).astype(np.uint8)
    q4 = q.reshape(-1, BLOCK_ELEMENTS // 4, 4)
    qs = (q4[:, :, 0] | (q4[:, :, 1] << 2) | (q4[:, :, 2] << 4) | (q4[:, :, 3] << 6)).astype(np.uint8)
    out = np.empty((blocks.shape[0], BLOCK_BYTES), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = qs
    return out.reshape(-1)


def dequantize_q2_0(raw, n=None):
    raw = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray)) else np.asarray(raw, dtype=np.uint8).reshape(-1)
    blocks = raw.reshape(-1, BLOCK_BYTES)
    d = np.ascontiguousarray(blocks[:, :2]).view(np.float16).astype(np.float32).reshape(-1)
    qs = blocks[:, 2:]
    q = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(-1, BLOCK_ELEMENTS)
    out = ((q.astype(np.int32) - 1).astype(np.float32) * d[:, None]).reshape(-1)
    return out if n is None else out[:n]
